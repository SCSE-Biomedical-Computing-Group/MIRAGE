#!/usr/bin/env python3
"""Build per-mutation template-structure features for AbAgym ranking."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1


AA_PROP = {
    "A": (1.8, 0.0, 0.0, 0.0, 88.6), "C": (2.5, 0.0, 0.0, 0.0, 108.5),
    "D": (-3.5, -1.0, 0.0, 1.0, 111.1), "E": (-3.5, -1.0, 0.0, 1.0, 138.4),
    "F": (2.8, 0.0, 1.0, 0.0, 189.9), "G": (-0.4, 0.0, 0.0, 0.0, 60.1),
    "H": (-3.2, 0.5, 1.0, 1.0, 153.2), "I": (4.5, 0.0, 0.0, 0.0, 166.7),
    "K": (-3.9, 1.0, 0.0, 1.0, 168.6), "L": (3.8, 0.0, 0.0, 0.0, 166.7),
    "M": (1.9, 0.0, 0.0, 0.0, 162.9), "N": (-3.5, 0.0, 0.0, 1.0, 114.1),
    "P": (-1.6, 0.0, 0.0, 0.0, 112.7), "Q": (-3.5, 0.0, 0.0, 1.0, 143.8),
    "R": (-4.5, 1.0, 0.0, 1.0, 173.4), "S": (-0.8, 0.0, 0.0, 1.0, 89.0),
    "T": (-0.7, 0.0, 0.0, 1.0, 116.1), "V": (4.2, 0.0, 0.0, 0.0, 140.0),
    "W": (-0.9, 0.0, 1.0, 0.0, 227.8), "Y": (-1.3, 0.0, 1.0, 1.0, 193.6),
}
ZERO = np.zeros(5, dtype=np.float32)


def properties(aa: str) -> np.ndarray:
    return np.asarray(AA_PROP.get(aa, ZERO), dtype=np.float32)


def chain_ids(value: object) -> list[str]:
    return sorted(set(re.findall(r"[A-Za-z0-9]", str(value))))


def parse_site(value: object) -> tuple[int, str]:
    match = re.match(r"(-?\d+)([A-Za-z]?)", str(value))
    if not match:
        raise ValueError(f"Cannot parse residue site {value}")
    return int(match.group(1)), match.group(2).strip().upper()


def parser_for(path: Path):
    if path.suffix == ".cif":
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def locate_residue(chain: object, number: int, insertion: str):
    insertion = insertion or " "
    for residue in chain.get_residues():
        if residue.id[1] == number and residue.id[2].strip().upper() == insertion.strip().upper():
            return residue
    if insertion.strip():
        for residue in chain.get_residues():
            if residue.id[1] == number:
                return residue
    return None


def heavy_coordinates(residue: object) -> np.ndarray:
    coords = [atom.coord for atom in residue.get_atoms() if not atom.element.upper().startswith("H")]
    return np.asarray(coords, dtype=np.float32)


def minimum_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0:
        return float("inf")
    return float(np.sqrt(((first[:, None, :] - second[None, :, :]) ** 2).sum(axis=2)).min())


def residue_aa(residue: object) -> str:
    return seq1(residue.resname, custom_map={"MSE": "M"}) if residue is not None else "X"


FEATURE_NAMES = [
    "structure_residue_found",
    "mutated_chain_copies_found",
    "partner_min_heavy_distance",
    "partner_contacts_le_4",
    "partner_contacts_le_6",
    "partner_contacts_le_8",
    "local_ca_neighbors_le_8",
    "local_ca_neighbors_le_12",
    "mutated_residue_bfactor",
] + [f"closest_partner_property_{index}" for index in range(5)] + [
    f"mutation_partner_property_interaction_{index}" for index in range(5)
]


def structural_features(row: object, model: object) -> np.ndarray:
    mutated_chains = chain_ids(row.chains)
    partner_chains = chain_ids(row.Antibody_Chains if str(row.DMS_on) == "antigen" else row.Antigen_Chains)
    number, insertion = parse_site(row.site)
    all_residues = [residue for chain in model for residue in chain.get_residues() if "CA" in residue]
    found = []
    for chain_id in mutated_chains:
        if chain_id in model:
            residue = locate_residue(model[chain_id], number, insertion)
            if residue is not None:
                found.append(residue)
    partner = [
        residue
        for chain_id in partner_chains
        if chain_id in model
        for residue in model[chain_id].get_residues()
        if residue.id[0] == " "
    ]
    if not found or not partner:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    partner_coords = [(residue, heavy_coordinates(residue)) for residue in partner]
    closest_residue = None
    closest_distance = float("inf")
    contact_counts = np.zeros(3, dtype=np.float32)
    local_counts = np.zeros(2, dtype=np.float32)
    bfactors = []
    for residue in found:
        coords = heavy_coordinates(residue)
        distances = [(candidate, minimum_distance(coords, candidate_coords)) for candidate, candidate_coords in partner_coords]
        contact_counts += np.asarray([sum(distance <= cutoff for _, distance in distances) for cutoff in (4.0, 6.0, 8.0)])
        candidate, distance = min(distances, key=lambda item: item[1])
        if distance < closest_distance:
            closest_distance = distance
            closest_residue = candidate
        ca = residue["CA"].coord if "CA" in residue else None
        if ca is not None:
            ca_distances = [
                float(np.linalg.norm(ca - other["CA"].coord))
                for other in all_residues
                if other is not residue
            ]
            local_counts += np.asarray([sum(distance <= cutoff for distance in ca_distances) for cutoff in (8.0, 12.0)])
        bfactors.append(float(np.mean([atom.bfactor for atom in residue.get_atoms()])))
    divisor = float(len(found))
    closest_prop = properties(residue_aa(closest_residue))
    mutant_delta = properties(str(row.mutation)) - properties(str(row.wildtype))
    return np.asarray(
        [1.0, divisor, closest_distance]
        + (contact_counts / divisor).tolist()
        + (local_counts / divisor).tolist()
        + [float(np.mean(bfactors))]
        + closest_prop.tolist()
        + (mutant_delta * closest_prop).tolist(),
        dtype=np.float32,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--template-dir", type=Path, required=True)
    ap.add_argument("--structure-id-column", default="template_PDB_ID")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records = pd.read_csv(args.records)
    available = {path.stem.lower(): path for path in args.template_dir.rglob("*") if path.suffix.lower() in {".pdb", ".cif"}}
    structures = {}
    for structure_id in sorted(records[args.structure_id_column].astype(str).unique()):
        path = available.get(structure_id.lower())
        if path is None:
            raise FileNotFoundError(f"Missing structure for {structure_id}")
        structures[structure_id] = next(parser_for(path).get_structure(structure_id, str(path)).get_models())
    normalized = records.rename(columns={"Antibody Chains": "Antibody_Chains", "Antigen Chains": "Antigen_Chains"})
    rows = []
    geometry_cache = {}
    for row in normalized.itertuples(index=False):
        cache_key = (
            str(getattr(row, args.structure_id_column)),
            str(row.chains),
            str(row.site),
            str(row.DMS_on),
            str(row.Antibody_Chains),
            str(row.Antigen_Chains),
        )
        if cache_key not in geometry_cache:
            geometry_cache[cache_key] = structural_features(
                row, structures[str(getattr(row, args.structure_id_column))]
            )
        values = geometry_cache[cache_key].copy()
        closest_prop = values[-10:-5]
        values[-5:] = (properties(str(row.mutation)) - properties(str(row.wildtype))) * closest_prop
        rows.append(values)
    x = np.asarray(rows, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, names=np.asarray(FEATURE_NAMES))
    audit = {
        "records": int(len(records)),
        "templates": int(len(structures)),
        "structure_id_column": args.structure_id_column,
        "features": len(FEATURE_NAMES),
        "resolved_fraction": float(x[:, 0].mean()),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
