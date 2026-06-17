#!/usr/bin/env python3
"""Compute an AbAgym-style -RSA/ASA baseline from official modeled complexes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley


MAX_ASA = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0, "CYS": 148.0,
    "GLN": 214.0, "GLU": 214.0, "GLY": 97.0, "HIS": 216.0, "ILE": 195.0,
    "LEU": 191.0, "LYS": 230.0, "MET": 203.0, "PHE": 228.0, "PRO": 154.0,
    "SER": 143.0, "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0,
}


def chain_ids(value: object) -> list[str]:
    return sorted(set(re.findall(r"[A-Za-z0-9]", str(value))))


def parse_site(value: object) -> tuple[int, str]:
    match = re.match(r"(-?\d+)([A-Za-z]?)", str(value))
    if not match:
        raise ValueError(f"Cannot parse site {value}")
    return int(match.group(1)), match.group(2).strip().upper()


def locate_residue(model: object, chains: list[str], site: object):
    number, insertion = parse_site(site)
    insertion = insertion or " "
    for chain_id in chains:
        if chain_id not in model:
            continue
        for residue in model[chain_id].get_residues():
            if residue.id[0] == " " and residue.id[1] == number and residue.id[2].strip().upper() == insertion.strip().upper():
                return residue
    raise KeyError(f"Cannot resolve {chains} {site}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--structure-dir", type=Path, required=True)
    ap.add_argument("--structure-id-column", default="PDB_file")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.records)
    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()
    models = {}
    for path in args.structure_dir.rglob("*.pdb"):
        structure = parser.get_structure(path.stem, str(path))
        sr.compute(structure, level="R")
        models[path.stem.lower()] = next(structure.get_models())
    asa_cache = {}
    rows = []
    for row in records.itertuples(index=False):
        structure_id = str(getattr(row, args.structure_id_column, row.PDB_file)).lower()
        key = (structure_id, str(row.chains), str(row.site))
        if key not in asa_cache:
            if structure_id not in models:
                raise KeyError(f"Cannot find structure {structure_id} from column {args.structure_id_column}")
            residue = locate_residue(models[structure_id], chain_ids(row.chains), row.site)
            asa = float(getattr(residue, "sasa", 0.0))
            rsa = asa / MAX_ASA.get(residue.resname, np.nan)
            asa_cache[key] = (asa, rsa)
        asa, rsa = asa_cache[key]
        rows.append(
            {
                "sample_id": row.sample_id,
                "DMS_name": row.DMS_name,
                "antigen_name": row.antigen_name,
                "DMS_on": row.DMS_on,
                "DMS_score": row.DMS_score,
                "rank_target": row.rank_target,
                "asa": asa,
                "rsa": rsa,
                "prediction": -asa,
                "prediction_negative_rsa": -rsa,
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(args.out_dir / "predictions.csv", index=False)
    audit = {
        "records": int(len(output)),
        "resolved_site_contexts": int(len(asa_cache)),
        "structure_id_column": args.structure_id_column,
        "prediction": "-absolute residue SASA in complex; larger score means less exposed/more buried",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
