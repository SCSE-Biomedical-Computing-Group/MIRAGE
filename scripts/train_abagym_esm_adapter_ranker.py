#!/usr/bin/env python3
"""Trainable ESM2 adapter-style ranker for AbAgym grouped mutation ranking."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: idx for idx, aa in enumerate(AA)}
AA_PROP = {
    "A": (1.8, 0.0, 0.0, 0.0, 88.6), "C": (2.5, 0.0, 0.0, 0.0, 108.5),
    "D": (-3.5, -1.0, 0.0, 1.0, 111.1), "E": (-3.5, -1.0, 0.0, 1.0, 138.4),
    "F": (2.8, 0.0, 1.0, 0.0, 189.9), "G": (-0.4, 0.0, 0.0, 0.0, 60.1),
    "H": (-3.2, 0.5, 1.0, 1.0, 153.2), "I": (4.5, 0.0, 0.0, 0.0, 166.7),
    "K": (-3.9, 1.0, 0.0, 0.0, 168.6), "L": (3.8, 0.0, 0.0, 0.0, 166.7),
    "M": (1.9, 0.0, 0.0, 0.0, 162.9), "N": (-3.5, 0.0, 0.0, 1.0, 114.1),
    "P": (-1.6, 0.0, 0.0, 0.0, 112.7), "Q": (-3.5, 0.0, 0.0, 1.0, 143.8),
    "R": (-4.5, 1.0, 0.0, 1.0, 173.4), "S": (-0.8, 0.0, 0.0, 1.0, 89.0),
    "T": (-0.7, 0.0, 0.0, 1.0, 116.1), "V": (4.2, 0.0, 0.0, 0.0, 140.0),
    "W": (-0.9, 0.0, 1.0, 0.0, 227.8), "Y": (-1.3, 0.0, 1.0, 1.0, 193.6),
}


def chain_ids(value: object) -> list[str]:
    return sorted(set(re.findall(r"[A-Za-z0-9]", str(value))))


def parse_site(value: object) -> tuple[int, str]:
    match = re.match(r"(-?\d+)([A-Za-z]?)", str(value))
    if not match:
        raise ValueError(f"Cannot parse site {value}")
    return int(match.group(1)), match.group(2).strip().upper()


def site_number(value: object) -> float:
    match = re.search(r"-?\d+", str(value))
    return float(match.group()) if match else 0.0


def chain_sequence_and_position(model: object, mutated_chains: list[str], site: object) -> tuple[str, int]:
    number, insertion = parse_site(site)
    insertion = insertion or " "
    for chain_id in mutated_chains:
        if chain_id not in model:
            continue
        residues = [residue for residue in model[chain_id].get_residues() if residue.id[0] == " "]
        for index, residue in enumerate(residues):
            if residue.id[1] == number and residue.id[2].strip().upper() == insertion.strip().upper():
                sequence = "".join(seq1(item.resname, custom_map={"MSE": "M"}) for item in residues)
                return sequence, index
    raise KeyError(f"Could not resolve chain/site {mutated_chains} {site}")


def centered_crop(sequence: str, position: int, maximum_residues: int) -> tuple[str, int]:
    if len(sequence) <= maximum_residues:
        return sequence, position
    start = max(0, min(position - maximum_residues // 2, len(sequence) - maximum_residues))
    return sequence[start : start + maximum_residues], position - start


def mutate_sequence(sequence: str, position: int, mutation: str) -> str:
    if position < 0 or position >= len(sequence):
        raise IndexError(position)
    mutation = mutation if mutation in AA else "X"
    return sequence[:position] + mutation + sequence[position + 1 :]


def load_models(structure_dir: Path, records: pd.DataFrame) -> dict[str, object]:
    parser = PDBParser(QUIET=True)
    available = {path.stem: path for path in structure_dir.rglob("*.pdb")}
    models = {}
    for pdb_file in sorted(records["PDB_file"].astype(str).unique()):
        path = available.get(pdb_file)
        if path is None:
            raise FileNotFoundError(f"Missing structure for {pdb_file}")
        models[pdb_file] = next(parser.get_structure(pdb_file, str(path)).get_models())
    return models


def aa_features(residue: str) -> tuple[np.ndarray, np.ndarray]:
    one_hot = np.zeros(len(AA), dtype=np.float32)
    if residue in AA_INDEX:
        one_hot[AA_INDEX[residue]] = 1.0
    return one_hot, np.asarray(AA_PROP.get(residue, (0, 0, 0, 0, 0)), dtype=np.float32)


def scalar_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dms_types = sorted(frame["experimental_DMS_type"].astype(str).unique())
    rows = []
    for row in frame.itertuples(index=False):
        wt_onehot, wt_prop = aa_features(str(row.wildtype))
        mut_onehot, mut_prop = aa_features(str(row.mutation))
        distance = float(row.closest_interface_atom_distance)
        scalar = np.asarray(
            [
                site_number(row.site) / 200.0,
                distance / 10.0,
                np.exp(-distance / 4.0),
                float(distance <= 3.5),
                float(distance <= 5.0),
                float(distance <= 8.0),
                float(str(row.DMS_on) == "antibody"),
                float(str(row.DMS_on) == "antigen"),
            ],
            dtype=np.float32,
        )
        type_onehot = np.asarray([float(str(row.experimental_DMS_type) == item) for item in dms_types], dtype=np.float32)
        rows.append(np.concatenate([wt_onehot, mut_onehot, wt_prop, mut_prop, mut_prop - wt_prop, scalar, type_onehot]))
    names = (
        [f"wildtype_{aa}" for aa in AA] + [f"mutation_{aa}" for aa in AA]
        + [f"wildtype_property_{i}" for i in range(5)]
        + [f"mutation_property_{i}" for i in range(5)]
        + [f"property_delta_{i}" for i in range(5)]
        + ["site_number_scaled", "interface_distance_scaled", "interface_distance_decay",
           "interface_le_3p5", "interface_le_5", "interface_le_8", "mutates_antibody", "mutates_antigen"]
        + [f"dms_type_{item}" for item in dms_types]
    )
    return np.asarray(rows, dtype=np.float32), names


def load_npz(path: Path, expected: int) -> tuple[np.ndarray, list[str]]:
    payload = np.load(path, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    if len(x) < expected:
        raise ValueError(f"{path} has {len(x)} rows, expected at least {expected}")
    if len(x) > expected:
        x = x[:expected]
    return x, [str(x) for x in payload["names"].tolist()]


def build_mutant_contexts(records: pd.DataFrame, structure_dir: Path, max_residues: int) -> tuple[list[str], np.ndarray]:
    models = load_models(structure_dir, records)
    resolved: dict[tuple[str, str, str], tuple[str, int]] = {}
    sequences: list[str] = []
    positions: list[int] = []
    for row in records.itertuples(index=False):
        key = (str(row.PDB_file), str(row.chains), str(row.site))
        if key not in resolved:
            resolved[key] = chain_sequence_and_position(models[str(row.PDB_file)], chain_ids(row.chains), row.site)
        sequence, position = centered_crop(*resolved[key], maximum_residues=max_residues)
        sequences.append(mutate_sequence(sequence, position, str(row.mutation)))
        positions.append(position)
    return sequences, np.asarray(positions, dtype=np.int64)


def tokenize_contexts(tokenizer, sequences: list[str], max_residues: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_residues + 2)
    return {key: value.cpu() for key, value in encoded.items()}


class EsmAdapterRanker(nn.Module):
    def __init__(self, model_name: str, scalar_dim: int, structure_dim: int, hidden_dim: int, dropout: float, local_files_only: bool) -> None:
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        esm_dim = int(self.esm.config.hidden_size)
        self.scalar_projection = nn.Sequential(nn.Linear(scalar_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.structure_projection = nn.Sequential(nn.Linear(structure_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)) if structure_dim else None
        fusion_dim = esm_dim * 3 + hidden_dim + (hidden_dim if structure_dim else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        scalar: torch.Tensor,
        structure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        token_pos = positions + 1
        pos_repr = output[torch.arange(output.size(0), device=output.device), token_pos]
        residue_mask = attention_mask.float()
        residue_mask[:, 0] = 0.0
        residue_mask[torch.arange(residue_mask.size(0), device=output.device), attention_mask.sum(dim=1).long() - 1] = 0.0
        mean_repr = (output * residue_mask.unsqueeze(-1)).sum(dim=1) / residue_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pieces = [pos_repr, mean_repr, pos_repr - mean_repr, self.scalar_projection(scalar)]
        if self.structure_projection is not None:
            if structure is None:
                raise ValueError("structure features required")
            pieces.append(self.structure_projection(structure))
        return self.head(torch.cat(pieces, dim=1)).squeeze(-1)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def install_attention_lora(esm: nn.Module, train_last_layers: int, rank: int, alpha: float, dropout: float, targets: list[str]) -> int:
    installed = 0
    layers = esm.encoder.layer
    for layer in layers[-train_last_layers:]:
        attn = layer.attention
        if "query" in targets:
            attn.self.query = LoRALinear(attn.self.query, rank, alpha, dropout)
            installed += 1
        if "key" in targets:
            attn.self.key = LoRALinear(attn.self.key, rank, alpha, dropout)
            installed += 1
        if "value" in targets:
            attn.self.value = LoRALinear(attn.self.value, rank, alpha, dropout)
            installed += 1
        if "output" in targets:
            attn.output.dense = LoRALinear(attn.output.dense, rank, alpha, dropout)
            installed += 1
    return installed


def configure_trainable(
    model: EsmAdapterRanker,
    train_last_layers: int,
    train_embeddings: bool,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    lora_targets: str = "query,value",
) -> dict[str, int]:
    for param in model.esm.parameters():
        param.requires_grad = False
    if train_embeddings:
        for param in model.esm.embeddings.word_embeddings.parameters():
            param.requires_grad = True
    layers = model.esm.encoder.layer
    installed_lora = 0
    if lora_rank > 0:
        targets = [item.strip() for item in lora_targets.split(",") if item.strip()]
        installed_lora = install_attention_lora(model.esm, train_last_layers, lora_rank, lora_alpha, lora_dropout, targets)
    else:
        for layer in layers[-train_last_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    for module in [model.scalar_projection, model.head]:
        for param in module.parameters():
            param.requires_grad = True
    if model.structure_projection is not None:
        for param in model.structure_projection.parameters():
            param.requires_grad = True
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "installed_lora_modules": int(installed_lora),
    }


def draw_pairs(frame: pd.DataFrame, train_indices: np.ndarray, pairs_per_study: int, minimum_delta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_parts, right_parts, label_parts = [], [], []
    train = frame.loc[train_indices]
    for _, study in train.groupby("DMS_name", sort=True):
        idx = study.index.to_numpy()
        targets = study["rank_target"].to_numpy()
        if len(idx) < 2:
            continue
        lookup = dict(zip(idx, targets))
        left, right = [], []
        attempts = 0
        while len(left) < pairs_per_study and attempts < 100:
            attempts += 1
            a = rng.choice(idx, size=pairs_per_study, replace=True)
            b = rng.choice(idx, size=pairs_per_study, replace=True)
            delta = np.asarray([lookup[x] - lookup[y] for x, y in zip(a, b)])
            valid = np.abs(delta) >= minimum_delta
            left.extend(a[valid].tolist())
            right.extend(b[valid].tolist())
        if not left:
            continue
        a = np.asarray(left[:pairs_per_study], dtype=int)
        b = np.asarray(right[:pairs_per_study], dtype=int)
        labels = np.sign(frame.loc[a, "rank_target"].to_numpy() - frame.loc[b, "rank_target"].to_numpy())
        left_parts.append(a)
        right_parts.append(b)
        label_parts.append(labels.astype(np.float32))
    return np.concatenate(left_parts), np.concatenate(right_parts), np.concatenate(label_parts)


def draw_listwise_batch(frame: pd.DataFrame, train_indices: np.ndarray, studies: np.ndarray, max_items: int, rng: np.random.Generator) -> np.ndarray:
    study = str(rng.choice(studies))
    idx = frame.index[(frame.index.isin(train_indices)) & (frame["DMS_name"].astype(str) == study)].to_numpy()
    if len(idx) > max_items:
        idx = rng.choice(idx, size=max_items, replace=False)
    return np.asarray(idx, dtype=int)


def batch_to_device(batch: np.ndarray, tensors: dict[str, torch.Tensor], positions: np.ndarray, scalar: np.ndarray, structure: np.ndarray | None, device: torch.device) -> dict[str, torch.Tensor]:
    out = {
        "input_ids": tensors["input_ids"][batch].to(device, non_blocking=True),
        "attention_mask": tensors["attention_mask"][batch].to(device, non_blocking=True),
        "positions": torch.as_tensor(positions[batch], dtype=torch.long, device=device),
        "scalar": torch.as_tensor(scalar[batch], dtype=torch.float32, device=device),
    }
    if structure is not None:
        out["structure"] = torch.as_tensor(structure[batch], dtype=torch.float32, device=device)
    return out


def score_indices(model: EsmAdapterRanker, indices: np.ndarray, tensors: dict[str, torch.Tensor], positions: np.ndarray, scalar: np.ndarray, structure: np.ndarray | None, device: torch.device, batch_size: int, amp: bool) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            inputs = batch_to_device(batch, tensors, positions, scalar, structure, device)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                preds.append(model(**inputs).detach().cpu().float().numpy())
    return np.concatenate(preds)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if any(name.startswith(prefix) for prefix in ("scalar_projection", "structure_projection", "head"))
        or "lora_" in name
    }


def load_compatible_trainable_state(model: nn.Module, checkpoint_path: Path) -> dict[str, int]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    source = payload.get("trainable_state_dict", payload.get("model_state_dict", payload))
    current = model.state_dict()
    compatible = {}
    skipped = 0
    for name, value in source.items():
        if name in current and tuple(current[name].shape) == tuple(value.shape):
            compatible[name] = value
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": int(len(compatible)),
        "skipped_tensors": int(skipped),
        "missing_tensors_after_partial_load": int(len(missing)),
        "unexpected_tensors_after_partial_load": int(len(unexpected)),
    }


def save_fold_checkpoint(
    args: argparse.Namespace,
    model: EsmAdapterRanker,
    heldout: str,
    scalar: np.ndarray,
    structure: np.ndarray | None,
    scalar_scaler: StandardScaler,
    structure_scaler: StandardScaler | None,
) -> str:
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(ckpt_dir / f"holdout_{heldout}.pt")
    payload = {
        "model_name": args.model_name,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "train_last_layers": int(args.train_last_layers),
        "train_embeddings": bool(args.train_embeddings),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_targets": args.lora_targets,
        "max_residues": int(args.max_residues),
        "scalar_dim": int(scalar.shape[1]),
        "structure_dim": int(0 if structure is None else structure.shape[1]),
        "heldout_group": heldout,
        "trainable_state_dict": trainable_state_dict(model),
        "scalar_mean": scalar_scaler.mean_.astype(np.float32),
        "scalar_scale": scalar_scaler.scale_.astype(np.float32),
        "structure_mean": None if structure_scaler is None else structure_scaler.mean_.astype(np.float32),
        "structure_scale": None if structure_scaler is None else structure_scaler.scale_.astype(np.float32),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def train_one_fold(args: argparse.Namespace, frame: pd.DataFrame, tensors: dict[str, torch.Tensor], positions: np.ndarray, scalar_all: np.ndarray, structure_all: np.ndarray | None, heldout: str, rng: np.random.Generator, run=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    group_values = frame[args.holdout_column].astype(str)
    test = frame.index[group_values == heldout].to_numpy()
    train = frame.index[group_values != heldout].to_numpy()
    scalar_scaler = StandardScaler().fit(scalar_all[train])
    scalar = scalar_scaler.transform(scalar_all).astype(np.float32)
    structure = None
    if structure_all is not None:
        structure_scaler = StandardScaler().fit(structure_all[train])
        structure = structure_scaler.transform(structure_all).astype(np.float32)
    else:
        structure_scaler = None
    model = EsmAdapterRanker(
        args.model_name,
        scalar_dim=scalar.shape[1],
        structure_dim=0 if structure is None else structure.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        local_files_only=args.local_files_only,
    ).to(device)
    param_counts = configure_trainable(
        model,
        args.train_last_layers,
        args.train_embeddings,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_targets,
    )
    init_summary = None
    if args.init_checkpoint is not None:
        init_summary = load_compatible_trainable_state(model, args.init_checkpoint)
        print(f"loaded init checkpoint {args.init_checkpoint}: {init_summary}", flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    history = []
    print(f"fold holdout={heldout} train={len(train)} test={len(test)} {param_counts}", flush=True)
    train_studies = frame.loc[train, "DMS_name"].astype(str).unique()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        if args.loss == "pairwise":
            left, right, labels = draw_pairs(frame, train, args.pairs_per_study, args.minimum_rank_delta, rng)
            order = rng.permutation(len(left))
            for start in range(0, len(order), args.pair_batch_size):
                sub = order[start : start + args.pair_batch_size]
                li, ri = left[sub], right[sub]
                y = torch.as_tensor(labels[sub], dtype=torch.float32, device=device)
                left_inputs = batch_to_device(li, tensors, positions, scalar, structure, device)
                right_inputs = batch_to_device(ri, tensors, positions, scalar, structure, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    diff = model(**left_inputs) - model(**right_inputs)
                    loss = F.softplus(-y * diff).mean()
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
        elif args.loss == "listwise":
            for _ in range(args.listwise_steps_per_epoch):
                batch = draw_listwise_batch(frame, train, train_studies, args.listwise_batch_items, rng)
                inputs = batch_to_device(batch, tensors, positions, scalar, structure, device)
                target = torch.as_tensor(frame.loc[batch, "rank_target"].to_numpy(dtype=np.float32), device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    scores = model(**inputs)
                    target_dist = F.softmax(target / args.listwise_temperature, dim=0)
                    loss = -(target_dist * F.log_softmax(scores, dim=0)).sum()
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
        else:
            raise ValueError(f"unknown loss {args.loss}")
        if args.checkpoint_only:
            fold_spearman = float("nan")
        else:
            pred = score_indices(model, test, tensors, positions, scalar, structure, device, args.eval_batch_size, args.amp)
            fold_spearman = float(spearmanr(frame.loc[test, "DMS_score"], pred).statistic)
        row = {"holdout_group": heldout, "epoch": epoch + 1, "loss": float(np.mean(losses)), "fold_spearman": fold_spearman}
        history.append(row)
        print(f"{heldout} epoch {epoch + 1}/{args.epochs} loss={row['loss']:.5f} fold_spearman={fold_spearman:.4f}", flush=True)
        if run is not None:
            run.log({f"{heldout}/loss": row["loss"], f"{heldout}/spearman": fold_spearman, "epoch": epoch + 1})
    if args.checkpoint_only:
        checkpoint_path = ""
        if args.save_checkpoints:
            checkpoint_path = save_fold_checkpoint(
                args, model, heldout, scalar, structure, scalar_scaler, structure_scaler
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        empty_pred = pd.DataFrame(
            columns=["sample_id", "DMS_name", "DMS_on", "DMS_score", "rank_target", "antigen_name", "holdout_group", "prediction"]
        )
        empty_metric = pd.DataFrame(columns=["DMS_name", "DMS_on", "n", "spearman", "ndcg_top_10pct", "holdout_group"])
        return empty_pred, empty_metric, {
            "history": history,
            "checkpoint_path": checkpoint_path,
            "init_checkpoint": str(args.init_checkpoint or ""),
            "init_summary": init_summary,
            **param_counts,
        }
    pred = score_indices(model, test, tensors, positions, scalar, structure, device, args.eval_batch_size, args.amp)
    metrics = []
    predictions = []
    for study in sorted(frame.loc[test, "DMS_name"].unique()):
        positions_in_test = np.flatnonzero(frame.loc[test, "DMS_name"].to_numpy() == study)
        indices = test[positions_in_test]
        study_pred = pred[positions_in_test]
        relevance = frame.loc[indices, "rank_target"].to_numpy(dtype=np.float32)
        top_k = max(1, int(np.ceil(0.1 * len(indices))))
        metrics.append(
            {
                "DMS_name": str(study),
                "DMS_on": str(frame.loc[indices, "DMS_on"].iloc[0]),
                "n": int(len(indices)),
                "spearman": float(spearmanr(frame.loc[indices, "DMS_score"], study_pred).statistic),
                "ndcg_top_10pct": float(ndcg_score(relevance.reshape(1, -1), study_pred.reshape(1, -1), k=top_k)),
                "holdout_group": heldout,
            }
        )
        out = frame.loc[indices, ["sample_id", "DMS_name", "DMS_on", "DMS_score", "rank_target", "antigen_name"]].copy()
        out["holdout_group"] = heldout
        out["prediction"] = study_pred
        predictions.append(out)
    checkpoint_path = ""
    if args.save_checkpoints:
        checkpoint_path = save_fold_checkpoint(args, model, heldout, scalar, structure, scalar_scaler, structure_scaler)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics), {
        "history": history,
        "checkpoint_path": checkpoint_path,
        "init_checkpoint": str(args.init_checkpoint or ""),
        "init_summary": init_summary,
        **param_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--structure-features", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-residues", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-last-layers", type=int, default=2)
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="query,value")
    parser.add_argument("--loss", choices=["pairwise", "listwise"], default="pairwise")
    parser.add_argument("--pairs-per-study", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--listwise-steps-per-epoch", type=int, default=256)
    parser.add_argument("--listwise-batch-items", type=int, default=64)
    parser.add_argument("--listwise-temperature", type=float, default=0.15)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--minimum-rank-delta", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint for compatible LoRA/head initialization.")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--holdout-column", default="antigen_name", help="Column used to define held-out groups.")
    parser.add_argument("--folds", default="", help="Comma-separated holdout groups. Defaults to all values in --holdout-column.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="GEPBind-AbAgym-AAAI")
    parser.add_argument("--wandb-entity", default="s230112")
    parser.add_argument("--wandb-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.records, low_memory=False)
    if args.limit:
        frame = frame.head(args.limit).copy()
    frame = frame.reset_index(drop=True)
    print("building mutation-centered ESM inputs", flush=True)
    sequences, positions = build_mutant_contexts(frame, args.structure_dir, args.max_residues)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    tensors = tokenize_contexts(tokenizer, sequences, args.max_residues)
    scalar, scalar_names = scalar_features(frame)
    structure = None
    structure_names = []
    if args.structure_features is not None:
        structure, structure_names = load_npz(args.structure_features, len(frame))
    if args.holdout_column not in frame.columns:
        raise ValueError(f"Missing holdout column {args.holdout_column!r}")
    groups = [x.strip() for x in args.folds.split(",") if x.strip()] or sorted(frame[args.holdout_column].astype(str).unique())
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name or None, config=vars(args))
    rng = np.random.default_rng(args.seed)
    predictions, metrics, fold_summaries = [], [], []
    for heldout in groups:
        pred, metric, summary = train_one_fold(args, frame, tensors, positions, scalar, structure, heldout, rng, run)
        predictions.append(pred)
        metrics.append(metric)
        summary["holdout_group"] = heldout
        fold_summaries.append(summary)
        pd.concat(predictions, ignore_index=True).to_csv(args.out_dir / "predictions.partial.csv", index=False)
        pd.concat(metrics, ignore_index=True).to_csv(args.out_dir / "per_study_scores.partial.csv", index=False)
        partial = {
            "model": "AIRank-ESMAdapter",
            "uses_foldx": False,
            "loss": args.loss,
            "lora_rank": int(args.lora_rank),
            "holdout_column": args.holdout_column,
            "completed_holdout_groups": [item["holdout_group"] for item in fold_summaries],
            "model_name": args.model_name,
            "max_residues": int(args.max_residues),
            "fold_summaries": fold_summaries,
        }
        (args.out_dir / "summary.partial.json").write_text(json.dumps(partial, indent=2) + "\n")
        print(f"wrote partial outputs after holdout={heldout}", flush=True)
    pred_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.concat(metrics, ignore_index=True)
    summary = {
        "model": "AIRank-ESMAdapter",
        "uses_foldx": False,
        "loss": args.loss,
        "lora_rank": int(args.lora_rank),
        "holdout_column": args.holdout_column,
        "lora_targets": args.lora_targets,
        "records": int(len(pred_frame)),
        "studies": int(len(metric_frame)),
        "holdout_groups": groups,
        "model_name": args.model_name,
        "max_residues": int(args.max_residues),
        "scalar_features": scalar_names,
        "structure_features": structure_names,
        "macro_mean_spearman": float(metric_frame["spearman"].mean()) if len(metric_frame) else None,
        "macro_mean_ndcg_top_10pct": float(metric_frame["ndcg_top_10pct"].mean()) if len(metric_frame) else None,
        "fold_summaries": fold_summaries,
    }
    pred_frame.to_csv(args.out_dir / "predictions.csv", index=False)
    metric_frame.to_csv(args.out_dir / "per_study_scores.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
