#!/usr/bin/env python3
"""Leave-one-study-out pairwise ranking baseline for AbAgym interface mutations."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from torch import nn
from torch.nn import functional as F


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: idx for idx, aa in enumerate(AA)}
AA_PROP = {
    "A": (1.8, 0.0, 0.0, 0.0, 88.6),
    "C": (2.5, 0.0, 0.0, 0.0, 108.5),
    "D": (-3.5, -1.0, 0.0, 1.0, 111.1),
    "E": (-3.5, -1.0, 0.0, 1.0, 138.4),
    "F": (2.8, 0.0, 1.0, 0.0, 189.9),
    "G": (-0.4, 0.0, 0.0, 0.0, 60.1),
    "H": (-3.2, 0.5, 1.0, 1.0, 153.2),
    "I": (4.5, 0.0, 0.0, 0.0, 166.7),
    "K": (-3.9, 1.0, 0.0, 1.0, 168.6),
    "L": (3.8, 0.0, 0.0, 0.0, 166.7),
    "M": (1.9, 0.0, 0.0, 0.0, 162.9),
    "N": (-3.5, 0.0, 0.0, 1.0, 114.1),
    "P": (-1.6, 0.0, 0.0, 0.0, 112.7),
    "Q": (-3.5, 0.0, 0.0, 1.0, 143.8),
    "R": (-4.5, 1.0, 0.0, 1.0, 173.4),
    "S": (-0.8, 0.0, 0.0, 1.0, 89.0),
    "T": (-0.7, 0.0, 0.0, 1.0, 116.1),
    "V": (4.2, 0.0, 0.0, 0.0, 140.0),
    "W": (-0.9, 0.0, 1.0, 0.0, 227.8),
    "Y": (-1.3, 0.0, 1.0, 1.0, 193.6),
}


class StudyRankNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def aa_features(residue: str) -> tuple[np.ndarray, np.ndarray]:
    one_hot = np.zeros(len(AA), dtype=np.float32)
    if residue in AA_INDEX:
        one_hot[AA_INDEX[residue]] = 1.0
    properties = np.asarray(AA_PROP.get(residue, (0, 0, 0, 0, 0)), dtype=np.float32)
    return one_hot, properties


def site_number(value: object) -> float:
    match = re.search(r"-?\d+", str(value))
    return float(match.group()) if match else 0.0


def build_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dms_types = sorted(frame["experimental_DMS_type"].astype(str).unique())
    rows = []
    for row in frame.itertuples(index=False):
        wt_onehot, wt_prop = aa_features(str(row.wildtype))
        mut_onehot, mut_prop = aa_features(str(row.mutation))
        distance = float(row.closest_interface_atom_distance)
        base = np.asarray(
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
        type_onehot = np.asarray(
            [float(str(row.experimental_DMS_type) == item) for item in dms_types],
            dtype=np.float32,
        )
        rows.append(np.concatenate([wt_onehot, mut_onehot, wt_prop, mut_prop, mut_prop - wt_prop, base, type_onehot]))
    names = (
        [f"wildtype_{aa}" for aa in AA]
        + [f"mutation_{aa}" for aa in AA]
        + [f"wildtype_property_{i}" for i in range(5)]
        + [f"mutation_property_{i}" for i in range(5)]
        + [f"property_delta_{i}" for i in range(5)]
        + ["site_number_scaled", "interface_distance_scaled", "interface_distance_decay",
           "interface_le_3p5", "interface_le_5", "interface_le_8", "mutates_antibody", "mutates_antigen"]
        + [f"dms_type_{item}" for item in dms_types]
    )
    return np.asarray(rows, dtype=np.float32), names


def draw_pairs(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    pairs_per_study: int,
    minimum_delta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    train = frame.loc[train_indices]
    for _, study in train.groupby("DMS_name", sort=True):
        indices = study.index.to_numpy()
        targets = study["rank_target"].to_numpy()
        if len(indices) < 2:
            continue
        lookup = dict(zip(indices, targets))
        left: list[int] = []
        right: list[int] = []
        while len(left) < pairs_per_study:
            a = rng.choice(indices, size=pairs_per_study, replace=True)
            b = rng.choice(indices, size=pairs_per_study, replace=True)
            delta = np.asarray([lookup[x] - lookup[y] for x, y in zip(a, b)])
            valid = np.abs(delta) >= minimum_delta
            left.extend(a[valid].tolist())
            right.extend(b[valid].tolist())
        a = np.asarray(left[:pairs_per_study], dtype=int)
        b = np.asarray(right[:pairs_per_study], dtype=int)
        labels = np.sign(frame.loc[a, "rank_target"].to_numpy() - frame.loc[b, "rank_target"].to_numpy())
        left_parts.append(a)
        right_parts.append(b)
        label_parts.append(labels.astype(np.float32))
    return np.concatenate(left_parts), np.concatenate(right_parts), np.concatenate(label_parts)


def draw_listwise_batch(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    studies: np.ndarray,
    max_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    study = str(rng.choice(studies))
    eligible = frame.index[(frame.index.isin(train_indices)) & (frame["DMS_name"].astype(str) == study)].to_numpy()
    if len(eligible) > max_items:
        eligible = rng.choice(eligible, size=max_items, replace=False)
    return np.asarray(eligible, dtype=int)


def predict(model: nn.Module, features: torch.Tensor, indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
            output.append(model(features[idx].to(device)).cpu().numpy())
    return np.concatenate(output)


def load_extra_features(specs: list[str]) -> tuple[np.ndarray | None, list[str]]:
    blocks = []
    names: list[str] = []
    for spec in specs:
        path_text, columns_text = spec.split(":", 1)
        columns = [item.strip() for item in columns_text.split(",") if item.strip()]
        frame = pd.read_csv(path_text, low_memory=False)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing extra feature columns in {path_text}: {missing}")
        block = frame[columns].to_numpy(dtype=np.float32)
        blocks.append(block)
        names.extend([f"{Path(path_text).stem}:{column}" for column in columns])
    if not blocks:
        return None, []
    return np.concatenate(blocks, axis=1), names


def evaluate_study(frame: pd.DataFrame, indices: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    study = frame.loc[indices]
    relevance = study["rank_target"].to_numpy(dtype=np.float32)
    top_k = max(1, int(np.ceil(0.1 * len(study))))
    selected = np.argsort(prediction)[-top_k:]
    rho = float(spearmanr(study["DMS_score"], prediction).statistic)
    return {
        "DMS_name": str(study["DMS_name"].iloc[0]),
        "DMS_on": str(study["DMS_on"].iloc[0]),
        "experimental_DMS_type": str(study["experimental_DMS_type"].iloc[0]),
        "n": int(len(study)),
        "spearman": rho,
        "ndcg_top_10pct": float(ndcg_score(relevance.reshape(1, -1), prediction.reshape(1, -1), k=top_k)),
        "top_10pct_enrichment_over_random": float(relevance[selected].mean() - 0.5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--structure-features", type=Path, default=None)
    ap.add_argument(
        "--extra-features",
        action="append",
        default=[],
        help="CSV:col1,col2,... extra feature block aligned to records by row order.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--loss", default="pairwise", choices=["pairwise", "listwise"])
    ap.add_argument("--pairs-per-study", type=int, default=512)
    ap.add_argument("--pair-batch-size", type=int, default=4096)
    ap.add_argument("--listwise-steps-per-epoch", type=int, default=384)
    ap.add_argument("--listwise-batch-items", type=int, default=256)
    ap.add_argument("--listwise-temperature", type=float, default=0.08)
    ap.add_argument("--eval-batch-size", type=int, default=32768)
    ap.add_argument("--minimum-rank-delta", type=float, default=0.05)
    ap.add_argument("--stratify-by-mutated-side", action="store_true")
    ap.add_argument("--holdout-column", default="DMS_name")
    ap.add_argument("--folds", default="", help="Optional comma-separated held-out groups for smoke runs.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="GEPBind-Rank")
    ap.add_argument("--wandb-entity", default="s230112")
    ap.add_argument("--wandb-name", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.holdout_column not in pd.read_csv(args.records, nrows=0).columns:
        raise ValueError(f"Missing holdout column {args.holdout_column!r}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.records)
    features_np, feature_names = build_features(frame)
    if args.structure_features is not None:
        payload = np.load(args.structure_features, allow_pickle=True)
        structural = payload["x"].astype(np.float32)
        if len(structural) != len(features_np):
            raise ValueError(f"Structure feature row mismatch: {len(structural)} versus {len(features_np)}")
        features_np = np.concatenate([features_np, structural], axis=1)
        feature_names.extend(payload["names"].tolist())
    extra_np, extra_names = load_extra_features(args.extra_features)
    if extra_np is not None:
        if len(extra_np) != len(features_np):
            raise ValueError(f"Extra feature row mismatch: {len(extra_np)} versus {len(features_np)}")
        features_np = np.concatenate([features_np, extra_np], axis=1)
        feature_names.extend(extra_names)
    np.savez_compressed(args.out_dir / "features.npz", x=features_np, names=np.asarray(feature_names))
    group_values = frame[args.holdout_column].astype(str)
    all_groups = sorted(group_values.unique())
    requested = [value.strip() for value in args.folds.split(",") if value.strip()]
    test_groups = requested or all_groups
    unknown = sorted(set(test_groups) - set(all_groups))
    if unknown:
        raise ValueError(f"Unknown held-out groups: {unknown}")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group="abagym-study-disjoint-ranknet",
            name=args.wandb_name or None,
            config=vars(args),
        )

    fold_metrics: list[dict[str, object]] = []
    predictions = []
    rng = np.random.default_rng(args.seed)
    fold_index = 0
    for heldout in test_groups:
        group_values = frame[args.holdout_column].astype(str)
        group_mask = group_values == heldout
        for side in sorted(frame.loc[group_mask, "DMS_on"].unique()):
            test = frame.index[group_mask & (frame["DMS_on"] == side)].to_numpy()
            train_mask = group_values != heldout
            if args.stratify_by_mutated_side:
                train_mask &= frame["DMS_on"] == side
            train = frame.index[train_mask].to_numpy()
            if len(frame.loc[train, "DMS_name"].unique()) < 2:
                raise ValueError(f"Not enough training studies for {heldout} in side stratum {side}.")
            mean = features_np[train].mean(axis=0, keepdims=True)
            std = features_np[train].std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            features = torch.from_numpy((features_np - mean) / std)
            model = StudyRankNet(features.shape[1], args.hidden_dim, args.dropout).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            train_studies = frame.loc[train, "DMS_name"].astype(str).unique()
            for _ in range(args.epochs):
                model.train()
                if args.loss == "pairwise":
                    left, right, labels = draw_pairs(frame, train, args.pairs_per_study, args.minimum_rank_delta, rng)
                    order = rng.permutation(len(left))
                    for start in range(0, len(order), args.pair_batch_size):
                        batch = order[start : start + args.pair_batch_size]
                        idx_l = torch.as_tensor(left[batch], dtype=torch.long)
                        idx_r = torch.as_tensor(right[batch], dtype=torch.long)
                        y = torch.as_tensor(labels[batch], dtype=torch.float32, device=device)
                        delta = model(features[idx_l].to(device)) - model(features[idx_r].to(device))
                        loss = F.softplus(-y * delta).mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                elif args.loss == "listwise":
                    for _step in range(args.listwise_steps_per_epoch):
                        batch = draw_listwise_batch(frame, train, train_studies, args.listwise_batch_items, rng)
                        idx = torch.as_tensor(batch, dtype=torch.long)
                        target = torch.as_tensor(frame.loc[batch, "rank_target"].to_numpy(dtype=np.float32), device=device)
                        scores = model(features[idx].to(device))
                        target_dist = F.softmax(target / args.listwise_temperature, dim=0)
                        loss = -(target_dist * F.log_softmax(scores, dim=0)).sum()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                else:
                    raise ValueError(f"Unknown loss: {args.loss}")
            group_prediction = predict(model, features, test, device, args.eval_batch_size)
            for study in sorted(frame.loc[test, "DMS_name"].unique()):
                positions = np.flatnonzero(frame.loc[test, "DMS_name"].to_numpy() == study)
                indices = test[positions]
                metric = evaluate_study(frame, indices, group_prediction[positions])
                metric["holdout_group"] = str(heldout)
                fold_metrics.append(metric)
                predicted = frame.loc[indices, ["sample_id", "DMS_name", "DMS_on", "DMS_score", "rank_target"]].copy()
                predicted["holdout_group"] = str(heldout)
                predicted["prediction"] = group_prediction[positions]
                predictions.append(predicted)
                if run is not None:
                    run.log({"fold_index": fold_index, "fold_spearman": metric["spearman"], "fold_ndcg_top_10pct": metric["ndcg_top_10pct"]})
                fold_index += 1

    metrics = pd.DataFrame(fold_metrics)
    pred_frame = pd.concat(predictions, ignore_index=True)
    by_side = {}
    for side, group in metrics.groupby("DMS_on", sort=True):
        by_side[side] = {
            "studies": int(len(group)),
            "records": int(group["n"].sum()),
            "macro_mean_spearman": float(group["spearman"].mean()),
            "record_weighted_mean_spearman": float(np.average(group["spearman"], weights=group["n"])),
            "macro_mean_ndcg_top_10pct": float(group["ndcg_top_10pct"].mean()),
        }
    summary = {
        "model": (
            "StudyRankNet mutation physicochemistry, interface distance and official modeled-structure contacts"
            if args.structure_features is not None
            else "StudyRankNet mutation physicochemistry plus released interface distance"
        ),
        "evaluation": f"leave-one-{args.holdout_column}-out within-study ranking",
        "holdout_column": args.holdout_column,
        "loss": args.loss,
        "n_holdout_groups": int(len(test_groups)),
        "training_scope": "same mutated-side studies only" if args.stratify_by_mutated_side else "all other studies",
        "endpoint_warning": "DMS endpoints are mutation/escape or enrichment measures, not Delta G.",
        "n_studies": int(len(metrics)),
        "n_records": int(metrics["n"].sum()),
        "macro_mean_spearman": float(metrics["spearman"].mean()),
        "record_weighted_mean_spearman": float(np.average(metrics["spearman"], weights=metrics["n"])),
        "macro_mean_ndcg_top_10pct": float(metrics["ndcg_top_10pct"].mean()),
        "macro_mean_top_10pct_enrichment_over_random": float(metrics["top_10pct_enrichment_over_random"].mean()),
        "by_mutated_side": by_side,
        "folds": fold_metrics,
    }
    metrics.to_csv(args.out_dir / "per_study_scores.csv", index=False)
    pred_frame.to_csv(args.out_dir / "predictions.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if run is not None:
        for key in ("macro_mean_spearman", "record_weighted_mean_spearman", "macro_mean_ndcg_top_10pct"):
            run.summary[key] = summary[key]
        run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
