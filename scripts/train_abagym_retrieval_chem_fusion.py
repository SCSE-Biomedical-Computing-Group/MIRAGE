#!/usr/bin/env python3
"""Split-clean chemistry/retrieval fusion for AbAgym mutation ranking.

This experimental model adds two non-neural branches to AIRank:
  * fChem: explicit mutation chemistry and local structural context.
  * fRetrieval: kNN label transfer from non-held-out antigen families using
    mutation chemistry, geometry, and existing branch scores.

The final calibrator is an antigen-family-disjoint HistGradientBoosting model
that includes assay metadata as non-monotonic context features while preserving
positive monotonic constraints for branch scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path("/home2/s230112/GEPBind")
DEFAULT_RECORDS = ROOT / "datasets/abagym_study_disjoint/abagym_interface_study_rank_records.csv"
DEFAULT_FEATURE_TABLE = (
    ROOT / "runs/abagym_math_fusion_zoo_t30_s17_s31_ranktarget/monotonic_hgb/predictions.csv"
)
DEFAULT_STRUCT_NPZ = ROOT / "datasets/abagym_study_disjoint/abagym_official_structure_features.npz"
DEFAULT_OUT_DIR = ROOT / "runs/abagym_retrieval_chem_fusion_t30_s17_s31_ranktarget"


AA_PROPS = {
    # hydrophobicity(KD), volume, charge, aromatic, polar, donor, acceptor, sulfur, proline, glycine
    "A": (1.8, 88.6, 0, 0, 0, 0, 0, 0, 0, 0),
    "C": (2.5, 108.5, 0, 0, 0, 0, 0, 1, 0, 0),
    "D": (-3.5, 111.1, -1, 0, 1, 0, 1, 0, 0, 0),
    "E": (-3.5, 138.4, -1, 0, 1, 0, 1, 0, 0, 0),
    "F": (2.8, 189.9, 0, 1, 0, 0, 0, 0, 0, 0),
    "G": (-0.4, 60.1, 0, 0, 0, 0, 0, 0, 0, 1),
    "H": (-3.2, 153.2, 0.5, 1, 1, 1, 1, 0, 0, 0),
    "I": (4.5, 166.7, 0, 0, 0, 0, 0, 0, 0, 0),
    "K": (-3.9, 168.6, 1, 0, 1, 1, 0, 0, 0, 0),
    "L": (3.8, 166.7, 0, 0, 0, 0, 0, 0, 0, 0),
    "M": (1.9, 162.9, 0, 0, 0, 0, 0, 1, 0, 0),
    "N": (-3.5, 114.1, 0, 0, 1, 1, 1, 0, 0, 0),
    "P": (-1.6, 112.7, 0, 0, 0, 0, 0, 0, 1, 0),
    "Q": (-3.5, 143.8, 0, 0, 1, 1, 1, 0, 0, 0),
    "R": (-4.5, 173.4, 1, 0, 1, 1, 0, 0, 0, 0),
    "S": (-0.8, 89.0, 0, 0, 1, 1, 1, 0, 0, 0),
    "T": (-0.7, 116.1, 0, 0, 1, 1, 1, 0, 0, 0),
    "V": (4.2, 140.0, 0, 0, 0, 0, 0, 0, 0, 0),
    "W": (-0.9, 227.8, 0, 1, 0, 1, 0, 0, 0, 0),
    "Y": (-1.3, 193.6, 0, 1, 1, 1, 1, 0, 0, 0),
}
PROP_NAMES = [
    "hydrophobicity",
    "volume",
    "charge",
    "aromatic",
    "polar",
    "donor",
    "acceptor",
    "sulfur",
    "proline",
    "glycine",
]


def aa_vec(aa: str) -> np.ndarray:
    return np.asarray(AA_PROPS.get(str(aa).strip().upper(), (0,) * len(PROP_NAMES)), dtype=np.float32)


def add_chemistry_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    wt = np.vstack([aa_vec(x) for x in df["wildtype"]])
    mut = np.vstack([aa_vec(x) for x in df["mutation"]])
    delta = mut - wt
    out = df.copy()
    cols = []
    for i, name in enumerate(PROP_NAMES):
        for prefix, values in [("wt", wt), ("mut", mut), ("delta", delta), ("abs_delta", np.abs(delta))]:
            col = f"chem_{prefix}_{name}"
            out[col] = values[:, i]
            cols.append(col)

    wt_charge, mut_charge = wt[:, 2], mut[:, 2]
    wt_hydro, mut_hydro = wt[:, 0], mut[:, 0]
    wt_vol, mut_vol = wt[:, 1], mut[:, 1]
    out["chem_charge_gain"] = (mut_charge > wt_charge).astype(float)
    out["chem_charge_loss"] = (mut_charge < wt_charge).astype(float)
    out["chem_charge_reversal"] = ((wt_charge * mut_charge) < 0).astype(float)
    out["chem_to_proline"] = mut[:, PROP_NAMES.index("proline")]
    out["chem_from_glycine"] = wt[:, PROP_NAMES.index("glycine")]
    out["chem_to_aromatic"] = mut[:, PROP_NAMES.index("aromatic")]
    out["chem_hydrophobic_gain"] = (mut_hydro - wt_hydro).astype(float)
    out["chem_volume_increase"] = (mut_vol - wt_vol).astype(float)
    cols += [
        "chem_charge_gain",
        "chem_charge_loss",
        "chem_charge_reversal",
        "chem_to_proline",
        "chem_from_glycine",
        "chem_to_aromatic",
        "chem_hydrophobic_gain",
        "chem_volume_increase",
    ]
    return out, cols


def add_structure_features(df: pd.DataFrame, npz_path: Path) -> tuple[pd.DataFrame, list[str]]:
    z = np.load(npz_path, allow_pickle=True)
    x = z["x"].astype(np.float32)
    names = [f"struct_raw_{name}" for name in z["names"].astype(str)]
    if len(df) != x.shape[0]:
        raise ValueError(f"row mismatch: records={len(df)} structure_features={x.shape[0]}")
    out = df.copy()
    for i, name in enumerate(names):
        out[name] = x[:, i]
    return out, names


def score_predictions(frame: pd.DataFrame, score_column: str = "prediction") -> tuple[dict, pd.DataFrame]:
    rows = []
    for study, group in frame.groupby("DMS_name", sort=True):
        truth = group["DMS_score"].to_numpy(dtype=float)
        pred = group[score_column].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        cutoff = np.quantile(truth, 0.95)
        label = (truth >= cutoff).astype(int)
        auc = float(roc_auc_score(label, pred)) if len(np.unique(label)) == 2 else float("nan")
        rows.append(
            {
                "DMS_name": study,
                "DMS_on": str(group["DMS_on"].iloc[0]),
                "antigen_name": str(group["antigen_name"].iloc[0]),
                "n": int(len(group)),
                "spearman": float(spearmanr(truth, pred).statistic),
                "roc_auc_top_5pct_high_score": auc,
            }
        )
    per_study = pd.DataFrame(rows)
    summary = {
        "studies": int(len(per_study)),
        "records": int(per_study["n"].sum()),
        "average_spearman": float(per_study["spearman"].mean()),
        "average_roc_auc": float(per_study["roc_auc_top_5pct_high_score"].mean()),
        "weighted_spearman": float(np.average(per_study["spearman"], weights=per_study["n"])),
        "weighted_roc_auc": float(np.average(per_study["roc_auc_top_5pct_high_score"], weights=per_study["n"])),
    }
    return summary, per_study


def fit_branch_hgb(x_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=220,
        max_leaf_nodes=16,
        l2_regularization=0.2,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=11,
    )
    model.fit(x_train, y_train)
    return model


def retrieval_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(y_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(x_train)
    dist, idx = nn.kneighbors(x_test, return_distance=True)
    weight = 1.0 / (dist + 1e-3)
    retrieved = y_train[idx]
    avg = (retrieved * weight).sum(axis=1) / weight.sum(axis=1)
    top = (retrieved >= np.quantile(y_train, 0.95)).mean(axis=1)
    return avg.astype(np.float32), top.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--structure-npz", type=Path, default=DEFAULT_STRUCT_NPZ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target", default="rank_target", choices=["rank_target", "DMS_score"])
    parser.add_argument("--retrieval-k", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(args.records, low_memory=False)
    records, struct_cols = add_structure_features(records, args.structure_npz)
    base = pd.read_csv(args.feature_table, low_memory=False)[
        ["sample_id", "foldx", "rsa", "struct", "esm_lora_listwise"]
    ]
    df = records.merge(base, on="sample_id", validate="one_to_one")
    df["holdout_group"] = df["antigen_name"].astype(str)
    df, chem_cols = add_chemistry_features(df)

    side_cols = pd.get_dummies(df["DMS_on"].astype(str), prefix="side", dtype=float)
    type_cols = pd.get_dummies(df["experimental_DMS_type"].astype(str), prefix="assay", dtype=float)
    df = pd.concat([df, side_cols, type_cols], axis=1)
    meta_cols = list(side_cols.columns) + list(type_cols.columns)

    branch_cols = ["foldx", "rsa", "struct", "esm_lora_listwise"]
    chem_train_cols = chem_cols + struct_cols
    retrieval_cols = chem_cols + struct_cols + branch_cols
    final_score_cols = branch_cols + ["chem_hgb", "retrieval_rank", "retrieval_top5"]
    final_cols = final_score_cols + meta_cols
    final_cols_nometa = final_score_cols
    monotonic = [1] * len(final_score_cols) + [0] * len(meta_cols)
    monotonic_nometa = [1] * len(final_cols_nometa)

    df = df.dropna(subset=[args.target, "DMS_score"] + branch_cols + chem_train_cols).copy()
    df["chem_hgb"] = np.nan
    df["retrieval_rank"] = np.nan
    df["retrieval_top5"] = np.nan
    df["prediction"] = np.nan
    df["prediction_no_meta"] = np.nan
    fold_summaries = []

    for holdout in sorted(df["holdout_group"].astype(str).unique()):
        train = df["holdout_group"].astype(str) != holdout
        test = ~train
        y_train = df.loc[train, args.target].to_numpy(dtype=np.float32)

        chem_scaler = StandardScaler().fit(df.loc[train, chem_train_cols])
        x_chem_train = chem_scaler.transform(df.loc[train, chem_train_cols]).astype(np.float32)
        x_chem_test = chem_scaler.transform(df.loc[test, chem_train_cols]).astype(np.float32)
        chem_model = fit_branch_hgb(x_chem_train, y_train)
        train_chem_hgb = chem_model.predict(x_chem_train).astype(np.float32)
        test_chem_hgb = chem_model.predict(x_chem_test).astype(np.float32)
        df.loc[test, "chem_hgb"] = test_chem_hgb

        ret_scaler = StandardScaler().fit(df.loc[train, retrieval_cols])
        x_ret_train = ret_scaler.transform(df.loc[train, retrieval_cols]).astype(np.float32)
        x_ret_test = ret_scaler.transform(df.loc[test, retrieval_cols]).astype(np.float32)
        ret_rank, ret_top = retrieval_scores(x_ret_train, y_train, x_ret_test, args.retrieval_k)
        df.loc[test, "retrieval_rank"] = ret_rank
        df.loc[test, "retrieval_top5"] = ret_top

        # For final-calibrator training, each training row must also have a
        # retrieval score derived from other training rows. A single train-fold
        # approximation avoids using held-out labels and keeps runtime modest.
        ret_train_rank, ret_train_top = retrieval_scores(x_ret_train, y_train, x_ret_train, args.retrieval_k + 1)
        train_frame = df.loc[train].copy()
        test_frame = df.loc[test].copy()
        train_frame["chem_hgb"] = train_chem_hgb
        train_frame["retrieval_rank"] = ret_train_rank
        train_frame["retrieval_top5"] = ret_train_top
        test_frame["chem_hgb"] = test_chem_hgb
        test_frame["retrieval_rank"] = ret_rank
        test_frame["retrieval_top5"] = ret_top

        final_model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=300,
            max_leaf_nodes=16,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            monotonic_cst=monotonic,
            random_state=17,
        )
        final_model.fit(
            train_frame[final_cols].to_numpy(dtype=np.float32),
            y_train,
        )
        df.loc[test, "prediction"] = final_model.predict(test_frame[final_cols].to_numpy(dtype=np.float32))

        final_model_nometa = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=12,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            monotonic_cst=monotonic_nometa,
            random_state=19,
        )
        final_model_nometa.fit(
            train_frame[final_cols_nometa].to_numpy(dtype=np.float32),
            y_train,
        )
        df.loc[test, "prediction_no_meta"] = final_model_nometa.predict(
            test_frame[final_cols_nometa].to_numpy(dtype=np.float32)
        )
        fold_summaries.append(
            {
                "holdout_group": holdout,
                "train_records": int(train.sum()),
                "test_records": int(test.sum()),
                "chem_hgb_iter": int(chem_model.n_iter_),
                "final_iter": int(final_model.n_iter_),
                "final_no_meta_iter": int(final_model_nometa.n_iter_),
            }
        )
        print(f"holdout={holdout} train={int(train.sum())} test={int(test.sum())}", flush=True)

    if df["prediction"].isna().any():
        raise RuntimeError("missing predictions")
    if df["prediction_no_meta"].isna().any():
        raise RuntimeError("missing no-meta predictions")

    predictions = args.out_dir / "predictions.csv"
    df[
        [
            "sample_id",
            "DMS_name",
            "DMS_on",
            "experimental_DMS_type",
            "antigen_name",
            "DMS_score",
            "rank_target",
            "holdout_group",
            *final_cols,
            "prediction",
            "prediction_no_meta",
        ]
    ].to_csv(predictions, index=False)
    summary, per_study = score_predictions(df, "prediction")
    summary_no_meta, per_study_no_meta = score_predictions(df, "prediction_no_meta")
    summary_chem, per_study_chem = score_predictions(df, "chem_hgb")
    summary_retrieval, per_study_retrieval = score_predictions(df, "retrieval_rank")
    per_study.to_csv(args.out_dir / "paper_metrics.csv", index=False)
    per_study_no_meta.to_csv(args.out_dir / "paper_metrics_no_meta.csv", index=False)
    per_study_chem.to_csv(args.out_dir / "paper_metrics_chem_hgb.csv", index=False)
    per_study_retrieval.to_csv(args.out_dir / "paper_metrics_retrieval_rank.csv", index=False)
    (args.out_dir / "paper_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "paper_metrics_no_meta.json").write_text(json.dumps(summary_no_meta, indent=2) + "\n")
    (args.out_dir / "paper_metrics_chem_hgb.json").write_text(json.dumps(summary_chem, indent=2) + "\n")
    (args.out_dir / "paper_metrics_retrieval_rank.json").write_text(json.dumps(summary_retrieval, indent=2) + "\n")
    payload = {
        "model": "AIRank retrieval+chemistry assay-aware MonoHGB",
        "target": args.target,
        "records": int(len(df)),
        "features": {
            "final_score_cols": final_score_cols,
            "metadata_cols": meta_cols,
            "chemistry_cols": chem_cols,
            "structure_cols": struct_cols,
            "retrieval_cols": retrieval_cols,
        },
        "retrieval_k": int(args.retrieval_k),
        "metrics": summary,
        "metrics_no_meta": summary_no_meta,
        "metrics_chem_hgb": summary_chem,
        "metrics_retrieval_rank": summary_retrieval,
        "fold_summaries": fold_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
