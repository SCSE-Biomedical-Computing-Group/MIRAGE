#!/usr/bin/env python3
"""Score AbAgym predictions with per-DMS Spearman and top-5% ROC-AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--score-column", default="prediction")
    ap.add_argument("--truth-column", default="DMS_score")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-fraction", type=float, default=0.05)
    args = ap.parse_args()

    frame = pd.read_csv(args.predictions, low_memory=False)
    total_records = int(len(frame))
    frame = frame.dropna(subset=[args.truth_column, args.score_column]).copy()
    dropped_records = total_records - int(len(frame))
    rows = []
    for study, group in frame.groupby("DMS_name", sort=True):
        truth = group[args.truth_column].to_numpy(dtype=float)
        pred = group[args.score_column].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        cutoff = np.quantile(truth, 1.0 - args.top_fraction)
        label = (truth >= cutoff).astype(int)
        auc = float(roc_auc_score(label, pred)) if len(np.unique(label)) == 2 else float("nan")
        rows.append(
            {
                "DMS_name": study,
                "DMS_on": str(group["DMS_on"].iloc[0]) if "DMS_on" in group else "",
                "antigen_name": str(group["antigen_name"].iloc[0]) if "antigen_name" in group else "",
                "n": int(len(group)),
                "spearman": float(spearmanr(truth, pred).statistic),
                "roc_auc_top_5pct_high_score": auc,
            }
        )
    result = pd.DataFrame(rows)
    summary = {
        "predictions": str(args.predictions),
        "score_column": args.score_column,
        "truth_column": args.truth_column,
        "top_fraction": args.top_fraction,
        "studies": int(len(result)),
        "records": int(result["n"].sum()),
        "input_records": total_records,
        "dropped_missing_records": dropped_records,
        "average_spearman": float(result["spearman"].mean()),
        "average_roc_auc": float(result["roc_auc_top_5pct_high_score"].mean()),
        "weighted_spearman": float(np.average(result["spearman"], weights=result["n"])),
        "weighted_roc_auc": float(np.average(result["roc_auc_top_5pct_high_score"], weights=result["n"])),
        "by_mutated_side": {
            side: {
                "studies": int(len(group)),
                "average_spearman": float(group["spearman"].mean()),
                "average_roc_auc": float(group["roc_auc_top_5pct_high_score"].mean()),
            }
            for side, group in result.groupby("DMS_on", sort=True)
        },
        "per_study": result.to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out.with_suffix(".csv"), index=False)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
