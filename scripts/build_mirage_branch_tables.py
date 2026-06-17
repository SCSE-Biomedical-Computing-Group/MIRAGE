#!/usr/bin/env python3
"""Build the four-branch MIRAGE base table from branch prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RECORD_COLUMNS = [
    "sample_id",
    "DMS_name",
    "DMS_on",
    "DMS_score",
    "rank_target",
    "antigen_name",
]


def read_branch(path: Path, output_name: str, candidates: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in candidates:
        if column in frame.columns:
            out = frame[["sample_id", column]].rename(columns={column: output_name})
            return out
    raise ValueError(
        f"{path} does not contain any accepted score columns for {output_name}: "
        + ", ".join(candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--foldx", type=Path, required=True)
    parser.add_argument("--rsa", type=Path, required=True)
    parser.add_argument("--struct", type=Path, required=True)
    parser.add_argument("--esm-lora", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    records = pd.read_csv(args.records, low_memory=False)
    missing_record_cols = [col for col in RECORD_COLUMNS if col not in records.columns]
    if missing_record_cols:
        raise ValueError(f"records file is missing columns: {missing_record_cols}")

    base = records[RECORD_COLUMNS].copy()
    base["holdout_group"] = base["antigen_name"].astype(str)

    branches = {
        "foldx": read_branch(args.foldx, "foldx", ["prediction_clipped_pm5", "prediction", "foldx_score"]),
        "rsa": read_branch(args.rsa, "rsa", ["prediction_negative_rsa", "prediction"]),
        "struct": read_branch(args.struct, "struct", ["prediction"]),
        "esm_lora_listwise": read_branch(args.esm_lora, "esm_lora_listwise", ["prediction"]),
    }

    merged = base
    merge_summary = {}
    for name, branch in branches.items():
        before = len(merged)
        branch = branch.drop_duplicates("sample_id", keep="first")
        merged = merged.merge(branch, on="sample_id", how="left", validate="one_to_one")
        merge_summary[name] = {
            "branch_rows": int(len(branch)),
            "missing_after_join": int(merged[name].isna().sum()),
            "rows_before_join": int(before),
            "rows_after_join": int(len(merged)),
        }

    feature_cols = ["foldx", "rsa", "struct", "esm_lora_listwise"]
    complete = merged.dropna(subset=feature_cols + ["rank_target", "DMS_score"]).copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(args.out, index=False)

    payload = {
        "source_records": int(len(records)),
        "complete_records": int(len(complete)),
        "dropped_records": int(len(records) - len(complete)),
        "features": feature_cols,
        "joins": merge_summary,
        "output": str(args.out),
    }
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
