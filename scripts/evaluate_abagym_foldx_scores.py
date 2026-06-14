#!/usr/bin/env python3
"""Join AbAgym FoldX benchmark CSVs to standardized AbAgym records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--foldx-dir", required=True)
    p.add_argument("--score-file", default="complex_ddG_values.csv")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def read_score_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["mut_names", "foldx_score"])
    df["mut_names"] = df["mut_names"].astype(str)
    df["foldx_score"] = pd.to_numeric(df["foldx_score"], errors="coerce")
    return df.dropna(subset=["foldx_score"])


def main() -> None:
    args = parse_args()
    records = pd.read_csv(args.records)
    foldx_root = Path(args.foldx_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    missing_score_files = []
    duplicate_keys = 0
    for pdb_file, group in records.groupby("PDB_file", sort=True):
        score_path = foldx_root / pdb_file / args.score_file
        if not score_path.exists():
            missing_score_files.append(str(score_path))
            continue
        scores = read_score_file(score_path)
        duplicate_keys += int(scores.duplicated("mut_names").sum())
        scores = scores.drop_duplicates("mut_names", keep="first")
        merged = group.merge(scores, on="mut_names", how="left", validate="many_to_one")
        parts.append(merged)

    if not parts:
        raise RuntimeError("no FoldX score files matched")

    out = pd.concat(parts, ignore_index=True)
    out["prediction"] = out["foldx_score"]
    out["prediction_clipped_pm5"] = out["foldx_score"].clip(-5.0, 5.0)
    out.to_csv(out_dir / "predictions.csv", index=False)
    summary = {
        "records": int(len(records)),
        "joined_records": int(len(out)),
        "scored_records": int(out["foldx_score"].notna().sum()),
        "missing_score_records": int(out["foldx_score"].isna().sum()),
        "pdb_files": int(records["PDB_file"].nunique()),
        "scored_pdb_files": int(out.loc[out["foldx_score"].notna(), "PDB_file"].nunique()),
        "missing_score_files": missing_score_files,
        "duplicate_score_keys": duplicate_keys,
        "score_file": args.score_file,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
