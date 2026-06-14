#!/usr/bin/env python
"""Standardize AbAgym for study-disjoint mutation/escape ranking evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutations", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mutations = pd.read_csv(args.mutations)
    metadata = pd.read_csv(args.metadata)
    keep_meta = [
        "DMS_name",
        "antigen_name",
        "template_PDB_ID",
        "Antibody Chains",
        "Antigen Chains",
        "DMS_on",
        "experimental_DMS_type",
    ]
    records = mutations.merge(metadata[keep_meta], on="DMS_name", how="left", validate="many_to_one")
    records["rank_target"] = records.groupby("DMS_name")["DMS_score"].rank(method="average", pct=True)
    records["sample_id"] = [f"ABGYM_{index:06d}" for index in range(len(records))]
    records.to_csv(out / "abagym_interface_study_rank_records.csv", index=False)

    study_summary = (
        records.groupby(["DMS_name", "DMS_on", "experimental_DMS_type"], dropna=False)
        .agg(records=("sample_id", "size"), score_min=("DMS_score", "min"), score_max=("DMS_score", "max"))
        .reset_index()
    )
    study_summary.to_csv(out / "abagym_study_summary.csv", index=False)
    fold_dir = out / "leave_one_study_out"
    fold_dir.mkdir(exist_ok=True)
    for study in study_summary["DMS_name"]:
        split = records[["sample_id", "DMS_name"]].copy()
        split["split"] = "train"
        split.loc[split["DMS_name"] == study, "split"] = "test"
        split.to_csv(fold_dir / f"holdout_{study}.csv", index=False)

    audit = {
        "interface_mutation_records": int(len(records)),
        "studies": int(records["DMS_name"].nunique()),
        "antigen_mutation_studies": int((study_summary["DMS_on"] == "antigen").sum()),
        "antibody_mutation_studies": int((study_summary["DMS_on"] == "antibody").sum()),
        "evaluation": "leave-one-DMS-study-out within-study ranking",
        "target": "within-study rank of DMS_score",
        "warning": "DMS scores are study-specific escape/enrichment endpoints, not Delta G.",
    }
    (out / "abagym_prepare_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
