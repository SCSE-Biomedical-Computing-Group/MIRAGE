#!/usr/bin/env python3
"""Check that archived MIRAGE predictions reproduce the paper metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED = {
    "studies": 68,
    "records": 36504,
    "average_spearman": 0.3610019854184025,
    "average_roc_auc": 0.763351458609284,
    "weighted_spearman": 0.3115013507265664,
    "weighted_roc_auc": 0.7431936528462341,
}

EXPECTED_MD5 = "a7368cd36168bb9e834b769c61e0c60b"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="results/final/rescored_mirage_monohgb_paper_metrics.json",
        help="Metrics JSON produced by score_abagym_predictions_like_paper.py.",
    )
    parser.add_argument(
        "--predictions",
        default="results/final/mirage_monohgb_predictions.csv",
        help="Archived final prediction table.",
    )
    parser.add_argument("--tol", type=float, default=1e-12)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    predictions_path = Path(args.predictions)

    with metrics_path.open() as handle:
        metrics = json.load(handle)

    failures: list[str] = []
    for key, expected in EXPECTED.items():
        observed = metrics.get(key)
        if isinstance(expected, int):
            if observed != expected:
                failures.append(f"{key}: expected {expected}, observed {observed}")
        elif not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=args.tol):
            failures.append(f"{key}: expected {expected}, observed {observed}")

    observed_md5 = file_md5(predictions_path)
    if observed_md5 != EXPECTED_MD5:
        failures.append(
            f"prediction MD5: expected {EXPECTED_MD5}, observed {observed_md5}"
        )

    if failures:
        print("MIRAGE archived-result check: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("MIRAGE archived-result check: PASS")
    print(f"  predictions_md5: {observed_md5}")
    for key in EXPECTED:
        print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
