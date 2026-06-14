#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

"${PYTHON_BIN}" scripts/score_abagym_predictions_like_paper.py \
  --predictions results/final/mirage_monohgb_predictions.csv \
  --score-column prediction \
  --out /tmp/mirage_archived_metrics.json \
  > /tmp/mirage_archived_score_stdout.json

"${PYTHON_BIN}" scripts/check_archived_metrics.py \
  --metrics /tmp/mirage_archived_metrics.json \
  --predictions results/final/mirage_monohgb_predictions.csv
