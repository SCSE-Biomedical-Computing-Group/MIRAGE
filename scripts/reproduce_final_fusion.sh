#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi
OUT_DIR="${OUT_DIR:-results/reproduced_final_fusion}"

"${PYTHON_BIN}" scripts/fit_abagym_fusion_model_zoo.py \
  --base results/branch_predictions/four_branch_base_predictions.csv \
  --feature foldx=results/branch_predictions/four_branch_base_predictions.csv:foldx \
  --feature rsa=results/branch_predictions/four_branch_base_predictions.csv:rsa \
  --feature struct=results/branch_predictions/four_branch_base_predictions.csv:struct \
  --feature esm_lora_listwise=results/branch_predictions/four_branch_base_predictions.csv:esm_lora_listwise \
  --feature chem_hgb=results/branch_predictions/chem_hgb_predictions.csv:chem_hgb \
  --out-dir "${OUT_DIR}" \
  --target rank_target \
  --models monotonic_hgb

"${PYTHON_BIN}" scripts/score_abagym_predictions_like_paper.py \
  --predictions "${OUT_DIR}/monotonic_hgb/predictions.csv" \
  --score-column prediction \
  --out "${OUT_DIR}/monotonic_hgb/paper_metrics.json" \
  > /tmp/mirage_reproduced_score_stdout.json

"${PYTHON_BIN}" scripts/check_archived_metrics.py \
  --metrics "${OUT_DIR}/monotonic_hgb/paper_metrics.json" \
  --predictions "${OUT_DIR}/monotonic_hgb/predictions.csv"

cmp --silent "${OUT_DIR}/monotonic_hgb/predictions.csv" results/final/mirage_monohgb_predictions.csv
echo "MIRAGE final-fusion training reproduction: PASS"
