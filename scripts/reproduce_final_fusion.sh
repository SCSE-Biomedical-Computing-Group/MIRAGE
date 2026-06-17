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
BASE_PRED="${BASE_PRED:-results/branch_predictions/four_branch_base_predictions.csv}"
CHEM_PRED="${CHEM_PRED:-results/branch_predictions/chem_hgb_predictions.csv}"

if [[ ! -f "${BASE_PRED}" ]]; then
  echo "Missing ${BASE_PRED}" >&2
  echo "Generate branch-score tables first, or set BASE_PRED=/path/to/four_branch_base_predictions.csv" >&2
  exit 1
fi

if [[ ! -f "${CHEM_PRED}" ]]; then
  echo "Missing ${CHEM_PRED}" >&2
  echo "Generate the chemistry branch table first, or set CHEM_PRED=/path/to/chem_hgb_predictions.csv" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/fit_abagym_fusion_model_zoo.py \
  --base "${BASE_PRED}" \
  --feature foldx="${BASE_PRED}":foldx \
  --feature rsa="${BASE_PRED}":rsa \
  --feature struct="${BASE_PRED}":struct \
  --feature esm_lora_listwise="${BASE_PRED}":esm_lora_listwise \
  --feature chem_hgb="${CHEM_PRED}":chem_hgb \
  --out-dir "${OUT_DIR}" \
  --target rank_target \
  --models monotonic_hgb

"${PYTHON_BIN}" scripts/score_abagym_predictions_like_paper.py \
  --predictions "${OUT_DIR}/monotonic_hgb/predictions.csv" \
  --score-column prediction \
  --out "${OUT_DIR}/monotonic_hgb/paper_metrics.json" \
  > /tmp/mirage_reproduced_score_stdout.json

echo "MIRAGE final-fusion training completed."
echo "Predictions: ${OUT_DIR}/monotonic_hgb/predictions.csv"
echo "Metrics: ${OUT_DIR}/monotonic_hgb/paper_metrics.json"
