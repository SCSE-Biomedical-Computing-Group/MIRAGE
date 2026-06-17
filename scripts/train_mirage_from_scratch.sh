#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    PYTHON_BIN=python3
  fi
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RECORDS="${RECORDS:-data/abagym_interface_study_rank_records.csv}"
STRUCTURE_FEATURES="${STRUCTURE_FEATURES:-data/abagym_official_structure_features.npz}"
RESULTS_DIR="${RESULTS_DIR:-results}"
BRANCH_DIR="${BRANCH_DIR:-${RESULTS_DIR}/branch_predictions}"
FINAL_DIR="${FINAL_DIR:-${RESULTS_DIR}/final/mirage_chem_monohgb}"
DEVICE="${DEVICE:-cuda:0}"

FOLDX_DIR="${FOLDX_DIR:-}"
STRUCTURE_DIR="${STRUCTURE_DIR:-}"
STRUCTURE_ID_COLUMN="${STRUCTURE_ID_COLUMN:-PDB_file}"
ESM_MODEL_NAME="${ESM_MODEL_NAME:-facebook/esm2_t30_150M_UR50D}"
ESM_LOCAL_FILES_ONLY="${ESM_LOCAL_FILES_ONLY:-0}"
SAVE_ESM_CHECKPOINTS="${SAVE_ESM_CHECKPOINTS:-0}"

missing=0
for path_var in RECORDS STRUCTURE_FEATURES; do
  path="${!path_var}"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${path_var}: ${path}" >&2
    missing=1
  fi
done
if [[ -z "${FOLDX_DIR}" || ! -d "${FOLDX_DIR}" ]]; then
  echo "Set FOLDX_DIR to the AbAgym FoldX output directory containing complex_ddG_values.csv files." >&2
  missing=1
fi
if [[ -z "${STRUCTURE_DIR}" || ! -d "${STRUCTURE_DIR}" ]]; then
  echo "Set STRUCTURE_DIR to the AbAgym modeled antibody-antigen PDB directory." >&2
  missing=1
fi
if [[ "${missing}" -ne 0 ]]; then
  echo "Example:" >&2
  echo "  FOLDX_DIR=/path/to/foldx_outputs STRUCTURE_DIR=/path/to/modeled_pdbs bash scripts/train_mirage_from_scratch.sh" >&2
  exit 1
fi

mkdir -p "${BRANCH_DIR}" "${FINAL_DIR}"

echo "[1/7] FoldX energy branch"
"${PYTHON_BIN}" scripts/evaluate_abagym_foldx_scores.py \
  --records "${RECORDS}" \
  --foldx-dir "${FOLDX_DIR}" \
  --score-file complex_ddG_values.csv \
  --out-dir "${BRANCH_DIR}/foldx_complex_ddg"

echo "[2/7] RSA solvent-exposure branch"
"${PYTHON_BIN}" scripts/evaluate_abagym_rsa_baseline.py \
  --records "${RECORDS}" \
  --structure-dir "${STRUCTURE_DIR}" \
  --structure-id-column "${STRUCTURE_ID_COLUMN}" \
  --out-dir "${BRANCH_DIR}/rsa_baseline"

echo "[3/7] Learned structural-context branch"
"${PYTHON_BIN}" scripts/train_abagym_study_disjoint_ranknet.py \
  --records "${RECORDS}" \
  --structure-features "${STRUCTURE_FEATURES}" \
  --out-dir "${BRANCH_DIR}/struct_ranknet" \
  --device "${DEVICE}" \
  --seed 42 \
  --epochs 30 \
  --loss pairwise \
  --holdout-column antigen_name

echo "[4/7] ESM2-LoRA listwise branch"
esm_args=()
if [[ "${ESM_LOCAL_FILES_ONLY}" == "1" ]]; then
  esm_args+=(--local-files-only)
fi
if [[ "${SAVE_ESM_CHECKPOINTS}" == "1" ]]; then
  esm_args+=(--save-checkpoints)
fi
"${PYTHON_BIN}" scripts/train_abagym_esm_adapter_ranker.py \
  --records "${RECORDS}" \
  --structure-dir "${STRUCTURE_DIR}" \
  --structure-id-column "${STRUCTURE_ID_COLUMN}" \
  --structure-features "${STRUCTURE_FEATURES}" \
  --out-dir "${BRANCH_DIR}/esm_lora_listwise" \
  --model-name "${ESM_MODEL_NAME}" \
  --device "${DEVICE}" \
  --seed 17 \
  --max-residues 384 \
  --epochs 3 \
  --train-last-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-targets query,value \
  --loss listwise \
  --holdout-column antigen_name \
  --amp \
  "${esm_args[@]}"

echo "[5/7] Build four-branch base table"
"${PYTHON_BIN}" scripts/build_mirage_branch_tables.py \
  --records "${RECORDS}" \
  --foldx "${BRANCH_DIR}/foldx_complex_ddg/predictions.csv" \
  --rsa "${BRANCH_DIR}/rsa_baseline/predictions.csv" \
  --struct "${BRANCH_DIR}/struct_ranknet/predictions.csv" \
  --esm-lora "${BRANCH_DIR}/esm_lora_listwise/predictions.csv" \
  --out "${BRANCH_DIR}/four_branch_base_predictions.csv" \
  --summary "${BRANCH_DIR}/four_branch_base_summary.json"

echo "[6/7] Chemistry/local-structure HGB branch"
"${PYTHON_BIN}" scripts/train_abagym_retrieval_chem_fusion.py \
  --records "${RECORDS}" \
  --feature-table "${BRANCH_DIR}/four_branch_base_predictions.csv" \
  --structure-npz "${STRUCTURE_FEATURES}" \
  --out-dir "${BRANCH_DIR}/chem_hgb" \
  --target rank_target \
  --retrieval-k 64

cp "${BRANCH_DIR}/chem_hgb/predictions.csv" "${BRANCH_DIR}/chem_hgb_predictions.csv"

echo "[7/7] Final five-branch monotonic HGB fusion"
BASE_PRED="${BRANCH_DIR}/four_branch_base_predictions.csv" \
CHEM_PRED="${BRANCH_DIR}/chem_hgb_predictions.csv" \
OUT_DIR="${FINAL_DIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash scripts/reproduce_final_fusion.sh

echo "Full MIRAGE scratch training completed."
echo "Four-branch table: ${BRANCH_DIR}/four_branch_base_predictions.csv"
echo "Chemistry table: ${BRANCH_DIR}/chem_hgb_predictions.csv"
echo "Final metrics: ${FINAL_DIR}/monotonic_hgb/paper_metrics.json"
