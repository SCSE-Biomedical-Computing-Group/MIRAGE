# MIRAGE Manifest

This package contains the processed inputs, branch-score tables, scripts, and
archived outputs needed to retrain and evaluate the final MIRAGE fusion model.

## Exact Reproduction Entry Point

- `scripts/reproduce_final_fusion.sh`
  - Retrains the final five-feature monotonic-HGB fusion model from included
    branch-score tables.
  - Scores the reproduced predictions.
  - Verifies the reproduced predictions match
    `results/final/mirage_monohgb_predictions.csv` exactly.

## Processed Data

- `data/abagym_interface_study_rank_records.csv`
  - Standardized AbAgym mutation records with DMS scores, rank targets,
    mutation identity, DMS metadata, and antigen-family split labels.
- `data/abagym_official_structure_features.npz`
  - 19-dimensional local structural descriptors used by learned structural
    and chemistry branches.
- `data/abagym_official_structure_features.json`
  - Metadata for the structural feature matrix.
- `data/abagym_study_summary.csv`
  - DMS study-level summary.
- `data/abagym_prepare_audit.json`
  - Dataset preparation audit.

## Branch-Score Tables

- `results/branch_predictions/four_branch_base_predictions.csv`
  - FoldX, RSA, learned structural-context, and ESM2-LoRA listwise branch
    scores aligned by `sample_id`.
- `results/branch_predictions/chem_hgb_predictions.csv`
  - Chemistry plus local-structure branch score aligned by `sample_id`.

These two tables are sufficient to retrain the final MIRAGE monotonic-HGB
fusion model exactly.

## Final Outputs

- `results/final/mirage_monohgb_predictions.csv`
  - Archived final MIRAGE predictions.
- `results/final/mirage_monohgb_summary.json`
  - Final model summary.
- `results/final/mirage_monohgb_paper_metrics.json`
  - Archived aggregate paper metrics.
- `results/final/mirage_monohgb_per_study_metrics.csv`
  - Archived per-DMS metrics.

## Scripts

- `scripts/fit_abagym_fusion_model_zoo.py`
  - Final fusion training script.
- `scripts/score_abagym_predictions_like_paper.py`
  - Evaluation script.
- `scripts/check_archived_metrics.py`
  - Metric and prediction-hash checker.
- `scripts/verify_archived_results.sh`
  - Archived-output verification only.
- `scripts/train_abagym_retrieval_chem_fusion.py`
  - Chemistry/local-structure branch training script.
- `scripts/train_abagym_study_disjoint_ranknet.py`
  - Learned structural-context branch training script.
- `scripts/train_abagym_esm_adapter_ranker.py`
  - ESM2-LoRA branch training script.
- `scripts/evaluate_abagym_foldx_scores.py`
  - FoldX branch scoring script.
- `scripts/evaluate_abagym_rsa_baseline.py`
  - RSA branch scoring script.
- `scripts/build_abagym_structure_features.py`
  - Structural feature construction script.
- `scripts/prepare_abagym_study_disjoint.py`
  - AbAgym preprocessing script.

## External Assets Not Included

- Raw AbAgym modeled PDB structures.
- Raw FoldX output directories.
- Downloaded HuggingFace ESM2 weights.
- Local GPU checkpoints from exploratory training runs.
