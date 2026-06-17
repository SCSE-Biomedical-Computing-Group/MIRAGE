# MIRAGE Manifest

This package contains source data, processed structural descriptors, training
scripts, and environment files for MIRAGE. Generated prediction and metric
files are intentionally not tracked.

## Environment

- `environment.yml`
  - Conda environment for MIRAGE training and evaluation.
- `requirements.txt`
  - Pip dependency list for the same workflow.
- `EXTERNAL_ASSETS.md`
  - Required external FoldX, PDB-structure, and ESM2 model assets for full
    train-from-scratch reproduction.

## Processed Data

- `data/abagym_interface_study_rank_records.csv`
  - Standardized AbAgym mutation records with DMS scores, rank targets,
    mutation identity, DMS metadata, and antigen-family split labels.
  - Contains 36,541 source records.
- `data/abagym_official_structure_features.npz`
  - 19-dimensional local structural descriptors used by learned structural
    and chemistry branches.
- `data/abagym_official_structure_features.json`
  - Metadata for the structural feature matrix.
- `data/abagym_study_summary.csv`
  - DMS study-level summary.
- `data/abagym_prepare_audit.json`
  - Dataset preparation audit.

## Scripts

- `scripts/prepare_abagym_study_disjoint.py`
  - AbAgym preprocessing script.
- `scripts/build_abagym_structure_features.py`
  - Structural feature construction script.
- `scripts/evaluate_abagym_foldx_scores.py`
  - FoldX branch scoring script.
- `scripts/evaluate_abagym_rsa_baseline.py`
  - RSA branch scoring script.
- `scripts/train_abagym_study_disjoint_ranknet.py`
  - Learned structural-context branch training script.
- `scripts/train_abagym_esm_adapter_ranker.py`
  - ESM2-LoRA branch training script.
- `scripts/train_abagym_retrieval_chem_fusion.py`
  - Chemistry/local-structure branch training script.
- `scripts/build_mirage_branch_tables.py`
  - Utility that merges FoldX, RSA, structural, and ESM2-LoRA branch
    predictions into the four-branch base table required by MIRAGE fusion.
- `scripts/fit_abagym_fusion_model_zoo.py`
  - Final fusion training script.
- `scripts/score_abagym_predictions_like_paper.py`
  - Evaluation script.
- `scripts/train_mirage_from_scratch.sh`
  - End-to-end training wrapper for the FoldX, RSA, structural, ESM2-LoRA,
    chemistry, and final monotonic-HGB fusion stages.
- `scripts/reproduce_final_fusion.sh`
  - Convenience wrapper for final fusion training and evaluation once branch
    score tables have been generated.

## Generated Outputs

The following directories are placeholders for user-generated outputs and are
ignored by Git except for `.gitkeep` files:

- `results/branch_predictions/`
- `results/final/`
- `results/reproduced_final_fusion/`

## External Assets Not Included

- Raw AbAgym modeled PDB structures.
- Raw FoldX output directories.
- Downloaded HuggingFace ESM2 weights.
- Local GPU checkpoints from exploratory training runs.
