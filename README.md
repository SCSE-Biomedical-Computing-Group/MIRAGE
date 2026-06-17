# MIRAGE

MIRAGE is a training package for antibody-antigen DMS mutation ranking on
AbAgym. The prediction task is **within-DMS mutation ranking**, not absolute
binding free-energy regression.

The final MIRAGE model is a leave-antigen-family-out monotonic
histogram-gradient-boosting calibrator over five branch scores:

1. `foldx`: clipped FoldX complex mutation-energy score.
2. `rsa`: negative relative solvent accessibility.
3. `struct`: learned structural-context ranker score.
4. `esm_lora_listwise`: ESM2-LoRA listwise mutation-ranker score.
5. `chem_hgb`: chemistry plus local-structure HGB branch score.

## Environment

Create the MIRAGE Conda environment:

```bash
conda env create -f environment.yml
conda activate mirage
```

Alternatively, install the same dependencies with pip:

```bash
python3 -m pip install -r requirements.txt
```

## Data Included

The packaged AbAgym mutation table contains 36,541 source records:

```text
data/abagym_interface_study_rank_records.csv
```

The package also includes processed local structural descriptors:

```text
data/abagym_official_structure_features.npz
data/abagym_official_structure_features.json
```

Generated predictions and metrics are intentionally **not tracked**. They are
written under `results/` when users run the training scripts.

## Training Workflow

MIRAGE training is staged. First generate branch-score tables, then train the
final fusion model.

Expected branch-score output paths:

```text
results/branch_predictions/four_branch_base_predictions.csv
results/branch_predictions/chem_hgb_predictions.csv
```

The final fusion script consumes those two tables:

```bash
bash scripts/reproduce_final_fusion.sh
```

It writes:

```text
results/reproduced_final_fusion/monotonic_hgb/predictions.csv
results/reproduced_final_fusion/monotonic_hgb/paper_metrics.json
results/reproduced_final_fusion/monotonic_hgb/paper_metrics.csv
results/reproduced_final_fusion/monotonic_hgb/summary.json
```

If your branch-score files are elsewhere, set:

```bash
BASE_PRED=/path/to/four_branch_base_predictions.csv \
CHEM_PRED=/path/to/chem_hgb_predictions.csv \
bash scripts/reproduce_final_fusion.sh
```

## Branch Training Inputs

Some branches require external assets that are not stored in this repository.

- FoldX branch requires the AbAgym FoldX output directory containing
  `complex_ddG_values.csv` files.
- RSA branch requires modeled antibody-antigen PDB structures.
- ESM2-LoRA branch requires modeled antibody-antigen PDB structures,
  HuggingFace/PyTorch access to `facebook/esm2_t30_150M_UR50D`, and a GPU for
  practical runtime.
- The chemistry branch can be rerun from the packaged records, packaged
  structure features, and a four-branch base prediction table.

## Repository Layout

```text
MIRAGE/
  data/
    abagym_interface_study_rank_records.csv
    abagym_official_structure_features.npz
    abagym_official_structure_features.json
    abagym_study_summary.csv
    abagym_prepare_audit.json

  scripts/
    prepare_abagym_study_disjoint.py
    build_abagym_structure_features.py
    evaluate_abagym_foldx_scores.py
    evaluate_abagym_rsa_baseline.py
    train_abagym_study_disjoint_ranknet.py
    train_abagym_esm_adapter_ranker.py
    train_abagym_retrieval_chem_fusion.py
    fit_abagym_fusion_model_zoo.py
    score_abagym_predictions_like_paper.py
    reproduce_final_fusion.sh

  results/
    branch_predictions/
    final/
```

## Notes

The final evaluated row count can be smaller than 36,541 because the final
fusion model uses the intersection of records with all required branch scores.
