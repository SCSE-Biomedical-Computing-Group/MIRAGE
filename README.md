# MIRAGE

MIRAGE is a reproducibility package for antibody-antigen DMS mutation ranking
on AbAgym. The prediction task is **within-DMS mutation ranking**, not absolute
binding free-energy regression.

The final MIRAGE model is a leave-antigen-family-out monotonic
histogram-gradient-boosting calibrator over five branch scores:

1. `foldx`: clipped FoldX complex mutation-energy score.
2. `rsa`: negative relative solvent accessibility.
3. `struct`: learned structural-context ranker score.
4. `esm_lora_listwise`: ESM2-LoRA listwise mutation-ranker score.
5. `chem_hgb`: chemistry plus local-structure HGB branch score.

## What This Repository Reproduces

This repository supports two levels of reproducibility.

**Exact final-model reproduction.**
The included branch-score tables are sufficient to retrain the final
five-branch MIRAGE monotonic-HGB model and reproduce the archived final
prediction table byte-for-byte.

**Branch-level reruns.**
The branch-training scripts and processed AbAgym feature tables are included
for inspection and reruns. Some branch regeneration requires external raw
assets that are not stored here, such as FoldX output directories, official
modeled structures, and downloaded ESM2 weights.

## Quick Start: Reproduce Final MIRAGE Training

Install the lightweight metric/fusion dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Train the final MIRAGE fusion model from the included five branch-score tables
and verify that it exactly matches the archived result:

```bash
bash scripts/reproduce_final_fusion.sh
```

On the original server, use the existing environment:

```bash
PYTHON_BIN=/home2/s230112/envs/aapred/bin/python bash scripts/reproduce_final_fusion.sh
```

Expected output ends with:

```text
MIRAGE archived-result check: PASS
MIRAGE final-fusion training reproduction: PASS
```

## Verify Archived Final Predictions Only

To rescore the archived final prediction table without retraining:

```bash
bash scripts/verify_archived_results.sh
```

Expected metrics:

```text
studies: 68
records: 36504
average_spearman: 0.3610019854184025
average_roc_auc: 0.763351458609284
weighted_spearman: 0.3115013507265664
weighted_roc_auc: 0.7431936528462341
```

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
    fit_abagym_fusion_model_zoo.py
    score_abagym_predictions_like_paper.py
    check_archived_metrics.py
    verify_archived_results.sh
    reproduce_final_fusion.sh

    train_abagym_retrieval_chem_fusion.py
    train_abagym_study_disjoint_ranknet.py
    train_abagym_esm_adapter_ranker.py
    evaluate_abagym_foldx_scores.py
    evaluate_abagym_rsa_baseline.py
    build_abagym_structure_features.py
    prepare_abagym_study_disjoint.py

  results/
    branch_predictions/
      four_branch_base_predictions.csv
      four_branch_base_summary.json
      chem_hgb_predictions.csv
      chem_hgb_summary.json

    final/
      mirage_monohgb_predictions.csv
      mirage_monohgb_summary.json
      mirage_monohgb_paper_metrics.json
      mirage_monohgb_per_study_metrics.csv
```

## Full Branch Reruns

Install the optional branch-training dependencies:

```bash
python -m pip install -r requirements-train.txt
```

Branch reruns are intentionally separated from the exact final-fusion
reproduction because some inputs are external:

- FoldX branch requires the AbAgym FoldX output directory containing
  `complex_ddG_values.csv` files.
- RSA branch requires modeled antibody-antigen PDB structures.
- ESM2-LoRA branch requires HuggingFace/PyTorch and an available ESM2
  checkpoint, for example `facebook/esm2_t30_150M_UR50D`.
- The chemistry branch can be rerun from the packaged records, packaged
  structure features, and the four-branch prediction table.

The final paper result is reproduced exactly by `scripts/reproduce_final_fusion.sh`.
