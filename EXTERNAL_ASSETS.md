# External Assets for Full MIRAGE Training

MIRAGE includes the processed AbAgym mutation table and the 19-dimensional
local structural feature matrix. A full train-from-scratch run also needs the
following external assets.

## FoldX Outputs

Set `FOLDX_DIR` to a directory containing one subdirectory per modeled complex.
Each subdirectory should contain:

```text
complex_ddG_values.csv
```

The FoldX scoring script joins rows using the AbAgym `PDB_file` and
`mut_names` columns.

## Modeled Antibody-Antigen Structures

Set `STRUCTURE_DIR` to the directory containing the AbAgym modeled
antibody-antigen PDB structures. The expected files are named by the
`PDB_file` column in `data/abagym_interface_study_rank_records.csv`, for
example:

```text
G6_27_30A_corrected_4zfg.pdb
AZD1061_7l7e.pdb
```

These structures are used by:

- `scripts/evaluate_abagym_rsa_baseline.py`
- `scripts/train_abagym_esm_adapter_ranker.py`

## ESM2 Weights

The ESM2-LoRA branch loads:

```text
facebook/esm2_t30_150M_UR50D
```

The model can be downloaded automatically by HuggingFace Transformers, or used
from a local cache with:

```bash
ESM_LOCAL_FILES_ONLY=1 bash scripts/train_mirage_from_scratch.sh
```

## One-Command Run

After creating the conda environment, run:

```bash
FOLDX_DIR=/path/to/abagym_foldx_outputs \
STRUCTURE_DIR=/path/to/abagym_modeled_pdbs \
DEVICE=cuda:0 \
bash scripts/train_mirage_from_scratch.sh
```

The script writes generated outputs under `results/`.
