# tide-var

Train TabM and ResNet models on the Mendelian dataset. Dataset splits are stored
under `src/tremm/data/tensors/<dataset_folder>/<dataset_name>/`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

This installs the `tremm` CLI.

## Data and datasets

The CLI downloads raw data into `src/tremm/data/raw_data` on first use.

```bash
tremm create_kfold --dataset_size 100000 --ext_test_ratio 0.2 --int_valid_ratio 0.2 --random_seed 42
tremm create_dataset --dataset_size 100000 --split_ratios 0.7 0.15 0.15 --positive_ratios 0.5 0.25 0.25 --random_seed 42
tremm create_n_kfold --dataset_size 100000 --ext_test_ratio 0.2 --global_valid_ratio 0.1 --num_internal_folds 5 --int_valid_ratio 0.2 --random_seed 42
tremm list
```

If you do not want to install the package:

```bash
PYTHONPATH=src python -m tremm.scripts.cli <subcommand> [args...]
```

## Train

TabM (uses CUDA/MPS if available, otherwise CPU):

```bash
python src/train_tide_var.py -d kfold_100k -n <dataset_name> -e 25 -k 1 -m ple -de 12
```

Single-fold / single-worker run:

```bash
python src/train_tide_var.py -d kfold_100k -n <dataset_name> -e 25 -k 1 -m ple -de 12 --fold 0 --workers 1
```

Results are saved as CSVs under `src/csv_results/`.

ResNet:

1. Update `DATASET_SPECS` in `src/train_resnet.py`.
2. Run:

```bash
python src/train_resnet.py
```
