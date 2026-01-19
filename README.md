# tide-var
This repository includes the python code of *TIDE-Var* and of *T-ResNet*, two deep neural models for the prediction of genetic variants in Mandelian diseases.

*TIDE-Var*  is an implicit ensemble of deep neural networks that partially share parameters
and in parallel contextually optimize the same objective function, borrowed from the recently proposed TabM model (Gorishniy et al., ICLR, 2025).

*T-ResNet* (Tabular Residual Neural Network) adopts a modular architecture with residual connections to
support efficient hyperparameter optimization, along with a mini-batch balancing strategy to address
class imbalance.

Datasets used for training the model are constiduted by 26-dimensional genetic, epigenetic and conservation features originally collected by Smedley et al, Am. J. Hum. Genet. 99, 595–606 (2016).
Dataset splits used in this study are avaialble
in `src/tremm/data/tensors/<dataset_folder>/<dataset_name>/`.

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
