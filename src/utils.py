from numpy.random import f
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from scipy import special

from pathlib import Path
import re
from typing import Tuple, Optional, Union
import numpy as np
from tabulate import tabulate 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score

from tremm.data.mendelian_dataset import InternalFold, ExternalFold
from tremm.experiments.utils.loaders import get_train_loader, get_valid_loader, get_test_loader
from tremm.data.mendelian_dataset import DatasetEntry

def print_scores(scores: dict, floatfmt: str = ".4f"):
    rows = [(k, float(scores[k])) for k in sorted(scores)]
    print(tabulate(
        rows,
        headers=["Metric", "Value"],
        tablefmt="github",
        floatfmt=floatfmt
    ))

def get_scores(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
):


    scores = {
        "accuracy": lambda: accuracy_score(targets, preds),
        "f1": lambda: f1_score(targets, preds, pos_label=1),
        "recall": lambda: recall_score(targets, preds, pos_label=1),
        "precision": lambda: precision_score(targets, preds, pos_label=1),
        "auprc": lambda: average_precision_score(targets, probs),
    }

    results = {}
    for metric in scores.keys():
        results[metric] = scores[metric]()
    
    return results

def compute_normalization_stats(train: DatasetEntry):
    mean = train.X.mean(0, keepdim=True)
    std  = train.X.std(0, unbiased=False, keepdim=True)
    return mean, std
    
def apply_normalization(split: DatasetEntry, mean: torch.Tensor, std: torch.Tensor):
    device = split.X.device
    mean = mean.to(device=device, dtype=split.X.dtype)
    std = std.to(device=device, dtype=split.X.dtype)
    split.X = (split.X - mean) / std
    return split

def load_internal(
    dataset_folder_size: str,
    dataset_folder_name: str,
    model_path: Path,
    regex: bool = True,
) -> InternalFold:
    datasets_path = (
        Path(__file__).resolve().parent
        / "tremm"
        / "data"
        / "tensors"
        / dataset_folder_size
        / dataset_folder_name
    )
    internal = None 
    split_name = None
    
    if regex:
        m = re.search(r'(\d+_internal)', model_path.name)
        if m:
            split_name = m.group(1)
    
    assert split_name is not None, "split_name is None"
    for dataset in datasets_path.iterdir():
        if split_name in dataset.name:
            safe_ctx = getattr(torch.serialization, "safe_globals", None)
            if safe_ctx is not None:
                with safe_ctx([InternalFold]):
                    internal = torch.load(dataset)
            else:
                torch.serialization.add_safe_globals([InternalFold])
                internal = torch.load(dataset)
    assert internal is not None, "internal is None"
    return internal

def load_external(
    dataset_folder_size: str,
    dataset_folder_name: str,
) -> ExternalFold:    
    datasets_path = (
        Path(__file__).resolve().parent
        / "tremm"
        / "data"
        / "tensors"
        / dataset_folder_size
        / dataset_folder_name
    )
    external = None 
    
    for dataset in datasets_path.iterdir():
        if "external" in dataset.name:
            safe_ctx = getattr(torch.serialization, "safe_globals", None)
            if safe_ctx is not None:
                with safe_ctx([ExternalFold]):
                    external = torch.load(dataset, map_location="cpu")
            else:
                torch.serialization.add_safe_globals([ExternalFold])
                external = torch.load(dataset, map_location="cpu")
            
    assert external is not None, "external is None"
    return external

def load_datasets(dataset_folder_size: str,
                  dataset_folder_name: str,
                  model_path: Path):
    internal = load_internal(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        model_path=model_path,
        regex=True,
    )
    external = load_external(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
    )
    mean, std = compute_normalization_stats(internal.train)
    train_set = apply_normalization(internal.train, mean, std)
    valid_set = apply_normalization(internal.valid, mean, std)
    test_set  = apply_normalization(external.test,  mean, std)
    return train_set, valid_set, test_set


def inference(
    model: torch.nn.Module,
    valid_set=None,
    test_set=None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    
    inputs, targets = None, None
    if valid_set is not None:
        inputs, targets = valid_set.X, valid_set.y
    if test_set is not None:
        inputs, targets = test_set.X, test_set.y
    assert inputs is not None and targets is not None, "inputs or targets is None"
    
    model.eval()
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        probs = probs.cpu().numpy()
    
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()
    
    return targets, preds, probs
    
def inference_on_dataloader(
    model: torch.nn.Module,
    valid_loader: Optional[DataLoader] = None,
    test_loader:  Optional[DataLoader] = None,
    device:      str               = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    loader = valid_loader if valid_loader is not None else test_loader
    assert loader is not None, "Either valid_loader or test_loader must be provided"

    model.eval()
    all_targets = []
    all_preds   = []

    with torch.no_grad():
        # Compute the mean of the k predictions.
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)
            outputs = model(inputs)
            all_targets.append(targets.cpu())
            all_preds.append(outputs.cpu())

    targets = torch.cat(all_targets).numpy()
    preds   = torch.cat(all_preds).numpy()     # shape: (16000, 32, 2), i.e. (datapts, k, n_classes)
    
    # https://github.com/yandex-research/tabm/blob/main/example.ipynb 
    # For classification, the mean must be computed in the probability space.
    preds = special.softmax(preds, axis=-1)
    preds = preds.mean(1) # shape: (16000, 2)
    probs = preds[:,1] # probs.shape = (16000,)
    preds = preds.argmax(1)

    return targets, preds, probs
