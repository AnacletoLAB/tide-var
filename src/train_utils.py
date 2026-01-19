from pathlib import Path
import re
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tremm.data.mendelian_dataset import DatasetEntry, ExternalFold, InternalFold
from tremm.experiments.utils.loaders import get_train_loader, get_valid_loader, get_test_loader


def print_scores(scores: dict, floatfmt: str = ".4f"):
    rows = [(k, float(scores[k])) for k in sorted(scores)]
    print(
        tabulate(
            rows,
            headers=["Metric", "Value"],
            tablefmt="github",
            floatfmt=floatfmt,
        )
    )

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
    std = train.X.std(0, unbiased=False, keepdim=True)
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
        m = re.search(r"(\d+_internal)", model_path.name)
        if m:
            split_name = m.group(1)
    else:
        split_name = model_path.stem

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


def load_datasets(
    dataset_folder_size: str,
    dataset_folder_name: str,
    model_path: Path,
    load_test: bool = True,
    load_test_only: bool = False,
) -> Tuple[DatasetEntry, DatasetEntry] | Tuple[DatasetEntry, DatasetEntry, DatasetEntry]:
    internal = load_internal(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        model_path=model_path,
        regex=True,
    )
    mean, std = compute_normalization_stats(internal.train)
    train_set = apply_normalization(internal.train, mean, std)
    valid_set = apply_normalization(internal.valid, mean, std)

    if load_test or load_test_only:
        external = load_external(
            dataset_folder_size=dataset_folder_size,
            dataset_folder_name=dataset_folder_name,
        )
        test_set = apply_normalization(external.test, mean, std)
        if load_test_only:
            return test_set
        return train_set, valid_set, test_set

    return train_set, valid_set


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
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        pos_probs = probs[:, 1]

    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()
    pos_probs = pos_probs.cpu().numpy()

    return targets, preds, pos_probs


def inference_on_dataloader(
    model: torch.nn.Module,
    valid_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model.to(device)
    loader = valid_loader if valid_loader is not None else test_loader
    assert loader is not None, "Either valid_loader or test_loader must be provided"

    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)
            outputs = model(inputs)

            if outputs.dim() == 3:
                probs = torch.softmax(outputs, dim=-1).mean(dim=1)
                preds = probs.argmax(dim=1)
                pos_probs = probs[:, 1]
            else:
                probs = torch.softmax(outputs, dim=-1)
                preds = probs.argmax(dim=1)
                pos_probs = probs[:, 1]

            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(pos_probs.cpu())

    targets = torch.cat(all_targets).numpy()
    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()

    return targets, preds, probs


def extract_cfg(model: nn.Module) -> dict:
    input_size = getattr(model, "input_size", None)
    if input_size is None and hasattr(model, "input_layer"):
        input_size = model.input_layer.in_features

    units = getattr(model, "units", None)
    if units is None and hasattr(model, "input_layer"):
        units = model.input_layer.out_features

    num_classes = getattr(model, "num_classes", None)
    if num_classes is None and hasattr(model, "classifier"):
        num_classes = model.classifier.out_features

    num_groups = getattr(model, "num_groups", len(model.groups))
    blocks_in_group = getattr(model, "blocks_in_group", None)
    if blocks_in_group is None and hasattr(model, "groups") and len(model.groups) > 0:
        linear_layers = sum(
            1 for layer in model.groups[0] if isinstance(layer, nn.Linear)
        )
        blocks_in_group = linear_layers

    focal = bool(getattr(model, "focal", False))

    act_name, dropout_p, batch_norm = None, None, False
    if hasattr(model, "groups") and len(model.groups) > 0:
        for layer in model.groups[0]:
            if isinstance(layer, nn.Dropout):
                dropout_p = layer.p
            elif isinstance(layer, nn.BatchNorm1d):
                batch_norm = True
            else:
                if layer.__class__.__name__ in (
                    "ReLU",
                    "LeakyReLU",
                    "ELU",
                    "SiLU",
                    "GELU",
                    "PReLU",
                ):
                    act_name = layer.__class__.__name__

    return {
        "input_size": input_size,
        "num_classes": num_classes,
        "num_groups": num_groups,
        "blocks_in_group": blocks_in_group,
        "units": units,
        "activation_fn": act_name,
        "dropout": dropout_p,
        "batch_norm": batch_norm,
        "focal": focal,
    }
