#!/usr/bin/env python3
"""
Train single ResNet models either across internal folds or using the external
(train/test) split, saving checkpoints and metrics for each experiment.

Select the desired behaviour per dataset via ``DATASET_SPECS`` below.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = SCRIPT_DIR.parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

NN_ENSEMBLE_DIR = SCRIPT_DIR.parent / "nn_ensemble"
if NN_ENSEMBLE_DIR.exists():
    sys.path.insert(0, str(NN_ENSEMBLE_DIR))

from utilsz import (  # type: ignore
    apply_normalization,
    compute_normalization_stats,
    extract_cfg,
    get_scores,
    get_valid_loader,
    inference_on_dataloader,
    load_datasets,
    load_external,
    load_internal,
)
from resnet_modeling import ResNet  # type: ignore
from tremm.data.mendelian_dataset import DatasetEntry  # type: ignore
from tremm.experiments.utils.loaders import get_train_loader

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUNS_ROOT = SCRIPT_DIR / "single_model_runs"


@dataclass
class DatasetSpec:
    dataset_folder_size: str
    dataset_folder_name: str
    mode: str  # "internal" or "external"
    max_epochs: int = 30
    base_seed: int = 12345
    internal_folds: Optional[Iterable[str]] = None
    score_interval: int = 2


DATASET_SPECS: List[DatasetSpec] = [
    # DatasetSpec(
    #     dataset_folder_size="kfold_full",
    #     dataset_folder_name="29-07-25_22-07_rand_seed_1",
    #     mode="external",
    #     max_epochs=40,
    #     base_seed=12345,
    # ),
    # DatasetSpec(
    #     dataset_folder_size="kfold_full",
    #     dataset_folder_name="05-08-25_09-41_rand_seed_5",
    #     mode="external",
    #     max_epochs=40,
    #     base_seed=12345,
    # ),
    # DatasetSpec(
    #     dataset_folder_size="kfold_full",
    #     dataset_folder_name="29-07-25_22-08_rand_seed_10",
    #     mode="external",
    #     max_epochs=40,
    #     base_seed=12345,
    # ),
    # DatasetSpec(
    #     dataset_folder_size="kfold_full",
    #     dataset_folder_name="14-02-25_16-02_rand_seed_42",
    #     mode="external",
    #     max_epochs=40,
    #     base_seed=12345,
    # ),
    # DatasetSpec(
    #     dataset_folder_size="kfold_full",
    #     dataset_folder_name="29-07-25_22-09_rand_seed_100",
    #     mode="external",
    #     max_epochs=40,
    #     base_seed=12345,
    # ),

    # DatasetSpec(
    #     dataset_folder_size="kfold_100k",
    #     dataset_folder_name="14-02-25_12-56_rand_seed_42",
    #     mode="external",
    #     max_epochs=30,
    #     base_seed=54321,
    # ),

    # DatasetSpec(
    #     dataset_folder_size="kfold_1m",
    #     dataset_folder_name="19-02-25_09-55_rand_seed_42",
    #     mode="internal",
    #     max_epochs=30,
    #     base_seed=54321,
    # ),    

    DatasetSpec(
        dataset_folder_size="kfold_full",
        dataset_folder_name="14-02-25_16-02_rand_seed_42_a",
        mode="external",
        max_epochs=40,
        base_seed=12345,
    ),
    
    # DatasetSpec(
    #     dataset_folder_size="kfold_100k",
    #     dataset_folder_name="14-02-25_12-56_rand_seed_42_a",
    #     mode="external",
    #     max_epochs=30,
    #     base_seed=12345,
    # ), 

]


def print_scores(title: str, scores: Dict[str, float], floatfmt: str = ".4f") -> None:
    rows = [(k, float(scores[k])) for k in sorted(scores)]
    print(title)
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github", floatfmt=floatfmt))


def _shuffle_dataset(ds: DatasetEntry, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ds.y), generator=generator).numpy()

    def _reindex(x):
        if torch.is_tensor(x):
            return x[torch.from_numpy(perm)]
        return x[perm]

    ds.X = _reindex(ds.X)
    ds.y = _reindex(ds.y)

    for attr in ("ids", "weights", "mask"):
        if hasattr(ds, attr):
            setattr(ds, attr, _reindex(getattr(ds, attr)))


def _make_run_dir(dataset_folder_size: str, dataset_folder_name: str, mode: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_dataset = f"{dataset_folder_size}-{dataset_folder_name}".replace(os.sep, "-")
    run_dir = RUNS_ROOT / f"single_resnet_{mode}_{safe_dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_and_evaluate(
    train_set: DatasetEntry,
    valid_set: Optional[DatasetEntry],
    max_epochs: int,
    model_seed: Optional[int] = None,
    data_seed: Optional[int] = None,
    score_interval: int = 0,
    score_loader: Optional[torch.utils.data.DataLoader] = None,
    score_label: str = "validation",
):
    if data_seed is not None:
        np.random.seed(data_seed)

    train_loader = get_train_loader(
        train_set,
        balanced=True,
        batch_size=32768,
        neg_batch_ratio=0.95,
    )
    valid_loader = (
        get_valid_loader(valid_set, batch_size=32768) if valid_set is not None else None
    )

    if model_seed is not None:
        random.seed(model_seed)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)

    model = ResNet(
        input_size=train_set.X.shape[1],
        num_classes=2,
        num_groups=3,
        blocks_in_group=3,
        units=128,
        activation_fn="ReLU",
        dropout=0.3,
        batch_norm=True,
        focal=False,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(max_epochs):
        print(f"epoch: {epoch}")
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        loader_for_scoring = score_loader or valid_loader
        if score_interval and loader_for_scoring is not None and (epoch + 1) % score_interval == 0:
            interim_scores = get_scores(
                *inference_on_dataloader(model, valid_loader=loader_for_scoring, device=DEVICE)
            )
            print(f"[epoch {epoch + 1}] {score_label} metrics:")
            print_scores("", interim_scores)

    if valid_loader is not None:
        val_scores = get_scores(
            *inference_on_dataloader(model, valid_loader=valid_loader, device=DEVICE)
        )
    else:
        val_scores = {"auprc": float("nan"), "precision": float("nan"), "recall": float("nan")}
    return model, val_scores


def _save_model_with_metadata(
    model: torch.nn.Module,
    run_dir: Path,
    split_label: str,
    model_seed: int,
    data_seed: int,
    dataset_folder_size: str,
    dataset_folder_name: str,
    cfg: Dict[str, float],
    val_scores: Dict[str, float],
    test_scores: Dict[str, float],
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> Tuple[Path, Dict[str, object]]:
    split_dir = run_dir / split_label
    split_dir.mkdir(parents=True, exist_ok=False)

    model.norm_mean = norm_mean.squeeze(0).cpu()
    model.norm_std = norm_std.squeeze(0).cpu()

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path = split_dir / "model.pt"
    torch.save(model, model_path)

    fold_metadata = {
        "split": split_label,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "dataset_folder_size": dataset_folder_size,
        "dataset_folder_name": dataset_folder_name,
        "model_config": cfg,
        "validation_metrics": val_scores,
        "test_metrics": test_scores,
        "normalization_mean": model.norm_mean.tolist(),
        "normalization_std": model.norm_std.tolist(),
        "model_path": model_path.name,
    }

    with (split_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(fold_metadata, handle, indent=2)

    return model_path, fold_metadata


def _train_internal_fold(
    dataset_folder_size: str,
    dataset_folder_name: str,
    fold_path: Path,
    max_epochs: int,
    model_seed: int,
    data_seed: int,
    run_dir: Path,
    score_interval: int,
) -> Dict[str, object]:
    train_set, valid_set, test_set = load_datasets(
        dataset_folder_size,
        dataset_folder_name,
        fold_path,
        load_test=True,
    )

    internal_raw = load_internal(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        model_path=fold_path,
        regex=True,
    )
    norm_mean, norm_std = compute_normalization_stats(internal_raw.train)

    _shuffle_dataset(train_set, data_seed)
    _shuffle_dataset(valid_set, data_seed + 1)

    model, val_scores = train_and_evaluate(
        train_set=train_set,
        valid_set=valid_set,
        max_epochs=max_epochs,
        model_seed=model_seed,
        data_seed=data_seed,
        score_interval=score_interval,
    )

    test_loader = get_valid_loader(test_set, batch_size=32768)
    test_scores = get_scores(
        *inference_on_dataloader(model, test_loader=test_loader, device=DEVICE)
    )

    cfg = extract_cfg(model)

    fold_label = f"fold_{fold_path.stem}"
    model_path, fold_metadata = _save_model_with_metadata(
        model=model,
        run_dir=run_dir,
        split_label=fold_label,
        model_seed=model_seed,
        data_seed=data_seed,
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        cfg=cfg,
        val_scores=val_scores,
        test_scores=test_scores,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    return {
        "split": fold_label,
        "fold": fold_path.stem,
        "modelSeed": model_seed,
        "dataSeed": data_seed,
        "source": "single_model_internal",
        "n_models": 1,
        **{f"val_{k}": v for k, v in val_scores.items()},
        **{f"test_{k}": v for k, v in test_scores.items()},
        **cfg,
        "dataset": f"{dataset_folder_size}/{dataset_folder_name}",
        "model_path": str(model_path.relative_to(run_dir)),
    }


def _prepare_external_splits(
    dataset_folder_size: str,
    dataset_folder_name: str,
) -> Tuple[DatasetEntry, DatasetEntry, torch.Tensor, torch.Tensor]:
    external_raw = load_external(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
    )

    norm_mean, norm_std = compute_normalization_stats(external_raw.train)

    train_norm = apply_normalization(external_raw.train, norm_mean, norm_std)
    test_norm = apply_normalization(external_raw.test, norm_mean, norm_std)

    train_set = DatasetEntry(train_norm.X.clone(), train_norm.y.clone())
    test_set = DatasetEntry(test_norm.X.clone(), test_norm.y.clone())

    return train_set, test_set, norm_mean, norm_std


def _train_external_split(
    dataset_folder_size: str,
    dataset_folder_name: str,
    max_epochs: int,
    model_seed: int,
    data_seed: int,
    run_dir: Path,
    score_interval: int,
) -> Dict[str, object]:
    train_set, test_set, norm_mean, norm_std = _prepare_external_splits(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
    )

    _shuffle_dataset(train_set, data_seed)

    test_loader = get_valid_loader(test_set, batch_size=32768)

    model, val_scores = train_and_evaluate(
        train_set=train_set,
        valid_set=None,
        max_epochs=max_epochs,
        model_seed=model_seed,
        data_seed=data_seed,
        score_interval=score_interval,
        score_loader=test_loader,
        score_label="test",
    )

    test_scores = get_scores(
        *inference_on_dataloader(model, test_loader=test_loader, device=DEVICE)
    )

    cfg = extract_cfg(model)

    model_path, split_metadata = _save_model_with_metadata(
        model=model,
        run_dir=run_dir,
        split_label="external",
        model_seed=model_seed,
        data_seed=data_seed,
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        cfg=cfg,
        val_scores=val_scores,
        test_scores=test_scores,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    return {
        "split": "external",
        "fold": "external",
        "modelSeed": model_seed,
        "dataSeed": data_seed,
        "source": "single_model_external",
        "n_models": 1,
        **{f"val_{k}": v for k, v in val_scores.items()},
        **{f"test_{k}": v for k, v in test_scores.items()},
        **cfg,
        "dataset": f"{dataset_folder_size}/{dataset_folder_name}",
        "model_path": str(model_path.relative_to(run_dir)),
    }


def _run_internal(spec: DatasetSpec) -> None:
    folds = list(spec.internal_folds) if spec.internal_folds is not None else [f"{i}_internal" for i in range(5)]
    fold_paths = [Path(name) for name in folds]

    run_dir = _make_run_dir(spec.dataset_folder_size, spec.dataset_folder_name, spec.mode)

    rows: List[Dict[str, object]] = []
    args = []
    for fold_idx, fold_path in enumerate(fold_paths):
        data_seed = spec.base_seed + fold_idx
        model_seed = spec.base_seed + fold_idx
        args.append(
            (
                spec.dataset_folder_size,
                spec.dataset_folder_name,
                fold_path,
                spec.max_epochs,
                model_seed,
                data_seed,
                run_dir,
                spec.score_interval,
            )
        )

    processes = min(len(args), os.cpu_count() or 1)
    if processes > 1:
        print(f"Running {len(args)} internal folds across {processes} processes…")
        with Pool(processes=processes) as pool:
            for row in pool.starmap(_train_internal_fold, args):
                rows.append(row)
                print(f"\n=== {row['split']} (async) ===")
                print_scores("Validation metrics:", {k[4:]: row[k] for k in row if k.startswith("val_")})
                print_scores("Test metrics:", {k[5:]: row[k] for k in row if k.startswith("test_")})
    else:
        print("Running internal folds sequentially (single process)…")
        for arg in args:
            row = _train_internal_fold(*arg)
            rows.append(row)
            print(f"\n=== {row['split']} ===")
            print_scores("Validation metrics:", {k[4:]: row[k] for k in row if k.startswith("val_")})
            print_scores("Test metrics:", {k[5:]: row[k] for k in row if k.startswith("test_")})

    _finalize_run(spec, run_dir, rows, mode="internal")


def _run_external(spec: DatasetSpec) -> None:
    run_dir = _make_run_dir(spec.dataset_folder_size, spec.dataset_folder_name, spec.mode)

    row = _train_external_split(
        dataset_folder_size=spec.dataset_folder_size,
        dataset_folder_name=spec.dataset_folder_name,
        max_epochs=spec.max_epochs,
        model_seed=spec.base_seed,
        data_seed=spec.base_seed,
        run_dir=run_dir,
        score_interval=spec.score_interval,
    )

    print("\n=== external split ===")
    print_scores("Validation metrics:", {k[4:]: row[k] for k in row if k.startswith("val_")})
    print_scores("Test metrics:", {k[5:]: row[k] for k in row if k.startswith("test_")})

    _finalize_run(spec, run_dir, [row], mode="external")

def _finalize_run(spec: DatasetSpec, run_dir: Path, rows: List[Dict[str, object]], mode: str) -> None:
    df = pd.DataFrame(rows)
    df_path = run_dir / "metrics.csv"
    df.to_csv(df_path, index=False)

    run_metadata = {
        "dataset_folder_size": spec.dataset_folder_size,
        "dataset_folder_name": spec.dataset_folder_name,
        "mode": mode,
        "max_epochs": spec.max_epochs,
        "base_seed": spec.base_seed,
        "created_at": datetime.datetime.now().isoformat(),
        "metrics_file": df_path.name,
        "splits": [row["model_path"] for row in rows],
    }
    if mode == "internal":
        run_metadata["folds"] = [str(f) for f in (spec.internal_folds or [f"{i}_internal" for i in range(5)])]
    else:
        run_metadata["note"] = "external_train_used_without_validation"

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    print(f"\nSaved run artifacts → {run_dir}\n")

def main() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    for spec in DATASET_SPECS:
        print("=" * 80)
        print(
            f"Dataset {spec.dataset_folder_size}/{spec.dataset_folder_name} | Mode: {spec.mode}"
        )
        print("=" * 80)

        if spec.mode == "internal":
            _run_internal(spec)
        elif spec.mode == "external":
            _run_external(spec)
        else:
            raise ValueError(f"Unknown mode '{spec.mode}'. Use 'internal' or 'external'.")

if __name__ == "__main__":
    main()
