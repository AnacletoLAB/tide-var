#!/usr/bin/env python3
import sys
from pathlib import Path
import multiprocessing as mp
import argparse
import csv

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SCRIPT_DIR))
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
from tabulate import tabulate
from torch import Tensor

import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
from tremm.data.mendelian_dataset import DatasetEntry
torch.serialization.add_safe_globals([DatasetEntry])


from tabm import TabM

from utils import (
    load_internal,
    load_external,
    inference_on_dataloader,
    get_scores,
    get_train_loader,
    get_valid_loader,
    compute_normalization_stats,
    apply_normalization,
)

# TabM global var
share_training_batches=True

DEVICE = "cuda"

def print_scores(scores: dict, floatfmt: str = ".4f"):
    rows = [(k, float(scores[k])) for k in sorted(scores)]
    proc = mp.current_process()
    print(f"[{proc.name} | PID {proc.pid}]")
    print(tabulate(
        rows,
        headers=["Metric", "Value"],
        tablefmt="github",
        floatfmt=floatfmt
    ))


def load_datasets(
    dataset_folder_size: str,
    dataset_folder_name: str,
    model_path: Path,
    load_test: bool = True,
):
    internal = load_internal(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        model_path=model_path,
        regex=True,
    )
    mean, std = compute_normalization_stats(internal.train)
    train_set = apply_normalization(internal.train, mean, std)
    valid_set = apply_normalization(internal.valid, mean, std)

    if load_test:
        external = load_external(
            dataset_folder_size=dataset_folder_size,
            dataset_folder_name=dataset_folder_name,
        )
        test_set  = apply_normalization(external.test,  mean, std)
        return train_set, valid_set, test_set
    return train_set, valid_set


def train_and_evaluate(
    dataset_folder_size: str,
    dataset_folder_name: str,
    model_path: Path,
    max_epochs: int,
    k: int,
    save_model: bool,
    embedding: str,
    dim_embedding: int
) -> dict:
    n_num_features = 26
    n_classes=2
    batch_size = 8192

    # Output of load_datasets are DatasetEntry objects with two torch.tensor objects, a normalized X and Y
    train_set, valid_set, _ = load_datasets(dataset_folder_size, dataset_folder_name, model_path)
    # These are DataLoader objects
    train_loader = get_train_loader(train_set, balanced=True, batch_size=batch_size, neg_batch_ratio=0.8)
    valid_loader = get_valid_loader(valid_set, batch_size=batch_size)
    
    if embedding=='ple':
        # Piecewise-linear embeddings
        # https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md#practical-notes
        # The possible starting points are d_embedding=12 with activation=False and d_embedding=24 with activation=True. 
        # For Transformer-like models, the embedding size is usually larger and depends on a model.
        num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            rtdl_num_embeddings.compute_bins(
                train_set.X, # further args: compute bins with trees and not quantiles 
                y=train_set.y,
                regression=False,
                tree_kwargs={'min_samples_leaf': 1, 'min_impurity_decrease': 1e-4},
                ), # We are not using balanced like this
            d_embedding=dim_embedding,
            activation=False,
            version='B',
        )
        # Feature 0, 1, 2, 11, 15, 16, 20 get the following warning: 
        # The 11-th feature has just two bin edges, which means only one bin. Strictly speaking, using a single bin for the 
        # piecewise-linear encoding should not break anything, but it is the same as using sklearn.preprocessing.MinMaxScaler
        
    elif embedding=='periodic':
        num_embeddings = rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)
    elif embedding=='simple':
        # Simple embeddings.
        num_embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_num_features)
    else:
        sys.exit("Unacceptable embedding...")

    train_size = len(train_set.X)
    train_set.y = train_set.y.long().to(DEVICE)
    train_set.X = train_set.X.to(DEVICE)

    model = TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=[], # Good, works also with categorical features
        num_embeddings=num_embeddings, # this is a class!
        k=k,
        d_out=1 if n_classes is None else n_classes,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    # Automatic mixed precision (AMP) torch.float16 is implemented for completeness,
    # but it was not tested in the project, so torch.bfloat16 is used by default.
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else None
    )
    # Changing False to True can speed up training of large enough models on compatible hardware.
    amp_enabled = True and amp_dtype is not None
    grad_scaler = torch.amp.GradScaler('cuda') if amp_dtype is torch.float16 else None  # type: ignore

    scores=[]
    proc = mp.current_process()
    print(f"[{proc.name} | PID {proc.pid}] Starting training for {model_path.name} (k={k}, epochs={max_epochs})")
    model.train()
    for epoch in range(max_epochs):
        batches = (
            # One shared permutation of indices
            torch.randperm(train_size, device=DEVICE).split(batch_size) 
            if share_training_batches
            # k separate permutations (one column per prediction head)
            else (
                torch.rand((train_size, model.backbone.k), device=DEVICE)
                .argsort(dim=0)
                .split(batch_size, dim=0)        # batch_idx has shape (B, k)
            )
        )
               
        for batch_idx in batches:
            Xb = train_set.X[batch_idx]
            yb = train_set.y[batch_idx]
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb, model)
            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()  # type: ignore
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
        if epoch % 5 == 0 or epoch==max_epochs-1:
            score = get_scores(*inference_on_dataloader(model, valid_loader, None, DEVICE))
            print(f"Current epoch: {epoch}")
            print_scores(score)
            scores.append({"epoch": epoch, "score": score})
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if save_model:
        torch.save(model.state_dict(), 
            model_dir / f"state_dict-{dataset_folder_size}-{dataset_folder_name}-{model_path.name}.pth" )
    return scores


def loss_fn(y_pred: Tensor, y_true: Tensor, model: TabM) -> Tensor:

    # TabM produces k predictions. Each of them must be trained separately.

    # Regression:     (batch_size, k)            -> (batch_size * k,)
    # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
    y_pred = y_pred.flatten(0, 1)

    if share_training_batches:
        # (batch_size,) -> (batch_size * k,)
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        # (batch_size, k) -> (batch_size * k,)
        y_true = y_true.flatten(0, 1)

    return nn.functional.cross_entropy(y_pred, y_true)

def worker_eval(dataset_folder_size, dataset_folder_name, model_path, max_epochs, k, save_model, embedding: str, dim_embedding: int):
    return model_path.name, "eval", train_and_evaluate(dataset_folder_size, dataset_folder_name, model_path, max_epochs, k, save_model, embedding, dim_embedding)


def run_model(k: int, dataset_folder_size: str, dataset_folder_name: str, max_epochs: int, save_model: bool, embedding: str, dim_embedding: int) -> None:
    in_parallel  = 1
    model_paths  = [Path(f"{i}_internal") for i in range(5)]

    # ─── launch workers ─────────────────────────────────────────────────────────
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(model_paths) * in_parallel) as pool:
        results = pool.starmap(worker_eval, [ (dataset_folder_size, dataset_folder_name, mpth, max_epochs, k, save_model, embedding, dim_embedding) 
                                                for mpth in model_paths ]
                               )
    # ─────────────────────────────────────────────────────────────────────────────
    table = []
    lst   = []
    # ─── build the per-run rows ─────────────────────────────────────────────────
    for res in results:
        model_name, run_type, scep = res
        for scse in scep:
            sc = scse['score']
            nep = scse['epoch']
            table.append([
                model_name, run_type, nep,
                f"{sc['accuracy']:.4f}", f"{sc['f1']:.4f}",
                f"{sc['recall']:.4f}",  f"{sc['precision']:.4f}",
                f"{sc['auprc']:.4f}"
            ])
        lst.append(sc)
    table = sorted(table, key=lambda row: (row[2], row[0][0]))
    # ─────────────────────────────────────────────────────────────────────────────
    table.append([""] * 7)                         # spacer before averages

    def avg(lst, key): return sum(d[key] for d in lst) / len(lst) if lst else 0

    table.append(["Average", "eval", "Final",f"{avg(lst,'accuracy'):.4f}", f"{avg(lst,'f1'):.4f}",
                f"{avg(lst,'recall'):.4f}",  f"{avg(lst,'precision'):.4f}",
                f"{avg(lst,'auprc'):.4f}"
    ])

    # ─── print to console ───────────────────────────────────────────────────────
    print(tabulate(table,
                   headers=["Model", "Run", "N_Epoch", "Accuracy", "F1", "Recall", "Precision", "AUPRC"],
                   tablefmt="github"))

    # ─── prepare for Markdown file ────────────────────────────────────────────────────
    headers   = ["Model", "Run", "N_Epoch", "Accuracy", "F1", "Recall", "Precision", "AUPRC"]
    table_md  = tabulate(table, headers=headers, tablefmt="github")

    parent_path = Path(__file__).parent
    import datetime
    ts         = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path   = parent_path / "md_results"/ f"results_{dataset_folder_size}_{ts}_k{k}_embedding-{embedding}_demb_{dim_embedding}.md"

    # ─── write Markdown file ────────────────────────────────────────────────────
    with out_path.open("w") as f:
        f.write(f"Used {embedding} embedding d={dim_embedding}, an average of k={k} MLPs, batch_size of 8192,\n AdamW optimizer with lr=2e-3, weight_decay=3e-4")
        f.write(table_md + "\n\n")       # metrics first
    print(f"\nSaved results and configs to {out_path}\n")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Run model with parameters.")
    parser.add_argument("-k", "--num-mlps", type=int, default=1, help="Number of MLPs to use, TabM represents an ensemble of k MLPs.")
    
    # parser.add_argument("-d", "--data-folder", type=str, default= "kfold_full")
    # parser.add_argument("-n", "--data-name", type=str, default="14-02-25_16-02_rand_seed_42")

    parser.add_argument("-d", "--data-folder", type=str, default= "kfold_100k")
    parser.add_argument("-n", "--data-name", type=str, default="14-02-25_12-56_rand_seed_42")

    parser.add_argument( "-e", "--max-epochs", type=int, default=25)
    parser.add_argument( "-s", "--save-model", action='store_true', help="Save model state dict of each fold.")
    parser.add_argument( "-m", "--embedding", type=str, choices=['simple', 'ple', 'periodic'], default='ple')
    parser.add_argument( "-de", "--dim-embedding", type=int, default=12, help="Dimension of the embedding, default is 12 for simple embeddings.")
    args = parser.parse_args()

    run_model(args.num_mlps, args.data_folder, args.data_name, args.max_epochs, args.save_model, args.embedding, args.dim_embedding)
