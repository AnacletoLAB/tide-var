from pathlib import Path
import re
import torch
from typing import Tuple, Optional
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import List
from scipy.optimize import minimize

from tremm.data.mendelian_dataset import InternalFold, ExternalFold, NExternalFold
from tremm.experiments.utils.loaders import get_valid_loader
from tremm.data.mendelian_dataset import DatasetEntry

import os
# Don't set CUDA_VISIBLE_DEVICES here - let the calling script control it

DEVICE = "cuda"

def get_scores(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
):
    # scores = {
    #     "accuracy": lambda: accuracy_score(targets, preds),
    #     "f1": lambda: f1_score(targets, preds, pos_label=1),
    #     "recall": lambda: recall_score(targets, preds, pos_label=1),
    #     "precision": lambda: precision_score(targets, preds, pos_label=1),
    #     "auprc": lambda: average_precision_score(targets, probs),
    # }
    scores = {
        "auprc": lambda: average_precision_score(targets, probs),
        "precision": lambda: precision_score(targets, preds, pos_label=1),
        "recall": lambda: recall_score(targets, preds, pos_label=1),
    }
    results = {}
    for metric in scores.keys():
        results[metric] = scores[metric]()
    
    return results

def compute_normalization_stats(train: DatasetEntry):
    device = train.X.device
    mean = train.X.mean(0, keepdim=True).to(device)
    std  = train.X.std(0, unbiased=False, keepdim=True).to(device)
    return mean, std

def apply_normalization(split: DatasetEntry, mean: torch.Tensor, std: torch.Tensor):
    split.X = (split.X - mean.to(split.X.device)) / std.to(split.X.device)
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
    else:
        # When regex=False, use the model_path.name directly (should be like "0_internal.pt")
        split_name = model_path.stem  # removes .pt extension, gives "0_internal"
    
    assert split_name is not None, "split_name is None"
    for dataset in datasets_path.iterdir():
        if split_name in dataset.name:
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
            external = torch.load(dataset, map_location="cpu")
            
    assert external is not None, "external is None"
    return external

def load_n_external(
    dataset_folder_size: str,
    dataset_folder_name: str,
) -> NExternalFold:    
    """Load NExternalFold dataset for n_kfold datasets with global validation set."""
    datasets_path = (
        Path(__file__).resolve().parent
        / "tremm"
        / "data"
        / "tensors"
        / dataset_folder_size
        / dataset_folder_name
    )
    n_external = None 
    
    for dataset in datasets_path.iterdir():
        if "external" in dataset.name:
            n_external = torch.load(dataset)
            
    assert n_external is not None, "n_external is None"
    assert isinstance(n_external, NExternalFold), f"Expected NExternalFold, got {type(n_external)}"
    return n_external

def load_datasets(
    dataset_folder_size: str,
    dataset_folder_name: str,
    model_path: Path,
    load_test: bool = False,
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

    if load_test:
        external = load_external(
            dataset_folder_size=dataset_folder_size,
            dataset_folder_name=dataset_folder_name,
        )
        test_set  = apply_normalization(external.test,  mean, std)
        return train_set, valid_set, test_set
    if load_test_only:
        external = load_external(
            dataset_folder_size=dataset_folder_size,
            dataset_folder_name=dataset_folder_name,
        )
        test_set  = apply_normalization(external.test,  mean, std)
        return test_set

    return train_set, valid_set

def shuffle_split(
    data_split,
    seed: int
):
    torch.manual_seed(seed)
    indices = torch.randperm(data_split.X.size(0))
    
    data_split.X = data_split.X[indices]
    
    if hasattr(data_split, 'y'):
        data_split.y = data_split.y[indices]
    return data_split  

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
    all_probs   = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs  = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs    = torch.softmax(outputs, dim=1)[:, 1]

            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    targets = torch.cat(all_targets).numpy()
    preds   = torch.cat(all_preds).numpy()
    probs   = torch.cat(all_probs).numpy()

    return targets, preds, probs
    
def inference_on_dataloader_ft(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For FTTransformer (d_out=1 + BCEWithLogitsLoss), returns
      targets: shape (N,) integers {0,1}
      preds:   shape (N,) integers {0,1}
      probs:   shape (N,) floats sigmoid(logit)
    """
    model.eval()
    all_targets = []
    all_preds   = []
    all_probs   = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X = X_batch.to(device)
            # y_batch is float or long, but we want ints for targets
            y = y_batch.to(device).long().squeeze()
            logits = model(X, None).squeeze(1)  
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).long()
            all_targets.append(y.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    targets = torch.cat(all_targets).numpy()
    preds   = torch.cat(all_preds).numpy()
    probs   = torch.cat(all_probs).numpy()
    return targets, preds, probs

def build_ensemble_rows(
    results: list,
    all_model_paths: list[Path],
    dataset_folder_size: str,
    dataset_folder_name: str,
    use_weighted: bool = False,
    use_logit_avg: bool = False,
    use_stacking: bool = False,
) -> list[dict]:
    """
    Build per‐fold and overall ensemble rows.

    - If use_weighted=True: optimize soft‐voting weights on the fold’s val set.
    - Elif use_logit_avg=True: average raw logits then sigmoid.
    - Else: classical equal‐weight soft‐voting.
    """
    # 1) Group models & AUPRCs by fold
    auprc_by_fold: dict[str, list[float]] = {}
    models_by_fold: dict[str, list[nn.Module]] = {}
    for i, (model, scores, _) in enumerate(results):
        fold = all_model_paths[i].name
        auprc_by_fold.setdefault(fold, []).append(scores["auprc"])
        models_by_fold.setdefault(fold, []).append(model)

    ens_rows: list[dict] = []
    fold_min_max: list[tuple[float, float]] = []

    # 2) Per‐fold processing
    for fold_name, models in models_by_fold.items():
        # record individual‐model min/max AUPRC
        min_a = min(auprc_by_fold[fold_name])
        max_a = max(auprc_by_fold[fold_name])
        fold_min_max.append((min_a, max_a))

        # move all models to DEVICE & eval one time
        models = [m.to(DEVICE).eval() for m in models]

        # load validation split once
        _, valid_set = load_datasets(dataset_folder_size,     # type: ignore
                                     dataset_folder_name,
                                     Path(fold_name))
        valid_loader = get_valid_loader(valid_set, batch_size=32768) # type: ignore

        # extract labels for this fold
        labels = np.concatenate([yb.cpu().numpy() for _, yb in valid_loader], axis=0)

        if use_stacking:
            prob_mat = get_valid_prob_matrix(models, valid_loader).T  # (N, M)
            
            if use_weighted:
                weights = optimize_weights(prob_mat.T, labels)         # length M
                p_weighted = (prob_mat * weights[None, :]).sum(axis=1)
                X_stack = np.concatenate([prob_mat, p_weighted[:, None]], axis=1)
            else:
                X_stack = prob_mat                                  # (N, M)

            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler().fit(X_stack)
            X_scaled = scaler.transform(X_stack)
        
            from sklearn.linear_model import LogisticRegression
            meta = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200)
            meta.fit(X_scaled, labels)
        
            p_ens   = meta.predict_proba(X_scaled)[:, 1]
            preds   = (p_ens >= 0.5).astype(int)
            ens_scores = get_scores(labels, preds, p_ens)

        elif use_weighted and use_logit_avg:
            # build (M,N) of z1−z0
            logit_mat = get_valid_logit_matrix(models, valid_loader, DEVICE)
            weights   = optimize_logit_weights(logit_mat, labels)
            from scipy.special import expit
            p_ens     = expit(weights @ logit_mat)
            preds     = (p_ens >= 0.5).astype(int)
            ens_scores = get_scores(labels, preds, p_ens)

        # 2️⃣ weighted‐only branch
        elif use_weighted:
            print(f"    *** UTILSZ.PY: EXECUTING WEIGHTED BRANCH for fold {fold_name} ***")
            prob_mat  = get_valid_prob_matrix(models, valid_loader)
            weights   = optimize_weights(prob_mat, labels)
            print(f"    *** OPTIMIZED WEIGHTS: {weights} ***")
            p_ens     = weights @ prob_mat
            preds     = (p_ens >= 0.5).astype(int)
            ens_scores = get_scores(labels, preds, p_ens)

        # 3️⃣ logit‐avg‐only branch
        elif use_logit_avg:
            logit_mat   = get_valid_logit_matrix(models, valid_loader, DEVICE)
            mean_logits = logit_mat.mean(axis=0)
            from scipy.special import expit
            p_ens       = expit(mean_logits)
            preds       = (p_ens >= 0.5).astype(int)
            ens_scores  = get_scores(labels, preds, p_ens)
        else:
            # 2c) classical equal‐weight soft‐voting
            print(f"    *** UTILSZ.PY: EXECUTING SIMPLE AVERAGING BRANCH for fold {fold_name} ***")
            ens_scores = ensemble_valid_scores_from_models(
                models, dataset_folder_size, dataset_folder_name,
                model_path=Path(fold_name)
            )

        # 3) append the fold’s ensemble row
        ens_rows.append({
            "fold": fold_name,
            "modelSeed": "ensemble",
            "dataSeed": "",
            **ens_scores,
            "input_size": "", "num_classes": "", "num_groups": "",
            "blocks_in_group": "", "units": "", "activation_fn": "",
            "dropout": "", "batch_norm": "", "focal": "",
            "min_auprc": f"{min_a:.6f}",
            "max_auprc": f"{max_a:.6f}",
            "dataset": f"{dataset_folder_size}/{dataset_folder_name}",
        })

    # 4) overall summary across folds
    avg_scores = {k: sum(r[k] for r in ens_rows) / len(ens_rows)
                  for k in ("auprc", "precision", "recall")}
    global_min = min(m for m, _ in fold_min_max)
    global_max = max(M for _, M in fold_min_max)
    ens_rows.append({
        "fold": "AverageEnsemble",
        "modelSeed": "",
        "dataSeed": "",
        **avg_scores,
        "input_size": "", "num_classes": "", "num_groups": "",
        "blocks_in_group": "", "units": "", "activation_fn": "",
        "dropout": "", "batch_norm": "", "focal": "",
        "min_auprc": f"{global_min:.6f}",
        "max_auprc": f"{global_max:.6f}",
        "dataset": f"{dataset_folder_size}/{dataset_folder_name}",
    })

    return ens_rows

def _flatten_records(
    results,
    model_seeds,
    data_seeds,
    all_model_paths,
):
    rows = []
    for i, (_, scores, cfg) in enumerate(results):
        row = {
            "fold"       : all_model_paths[i].name,
            "modelSeed"  : model_seeds[i],
            "dataSeed"   : data_seeds[i],
            **scores,                       
            **cfg,                          
            "min_auprc"  : "",              
            "max_auprc"  : "",
            "dataset"    : "",              
        }
        rows.append(row)
    return rows

def extract_cfg(model: nn.Module) -> dict:
    input_size      = getattr(model, "input_size",      None)
    if input_size is None and hasattr(model, "input_layer"):
        input_size = model.input_layer.in_features

    units           = getattr(model, "units",           None)
    if units is None and hasattr(model, "input_layer"):
        units = model.input_layer.out_features

    num_classes     = getattr(model, "num_classes",     None)
    if num_classes is None and hasattr(model, "classifier"):
        num_classes = model.classifier.out_features

    num_groups       = getattr(model, "num_groups",      len(model.groups))
    blocks_in_group  = getattr(model, "blocks_in_group", None)
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
                    "ReLU", "LeakyReLU", "ELU", "SiLU", "GELU", "PReLU"
                ):
                    act_name = layer.__class__.__name__

    return {
        "input_size"      : input_size,
        "num_classes"     : num_classes,
        "num_groups"      : num_groups,
        "blocks_in_group" : blocks_in_group,
        "units"           : units,
        "activation_fn"   : act_name,
        "dropout"         : dropout_p,
        "batch_norm"      : batch_norm,
        "focal"           : focal,
    }

def ensemble_valid_scores_from_models(
    models: List[nn.Module],
    dataset_folder_size: str,
    dataset_folder_name: str,
    model_path: Path,        
    device: str = "cuda",
) -> dict:
    """Soft-vote the list of models on *their own* validation split."""
    _, valid_set = load_datasets( # type: ignore
        dataset_folder_size, dataset_folder_name, model_path
    ) 
    loader = get_valid_loader(valid_set, batch_size=32768) # type: ignore

    models = [m.to(device).eval() for m in models]

    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            prob_stack = torch.stack(
                [torch.softmax(m(xb), dim=1)[:, 1] for m in models], dim=0
            )
            probs_mean = prob_stack.mean(dim=0)
            all_probs.append(probs_mean.cpu())
            all_labels.append(yb.cpu())

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = (probs >= 0.5).astype(int)
    return get_scores(labels, preds, probs)

def get_valid_prob_matrix(models, valid_loader):
    all_probs = []
    for xb, _ in valid_loader:
        xb = xb.to(DEVICE)
        # shape: (M, batch_size)
        batch_probs = torch.stack([torch.softmax(m(xb), dim=1)[:,1] 
                                   for m in models], dim=0)
        all_probs.append(batch_probs.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=1)  # (M, N_samples)

def get_valid_logit_matrix(models, valid_loader, device="cuda"):
    models = [m.to(device).eval() for m in models]
    all_logs = []
    with torch.no_grad():
        for xb, _ in valid_loader:
            xb = xb.to(device)
            batch_logits = torch.stack([m(xb) for m in models], dim=0)
            pos_logit = batch_logits[:, :, 1] - batch_logits[:, :, 0]
            all_logs.append(pos_logit.cpu().numpy())
    return np.concatenate(all_logs, axis=1)  # shape (M, N_valid)

def optimize_weights(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Given `probs` of shape (N_models, N_samples) and binary `labels` of shape (N_samples,),
    find non-negative weights w (sum to 1) that minimize log-loss on the validation set.
    Returns the optimal weight vector of length N_models.
    """
    M, N = probs.shape

    eps = 1e-6
    P = np.clip(probs, eps, 1-eps)

    def log_loss(w):
        w = np.clip(w, eps, 1)
        w = w / w.sum()
        p_ens = (w[:, None] * P).sum(axis=0)
        return - np.mean(labels * np.log(p_ens) + (1-labels) * np.log1p(-p_ens))

    # start from uniform
    w0 = np.ones(M) / M
    bounds = [(0,1)] * M
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
    
    print(f"      OPTIMIZE_WEIGHTS: Starting uniform weights: {w0}")
    print(f"      OPTIMIZE_WEIGHTS: Initial loss: {log_loss(w0):.6f}")
    
    res = minimize(log_loss, w0, bounds=bounds, constraints=cons)
    w_opt = np.clip(res.x, eps, 1)
    w_final = w_opt / w_opt.sum()
    
    print(f"      OPTIMIZE_WEIGHTS: Final weights: {w_final}")
    print(f"      OPTIMIZE_WEIGHTS: Final loss: {log_loss(w_final):.6f}")
    print(f"      OPTIMIZE_WEIGHTS: Improvement: {log_loss(w0) - log_loss(w_final):.6f}")
    
    return w_final

def optimize_logit_weights(logit_mat: np.ndarray, labels: np.ndarray) -> np.ndarray:
    M, N = logit_mat.shape
    eps = 1e-6

    from scipy.special import expit
    def loss(w):
        # enforce simplex
        w = np.clip(w, eps, 1)
        w = w / w.sum()
        # weighted logit-average + sigmoid
        z = w @ logit_mat          # shape (N,)
        p = expit(z)               # in (0,1)
        # binary cross-entropy
        return -np.mean(labels * np.log(p + eps)
                        + (1-labels) * np.log(1 - p + eps))

    # setup: uniform start, box bounds, sum-to-1 constraint
    w0     = np.ones(M) / M
    bounds = [(0,1)] * M
    cons   = {'type':'eq', 'fun': lambda w: w.sum() - 1}

    res    = minimize(loss, w0, bounds=bounds, constraints=cons)
    w_opt  = np.clip(res.x, eps, 1)
    return w_opt / w_opt.sum()

def evaluate_weighted_global_ensemble_on_external_test(
    all_models: List[nn.Module],
    dataset_folder_size: str,
    dataset_folder_name: str,
) -> dict:
    """
    Evaluate weighted global ensemble using n_kfold dataset.
    
    Uses global validation set to optimize weights for all models,
    then applies those weights to the external test set.
    
    Args:
        all_models: List of all trained models (15 models for 5 folds × 3 models each)
        dataset_folder_size: Size folder name (e.g., "100k")
        dataset_folder_name: Dataset name with timestamp and seed
        
    Returns:
        Dictionary with ensemble scores on external test set
    """
    print(f"\n*** EVALUATING WEIGHTED GLOBAL ENSEMBLE ON EXTERNAL TEST ***")
    print(f"Dataset: {dataset_folder_size}/{dataset_folder_name}")
    print(f"Number of models: {len(all_models)}")
    
    # Load n_kfold dataset
    try:
        n_external_fold = load_n_external(dataset_folder_size, dataset_folder_name)
        print(f"✓ Loaded NExternalFold dataset with global validation set")
    except Exception as e:
        print(f"✗ Failed to load n_kfold dataset: {e}")
        print(f"Falling back to simple averaging on external test")
        return evaluate_simple_global_ensemble_on_external_test(all_models, dataset_folder_size, dataset_folder_name)
    
    # Normalize datasets using the first internal fold's training statistics
    # (This matches the normalization used during training)
    internal_fold = load_internal(dataset_folder_size, dataset_folder_name, Path("0_internal.pt"), regex=False)
    mean, std = compute_normalization_stats(internal_fold.train)
    
    global_valid_normalized = apply_normalization(n_external_fold.global_valid, mean, std)
    external_test_normalized = apply_normalization(n_external_fold.test, mean, std)
    
    # Create data loaders
    global_valid_loader = get_valid_loader(global_valid_normalized, batch_size=32768)
    external_test_loader = get_valid_loader(external_test_normalized, batch_size=32768)
    
    # Move all models to device and set to eval mode
    all_models = [m.to(DEVICE).eval() for m in all_models]
    
    # Extract labels from global validation set
    global_valid_labels = np.concatenate([yb.cpu().numpy() for _, yb in global_valid_loader], axis=0)
    
    print(f"Global validation set size: {len(global_valid_labels)}")
    print(f"Positive class ratio in global validation: {global_valid_labels.mean():.4f}")
    
    # Get probability matrix for global validation set
    print("Computing probability matrix on global validation set...")
    global_valid_prob_mat = get_valid_prob_matrix(all_models, global_valid_loader)  # Shape: (n_models, n_samples)
    print(f"Probability matrix shape: {global_valid_prob_mat.shape}")
    
    # Optimize weights using global validation set
    print("Optimizing global ensemble weights...")
    global_weights = optimize_weights(global_valid_prob_mat, global_valid_labels)
    print(f"✓ Optimized global weights: {global_weights}")
    
    # Apply weighted ensemble to external test set
    print("Applying weighted ensemble to external test set...")
    external_test_prob_mat = get_valid_prob_matrix(all_models, external_test_loader)  # Shape: (n_models, n_samples)
    
    # Weighted ensemble prediction
    p_ens = global_weights @ external_test_prob_mat  # Shape: (n_samples,)
    preds = (p_ens >= 0.5).astype(int)
    
    # Extract true labels from external test set
    external_test_labels = np.concatenate([yb.cpu().numpy() for _, yb in external_test_loader], axis=0)
    
    # Compute scores
    ensemble_scores = get_scores(external_test_labels, preds, p_ens)
    
    print(f"✓ Weighted Global Ensemble Results:")
    print(f"    AUPRC: {ensemble_scores['auprc']:.6f}")
    print(f"    Precision: {ensemble_scores['precision']:.6f}")
    print(f"    Recall: {ensemble_scores['recall']:.6f}")
    
    return ensemble_scores

def evaluate_simple_global_ensemble_on_external_test(
    all_models: List[nn.Module],
    dataset_folder_size: str,
    dataset_folder_name: str,
) -> dict:
    """
    Fallback function for simple averaging global ensemble on external test set.
    Used when n_kfold dataset is not available.
    """
    print(f"\n*** EVALUATING SIMPLE GLOBAL ENSEMBLE ON EXTERNAL TEST ***")
    print(f"Dataset: {dataset_folder_size}/{dataset_folder_name}")
    print(f"Number of models: {len(all_models)}")
    
    # Load regular external fold
    external_fold = load_external(dataset_folder_size, dataset_folder_name)
    
    # Normalize using first internal fold's training statistics
    internal_fold = load_internal(dataset_folder_size, dataset_folder_name, Path("0_internal.pt"), regex=False)
    mean, std = compute_normalization_stats(internal_fold.train)
    external_test_normalized = apply_normalization(external_fold.test, mean, std)
    
    # Create data loader
    external_test_loader = get_valid_loader(external_test_normalized, batch_size=32768)
    
    # Move all models to device and set to eval mode
    all_models = [m.to(DEVICE).eval() for m in all_models]
    
    # Simple averaging ensemble
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in external_test_loader:
            xb = xb.to(DEVICE)
            prob_stack = torch.stack(
                [torch.softmax(m(xb), dim=1)[:, 1] for m in all_models], dim=0
            )
            probs_mean = prob_stack.mean(dim=0)
            all_probs.append(probs_mean.cpu())
            all_labels.append(yb.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)
    
    ensemble_scores = get_scores(labels, preds, probs)
    
    print(f"✓ Simple Global Ensemble Results:")
    print(f"    AUPRC: {ensemble_scores['auprc']:.6f}")
    print(f"    Precision: {ensemble_scores['precision']:.6f}")
    print(f"    Recall: {ensemble_scores['recall']:.6f}")
    
    return ensemble_scores
