import sys
from pathlib import Path

project_root = (Path(__file__).resolve().parents[2])
sys.path.insert(0, str(project_root))

import argparse
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch

from tremm.data.mendelian_dataset import MendelianDataset
from tremm.data.mendelian_dataset import DatasetSplit, ExternalFold, NExternalFold, InternalFold

TENSOR_DATASETS_DIR = Path(__file__).parents[1] / "data" / "tensors"
DEFAULT_INPUT_DATA_PATH = Path(__file__).parents[1] / "data" / "raw_data"
DEFAULT_ARROW_DATA_PATH = Path(__file__).parents[1] / "data" / "arrow_datasets"

HASH_N = 10_000
HASH_SEED = 42

def list_datasets(dataset_dir: Path = TENSOR_DATASETS_DIR):
    """
    Prints a list of available dataset splits with their metadata.
    Returns a list of dataset folder names.
    """
    datasets = []

    if dataset_dir.exists() and dataset_dir.is_dir():
        for fold_path in dataset_dir.iterdir():
            if fold_path.is_dir():
                for dataset_path in fold_path.iterdir():
                    if dataset_path.is_dir():
                        metadata_path = dataset_path / "metadata.json"
                        if metadata_path.exists():
                            with open(metadata_path, "r") as meta_file:
                                metadata = json.load(meta_file)
                                if "kfold" in fold_path.name:
                                    extra_info = f"Internal splits: {metadata.get('internal_splits', [])}, " \
                                    f"Test set ratio: {metadata.get('ext_test_ratio')}, Validation set ratio: {metadata.get('int_valid_ratio')}, " \
                                    f"N. internal folds: {metadata.get('num_internal_folds')}"
                                else:
                                    extra_info = f"Shuffle: {metadata.get('shuffle')}, Random sampling: {metadata.get('random_sampling')}"
                                    
                                summary = f"Fold: {fold_path.name}, " \
                                f"Split name: {metadata.get('split_name')}, " \
                                f"Size: {metadata.get('dataset_size')}, " \
                                f"Train: {metadata.get('train_size')}, " \
                                f"Valid: {metadata.get('valid_size')}, " \
                                f"Test: {metadata.get('test_size')}, " \
                                f"Split Ratios: {metadata.get('split_ratios')}, Positive Ratios: {metadata.get('positive_ratios')}, " \
                                f"Created at: {metadata.get('created_at')}, Random seed: {metadata.get('random_seed')}\n" \
                                f"Hash: {metadata.get('hash')}, Hash number: {metadata.get('hash_n')}, Hash seed: {metadata.get('hash_seed')}, \n" \
                                f"{extra_info}"
                                datasets.append((fold_path.name, summary))
                        else:
                            datasets.append((fold_path.name, f"Dataset: {fold_path.name}\n  No metadata available.\n"))

    if datasets:
        print("\nAvailable datasets:")
        for i, (_, summary) in enumerate(datasets, 1):
            print(f"{i}. {summary}")
            print()
    else:
        print("\nNo datasets found in", dataset_dir)

    return [d[0] for d in datasets]

def load_metadata(dataset_path: Path) -> Dict:
    """Loads dataset metadata from `dataset_path/metadata.json` if it exists."""
    metadata_path = dataset_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as file:
            return json.load(file)
    return {}

def normalize_ratios(metadata: dict) -> Dict:
    """Converts lists in metadata to tuples for consistent handling."""
    meta_copy = metadata.copy()
    if "split_ratios" in meta_copy:
        meta_copy["split_ratios"] = tuple(meta_copy["split_ratios"])
    if "positive_ratios" in meta_copy:
        meta_copy["positive_ratios"] = tuple(meta_copy["positive_ratios"])
    return meta_copy

def find_matching_dataset(
    metadata: Dict,
    folders_dir: Path = TENSOR_DATASETS_DIR
) -> Path | None:
    for parent_folder in folders_dir.iterdir():
        if parent_folder.is_dir():
            for dataset_path in parent_folder.iterdir():
                if dataset_path.is_dir():
                    existing_metadata = load_metadata(dataset_path)
                    if not existing_metadata:
                        print(f"Skipping {dataset_path}: No metadata found.")
                        continue

                    if existing_metadata.get("hash") == metadata["hash"]:
                        print(f"Matching dataset found: {dataset_path}")
                        print(f"Hashes match: {metadata['hash']}")
                        return dataset_path
                    else:
                        print(f"No match in {dataset_path}: Different hash.\n")
    print()
    return None

def find_matching_kfold(
    metadata: Dict,
    folders_dir: Path = TENSOR_DATASETS_DIR,
) -> Path | None:
    if not folders_dir.exists():
        return None
    for parent_folder in folders_dir.iterdir():
        if parent_folder.is_dir() and "kfold" in parent_folder.name:
            for kfold_path in parent_folder.iterdir():
                if kfold_path.is_dir():
                    existing_metadata = load_metadata(kfold_path)
                    if not existing_metadata:
                        print(f"Skipping {kfold_path}: No metadata found.")
                        continue
                    if existing_metadata.get("hash") == metadata["hash"]:                        
                        print(f"Matching dataset found: {kfold_path}")
                        print(f"Hashes match: {metadata['hash']}")
                        return kfold_path
                    else:
                        print(f"No match in {kfold_path}: Different hash.\n")
    print()
    return None


def format_number(num: int) -> str:
    """Formats large integers into k/m strings (e.g. 1000 -> '1k', 1_000_000 -> '1m')."""
    if abs(num) >= 1_000_000:
        return f"{int(num / 1_000_000)}m"
    elif abs(num) >= 1_000:
        return f"{int(num / 1_000)}k"
    else:
        return str(num)

def hash_dataset(
    dataset: DatasetSplit,
    folders_dir: Path = TENSOR_DATASETS_DIR
)-> str:
    import torch
    hasher = hashlib.md5()
    rng = torch.Generator().manual_seed(HASH_SEED)

    for split in dataset:
        for arr in [split.X, split.y]:
            size = arr.shape[0]
            if size > HASH_N:
                indices = torch.randperm(size, generator=rng)[:HASH_N]
                truncated = arr[indices]
            else:
                truncated = arr

            truncated_np = truncated.detach().cpu().numpy()
            hasher.update(str(truncated_np.shape).encode("utf-8"))
            hasher.update(str(truncated_np.dtype).encode("utf-8"))
            hasher.update(truncated_np.tobytes())

    return hasher.hexdigest()    

def create_dataset(args) -> Path:
    from tremm.data.mendelian_dataset import MendelianDataset

    if args.dataset_size == "full":
        dataset_size = args.dataset_size
    else:
        try:
            dataset_size = int(args.dataset_size)
        except ValueError:
            raise ValueError(f"--dataset_size must be an integer or 'full', but got '{args.dataset_size}'.")

        if dataset_size >= 1_000_000:
            if dataset_size % 1_000_000 != 0:
                raise ValueError(
                    f"--dataset_size must be an exact multiple of 1,000,000 for sizes in the millions,\
                    but got {dataset_size}.\nExample: ...create_dataset --dataset_size 2000000 ..."
                )
        elif dataset_size >= 100_000:
            if dataset_size % 100_000 != 0:
                raise ValueError(
                    f"--dataset_size must be an exact multiple of 100,000 for sizes in the hundreds of thousands,\
                    but got {dataset_size}.\nExample: ...create_dataset --dataset_size 400000 ..."
                )
        else:
            raise ValueError(
                f"--dataset_size must be at least 100,000. Got {dataset_size}."
            )

    split_ratios = args.split_ratios or (0.7, 0.15, 0.15)
    if not abs(sum(split_ratios) - 1.0) < 1e-8:
        raise ValueError(
            f"--split_ratios must sum to 1.0, but got {split_ratios} with sum={sum(split_ratios):.4f}."
        )

    positive_ratios = args.positive_ratios or (0.5, 0.25, 0.25)
    if not abs(sum(positive_ratios) - 1.0) < 1e-8:
        raise ValueError(
            f"--positive_ratios must sum to 1.0, but got {positive_ratios} with sum={sum(positive_ratios):.4f}."
        )

    random_seed = args.random_seed or 42
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert dataset_size is not None, "dataset_size is None"
    
    files = ["data", "labels", "data.names"]    
    data = MendelianDataset(
        input_data_path=DEFAULT_INPUT_DATA_PATH,
        arrow_data_path=DEFAULT_ARROW_DATA_PATH,
        files=files,
        sample_n_random_rows = dataset_size if dataset_size != "full" else None,
        seed=random_seed,
        device=device,
    )

    dataset = data.train_valid_test_split(
        split_ratios=split_ratios,
        positive_ratios=positive_ratios,
        shuffle=True,
        random_seed=random_seed,
    )

    dataset_hash = hash_dataset(dataset=dataset)
    assert dataset_hash is not None, "Dataset hash is None."    
    assert isinstance(dataset_hash, str), "Dataset hash is not a string."
    assert len(dataset_hash) == 32, f"Dataset hash length is {len(dataset_hash)}; expected 32 characters."
    assert re.fullmatch(r'[0-9a-f]{32}', dataset_hash), f"Dataset hash '{dataset_hash}' is not a valid MD5 hash."

    assert dataset.valid is not None, "dataset.valid is None."
    
    dataset_size = len(dataset.train.y) + len(dataset.valid.y) + len(dataset.test.y)
    created_at = datetime.now().strftime("%d-%m-%y_%H-%M")
    dataset_name = f"{created_at}_rand_seed_{random_seed}"
   
    metadata = {
        "dataset_size": len(dataset.train.y) + len(dataset.valid.y) + len(dataset.test.y),
        "split_name": str(TENSOR_DATASETS_DIR / format_number(dataset_size) / dataset_name),  # Convert Path to str
        "train_size": len(dataset.train.y),
        "valid_size": len(dataset.valid.y),
        "test_size": len(dataset.test.y),
        "split_ratios": split_ratios,
        "positive_ratios": positive_ratios,
        "random_seed": random_seed,
        "shuffle": True,
        "random_sampling": True,
        "hash": dataset_hash,
        "hash_n": HASH_N,
        "hash_seed": HASH_SEED,
    }

    existing_dataset_path = find_matching_dataset(metadata)
    if existing_dataset_path:
        print(f"Using existing dataset: {existing_dataset_path}")
        return existing_dataset_path

    metadata["created_at"] = created_at
    new_dataset_folder = format_number(metadata["dataset_size"]) 
    (TENSOR_DATASETS_DIR / new_dataset_folder).mkdir(parents=True, exist_ok=True)

    dataset_name = f"{metadata['created_at']}_rand_seed_{metadata['random_seed']}"
    new_dataset_path = TENSOR_DATASETS_DIR / new_dataset_folder / dataset_name
    new_dataset_path.mkdir(parents=True, exist_ok=True)

    metadata_path = new_dataset_path / "metadata.json"
    with open(metadata_path, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    torch.save(dataset, new_dataset_path / "dataset.pt")

    print(f"\nNew dataset created at {new_dataset_path}")
    print(f"PATH: {new_dataset_path}/dataset.pt")
    print("\nDataset Metadata:")
    print("-" * 30)
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            formatted_value = ", ".join(map(str, value))
            print(f"{key.capitalize().replace('_', ' ')}: [{formatted_value}]")
        else:
            print(f"{key.capitalize().replace('_', ' ')}: {value}")
    print("-" * 30)

    return new_dataset_path
    
def update_hasher_with_array(hasher, arr, rng):
    size = arr.shape[0]
    if size > HASH_N:
        indices = torch.randperm(size, generator=rng)[:HASH_N]
        arr = arr[indices]
    truncated_np = arr.detach().cpu().numpy()
    hasher.update(str(truncated_np.shape).encode("utf-8"))
    hasher.update(str(truncated_np.dtype).encode("utf-8"))
    hasher.update(truncated_np.tobytes())

def update_hasher_with_fold(hasher, fold, rng):
    for split in fold:
        for arr in [split.X, split.y]:
            update_hasher_with_array(hasher, arr, rng)

def hash_kfold(
    external_fold: ExternalFold,
    internal_folds: List[InternalFold],
) -> str:
    hasher = hashlib.md5()
    rng = torch.Generator().manual_seed(HASH_SEED)

    update_hasher_with_fold(hasher, external_fold, rng)
    for _ in internal_folds:
        update_hasher_with_fold(hasher, external_fold, rng)

    return hasher.hexdigest()

def hash_n_kfold(
    n_external_fold: NExternalFold,
    internal_folds: List[InternalFold],
) -> str:
    hasher = hashlib.md5()
    rng = torch.Generator().manual_seed(HASH_SEED)

    update_hasher_with_fold(hasher, n_external_fold, rng)
    for internal_fold in internal_folds:
        update_hasher_with_fold(hasher, internal_fold, rng)

    return hasher.hexdigest()


def create_kfold(args) -> Path:
    if args.dataset_size == "full":
        dataset_size = args.dataset_size
    else:
        try:
            dataset_size = int(args.dataset_size)
        except ValueError:
            raise ValueError(f"--dataset_size must be an integer, but got '{args.dataset_size}'.")
    
        if dataset_size >= 1_000_000:
            if dataset_size % 1_000_000 != 0:
                raise ValueError(
                    f"--dataset_size must be an exact multiple of 1,000,000 for sizes in the millions, "
                    f"but got {dataset_size}.\nExample: ...create_nested_kfold --dataset_size 2000000 ..."
                )
        elif dataset_size >= 100_000:
            if dataset_size % 100_000 != 0:
                raise ValueError(
                    f"--dataset_size must be an exact multiple of 100,000 for sizes in the hundreds of thousands, "
                    f"but got {dataset_size}.\nExample: ...create_nested_kfold --dataset_size 400000 ..."
                )
        else:
            raise ValueError(f"--dataset_size must be at least 100,000. Got {dataset_size}.")

    ext_test_ratio = args.ext_test_ratio or 0.2
    if ext_test_ratio > 0.3:
        raise ValueError(f"--ext_test_ratio value {ext_test_ratio} is higher than 0.3.")

    int_valid_ratio = args.int_valid_ratio or 0.25
    if int_valid_ratio > 0.3:
        raise ValueError(f"--int_valid_ratio value {int_valid_ratio} is higher than 0.3.")

    n_ext = int(round(1 / ext_test_ratio))
    if abs((1 / ext_test_ratio) - n_ext) > 1e-6:
        raise ValueError("ext_test_ratio must be the reciprocal of an integer (e.g. 0.2, 0.25, etc.).")
    n_int = int(round(1 / int_valid_ratio))
    if abs((1 / int_valid_ratio) - n_int) > 1e-6:
        raise ValueError("int_valid_ratio must be the reciprocal of an integer (e.g. 0.25, 0.2, etc.).")

    random_seed = args.random_seed or 42

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert dataset_size is not None, "dataset_size is None"
    
    files = ["data", "labels", "data.names"]
    dataset = MendelianDataset(
        input_data_path=DEFAULT_INPUT_DATA_PATH,
        arrow_data_path=DEFAULT_ARROW_DATA_PATH,
        files=files,
        sample_n_random_rows = dataset_size if dataset_size != "full" else None,
        seed=random_seed,
        device=device,
    )
        
    external_fold, internal_folds = dataset.kfold_split(
        ext_test_ratio=ext_test_ratio,
        int_valid_ratio=int_valid_ratio,
        shuffle=True,
        random_seed=random_seed,
    )
    
    kfold_hash = hash_kfold(external_fold=external_fold, internal_folds=internal_folds)
    created_at = datetime.now().strftime("%d-%m-%y_%H-%M")
    dataset_name = f"{created_at}_rand_seed_{random_seed}"
    formatted_size = format_number(dataset_size) if isinstance(dataset_size, int) else dataset_size  
    assert dataset_size is not None, "dataset_size is None"
    
    metadata = {
        "created_at": created_at,
        "dataset_size": dataset_size,
        "split_name": dataset_name,
        "num_external_folds": n_ext,
        "num_internal_folds": n_int,
        "ext_test_ratio": ext_test_ratio,
        "int_valid_ratio": int_valid_ratio,
        "random_seed": random_seed,
        "internal_splits": [f"{i}_internal.pt" for i in range(len(internal_folds))],
        "hash": kfold_hash,
        "hash_n": HASH_N,
        "hash_seed": HASH_SEED,
    }
    metadata["created_at"] = datetime.now().strftime("%d-%m-%y_%H-%M")
    existing_kfold_path = find_matching_kfold(metadata)
    if existing_kfold_path:
        print(f"Using existing kfold: {existing_kfold_path}")
        return existing_kfold_path

    new_dataset_folder = TENSOR_DATASETS_DIR / f"kfold_{formatted_size}"
    new_dataset_folder.mkdir(parents=True, exist_ok=True)
    dataset_name = f"{created_at}_rand_seed_{random_seed}"
    new_dataset_path = new_dataset_folder / dataset_name
    new_dataset_path.mkdir(parents=True)
    
    torch.save(external_fold, new_dataset_path / "external.pt")
    for i, internal_fold in enumerate(internal_folds):
        torch.save(internal_fold, new_dataset_path / f"{i}_internal.pt")
        
    metadata_path = new_dataset_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nK-fold dataset created at {new_dataset_path}")
    print(f"PATH: {new_dataset_path}")
    return new_dataset_path

def create_n_kfold(args) -> Path:
    """Create n_kfold dataset with global validation set for global weight optimization."""
    if args.dataset_size == "full":
        dataset_size = args.dataset_size
    else:
        try:
            dataset_size = int(args.dataset_size)
        except ValueError:
            raise ValueError(f"--dataset_size must be an integer, but got '{args.dataset_size}'.")
    
        if dataset_size >= 1_000_000:
            if dataset_size % 1_000_000 != 0:
                raise ValueError(
                    f"--dataset_size must be an exact multiple of 1,000,000 for sizes in the millions, "
                    f"but got {dataset_size}.\nExample: ...create_n_kfold --dataset_size 2000000 ..."
                )
        elif dataset_size >= 100_000:
            if dataset_size % 100_000 != 0:
                raise ValueError(
                    f"--dataset_size must be an exact multiple of 100,000 for sizes in the hundreds of thousands, "
                    f"but got {dataset_size}.\nExample: ...create_n_kfold --dataset_size 400000 ..."
                )
        else:
            raise ValueError(f"--dataset_size must be at least 100,000. Got {dataset_size}.")

    ext_test_ratio = args.ext_test_ratio or 0.2
    if ext_test_ratio > 0.3:
        raise ValueError(f"--ext_test_ratio value {ext_test_ratio} is higher than 0.3.")

    global_valid_ratio = args.global_valid_ratio or 0.1
    if global_valid_ratio > 0.2:
        raise ValueError(f"--global_valid_ratio value {global_valid_ratio} is higher than 0.2.")

    int_valid_ratio = args.int_valid_ratio or 0.25
    if int_valid_ratio > 0.3:
        raise ValueError(f"--int_valid_ratio value {int_valid_ratio} is higher than 0.3.")

    n_ext = int(round(1 / ext_test_ratio))
    if abs((1 / ext_test_ratio) - n_ext) > 1e-6:
        raise ValueError("ext_test_ratio must be the reciprocal of an integer (e.g. 0.2, 0.25, etc.).")
    n_global = int(round(1 / global_valid_ratio))
    if abs((1 / global_valid_ratio) - n_global) > 1e-6:
        raise ValueError("global_valid_ratio must be the reciprocal of an integer (e.g. 0.1, 0.125, etc.).")
    n_int = int(round(1 / int_valid_ratio))
    if abs((1 / int_valid_ratio) - n_int) > 1e-6:
        raise ValueError("int_valid_ratio must be the reciprocal of an integer (e.g. 0.25, 0.2, etc.).")

    random_seed = args.random_seed or 42

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert dataset_size is not None, "dataset_size is None"
    
    files = ["data", "labels", "data.names"]
    dataset = MendelianDataset(
        input_data_path=DEFAULT_INPUT_DATA_PATH,
        arrow_data_path=DEFAULT_ARROW_DATA_PATH,
        files=files,
        sample_n_random_rows = dataset_size if dataset_size != "full" else None,
        seed=random_seed,
        device=device,
    )
        
    num_internal_folds = args.num_internal_folds or 5
    
    n_external_fold, internal_folds = dataset.n_kfold_split(
        ext_test_ratio=ext_test_ratio,
        global_valid_ratio=global_valid_ratio,
        num_internal_folds=num_internal_folds,
        int_valid_ratio=int_valid_ratio,
        shuffle=True,
        random_seed=random_seed,
    )
    
    n_kfold_hash = hash_n_kfold(n_external_fold=n_external_fold, internal_folds=internal_folds)
    created_at = datetime.now().strftime("%d-%m-%y_%H-%M")
    dataset_name = f"{created_at}_rand_seed_{random_seed}"
    formatted_size = format_number(dataset_size) if isinstance(dataset_size, int) else dataset_size  
    assert dataset_size is not None, "dataset_size is None"
    
    metadata = {
        "created_at": created_at,
        "dataset_size": dataset_size,
        "split_name": dataset_name,
        "num_external_folds": n_ext,
        "num_global_valid_folds": n_global,
        "num_internal_folds": n_int,
        "ext_test_ratio": ext_test_ratio,
        "global_valid_ratio": global_valid_ratio,
        "int_valid_ratio": int_valid_ratio,
        "random_seed": random_seed,
        "internal_splits": [f"{i}_internal.pt" for i in range(len(internal_folds))],
        "hash": n_kfold_hash,
        "hash_n": HASH_N,
        "hash_seed": HASH_SEED,
    }
    metadata["created_at"] = datetime.now().strftime("%d-%m-%y_%H-%M")
    
    # Check for existing n_kfold dataset
    existing_n_kfold_path = find_matching_kfold(metadata)  # Reuse same matching logic
    if existing_n_kfold_path:
        print(f"Using existing n_kfold: {existing_n_kfold_path}")
        return existing_n_kfold_path

    new_dataset_folder = TENSOR_DATASETS_DIR / f"n_kfold_{formatted_size}"
    new_dataset_folder.mkdir(parents=True, exist_ok=True)
    dataset_name = f"{created_at}_rand_seed_{random_seed}"
    new_dataset_path = new_dataset_folder / dataset_name
    new_dataset_path.mkdir(parents=True)
    
    # Save n_external_fold (with global_valid)
    torch.save(n_external_fold, new_dataset_path / "external.pt")
    for i, internal_fold in enumerate(internal_folds):
        torch.save(internal_fold, new_dataset_path / f"{i}_internal.pt")
        
    metadata_path = new_dataset_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nN-Kfold dataset created at {new_dataset_path}")
    print(f"PATH: {new_dataset_path}")
    print(f"Structure:")
    print(f"  - External test: {ext_test_ratio*100:.1f}% of full data")
    print(f"  - Global validation: {(1-ext_test_ratio)*global_valid_ratio*100:.1f}% of full data")
    print(f"  - Available for internal folds: {(1-ext_test_ratio)*(1-global_valid_ratio)*100:.1f}% of full data")
    return new_dataset_path

def main():
    parser = argparse.ArgumentParser(description="Dataset Manager CLI")
    subparsers = parser.add_subparsers(dest="action", help="Available actions")
    subparsers.add_parser("list", help="List available datasets")

    create_dataset_parser = subparsers.add_parser("create_dataset", help="Create a dataset")
    create_dataset_parser.add_argument(
        "--dataset_size",
        type=str, 
        required=True,
        help='Size of the dataset to create ("full" for no limit)',
    )
    create_dataset_parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        required=True,
        help="Split ratios for train, valid, test",
    )
    create_dataset_parser.add_argument(
        "--positive_ratios",
        type=float,
        nargs=3, 
        required=True,
        help="Positive instance ratios for train, valid, test",
    )
    create_dataset_parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Random seed for dataset split",
    )
    
    # Subcommand: nested_kfold
    create_kfold_parser = subparsers.add_parser("create_kfold", help="Create a nested kfold")
    create_kfold_parser.add_argument(
        "--dataset_size",
        type=str,
        required=True,
        help='Size of the dataset to create ("full" for no limit)',
    )
    create_kfold_parser.add_argument(
        "--ext_test_ratio",
        type=float,
        required=True,
        help='The size of the external test split',
    )
    create_kfold_parser.add_argument(
        "--int_valid_ratio",
        type=float,
        required=True,
        help='The size of the internal validation split',
    )
    create_kfold_parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Random seed for dataset test and validation split",
    )
    
    create_n_kfold_parser = subparsers.add_parser("create_n_kfold", help="Create n_kfold with global validation set")
    create_n_kfold_parser.add_argument(
        "--dataset_size",
        type=str,
        required=True,
        help='Size of the dataset to create ("full" for no limit)',
    )
    create_n_kfold_parser.add_argument(
        "--ext_test_ratio",
        type=float,
        required=True,
        help='The size of the external test split',
    )
    create_n_kfold_parser.add_argument(
        "--global_valid_ratio",
        type=float,
        default=0.1,
        help='The ratio of external train to reserve for global validation (default: 0.1)',
    )
    create_n_kfold_parser.add_argument(
        "--num_internal_folds",
        type=int,
        default=5,
        help='The number of internal folds to create (default: 5)',
    )
    create_n_kfold_parser.add_argument(
        "--int_valid_ratio",
        type=float,
        default=0.2,
        help='The validation ratio within each internal fold (default: 0.2)',
    )
    create_n_kfold_parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Random seed for dataset test and validation split",
    )

    args = parser.parse_args()

    if args.action == "list":
        list_datasets()
    elif args.action == "create_dataset":
        create_dataset(args)
    elif args.action == "create_kfold":
        create_kfold(args)
    elif args.action == "create_n_kfold":
        create_n_kfold(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
