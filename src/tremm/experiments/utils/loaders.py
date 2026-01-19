import sys
from pathlib import Path

project_root = (Path(__file__).resolve().parents[2])
sys.path.insert(0, str(project_root))

from pathlib import Path
from typing import Dict, Union, Literal, Optional, overload
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import re

from tremm.data.mendelian_dataset import DatasetEntry, ExternalFold, InternalFold

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(42)  

@dataclass
class DatasetLoader:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader

@dataclass
class ExternalDataLoader:
    train_loader: DataLoader
    test_loader: DataLoader

@dataclass
class InternalDataLoader:
    train_loader: DataLoader
    valid_loader: DataLoader

class BalancedBatchSampler:
    def __init__(self, labels, batch_size, neg_batch_ratio):
        self.labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        self.batch_size = batch_size
        self.neg_batch_ratio = neg_batch_ratio

        self.neg_indices = np.where(self.labels == 0)[0]
        self.pos_indices = np.where(self.labels == 1)[0]

        self.num_neg_per_batch = int(batch_size * neg_batch_ratio)
        self.num_pos_per_batch = batch_size - self.num_neg_per_batch

        self.num_neg_samples = len(self.neg_indices)

        self.num_batches = self.num_neg_samples // self.num_neg_per_batch

        self.leftover_negatives = self.num_neg_samples % self.num_neg_per_batch
        if self.leftover_negatives > 0:
            self.num_batches += 1

    def __iter__(self):
        neg_idx = np.random.permutation(self.neg_indices)
        pos_idx = np.random.permutation(self.pos_indices)
    
        batch_start = 0
        for batch_num in range(self.num_batches):
            if (batch_num == self.num_batches - 1) and (self.leftover_negatives > 0):
                num_neg_in_batch = self.leftover_negatives
                num_pos_in_batch = self.leftover_negatives
            else:
                num_neg_in_batch = self.num_neg_per_batch
                num_pos_in_batch = self.num_pos_per_batch
    
            neg_batch = neg_idx[batch_start:batch_start + num_neg_in_batch]
            batch_start += num_neg_in_batch
    
            pos_batch = np.random.choice(pos_idx, num_pos_in_batch, replace=True)
    
            batch_indices = np.concatenate([neg_batch, pos_batch])
            np.random.shuffle(batch_indices)
            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches    

def compute_normalization_stats(train: DatasetEntry):
    mean = train.X.mean(0, keepdim=True)
    std  = train.X.std(0, unbiased=False, keepdim=True)
    return mean, std

def apply_normalization(
    split: DatasetEntry,
    mean: torch.Tensor,
    std: torch.Tensor
):
    split.X = (split.X - mean) / std
    return split

def load_internal_datasets(
    dataset_folder_size: str,
    dataset_folder_name: str,
    dataset_name: Path,
    regex: bool = True,
) -> InternalFold:
    datasets_path = Path(__file__).parents[2] / "data" / "tensors" / dataset_folder_size / dataset_folder_name
    internal = None 
    split_name = None
    
    if regex:
        m = re.search(r'(\d+_internal)', dataset_name.name)
        if m:
            split_name = m.group(1)
    
    assert split_name is not None, "split_name is None"
    for dataset in datasets_path.iterdir():
        if split_name in dataset.name:
            internal = torch.load(dataset)
    assert internal is not None, "internal is None"
    return internal

def load_external_datasets(
    dataset_folder_size: str,
    dataset_folder_name: str,
) -> ExternalFold:    
    datasets_path = Path(__file__).parents[2] / "data" / "tensors" / dataset_folder_size / dataset_folder_name
    external = None 
    
    for dataset in datasets_path.iterdir():
        if "external" in dataset.name:
            external = torch.load(dataset)
            
    assert external is not None, "external is None"
    return external

def get_train_loader(
    training_set: DatasetEntry,
    balanced: bool = False,
    batch_size: int = 8192,
    neg_batch_ratio: float = 0.8,
    return_indices: bool = False,
) -> DataLoader:
    if balanced:
            train_sampler = BalancedBatchSampler(
                labels=training_set.y,
                batch_size=batch_size,
                neg_batch_ratio=neg_batch_ratio,
            )
            train_loader = DataLoader(dataset=training_set, batch_sampler=train_sampler)
    else:
        train_loader = DataLoader(dataset=training_set, batch_size=batch_size)
        
    return train_loader

def get_valid_loader(
    valid_set: DatasetEntry,
    batch_size: Optional[int] = None,
) -> DataLoader:
    if batch_size:
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size) # type: ignore
    else:
        valid_loader = DataLoader(dataset=valid_set) # type: ignore
    return valid_loader 

def get_test_loader(
    test_set: DatasetEntry,
    batch_size: Optional[int] = None,
) -> DataLoader:
    if batch_size:
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size) # type: ignore
    else:
        test_loader = DataLoader(dataset=test_set) # type: ignore
    return test_loader

@overload
def load_datasets(
    split_name: str,
    load_external: Literal[False] = False,
    use_normalization: bool = False,
    clustering: Optional[str] = None, 
    clustering_params: Optional[Dict] = None,
) -> Tuple[DatasetEntry, DatasetEntry]: ...

@overload
def load_datasets(
    split_name: str,
    load_external: Literal[True],
    use_normalization: bool = False,
    clustering: Optional[str] = None, 
    clustering_params: Optional[Dict] = None,
) -> Tuple[DatasetEntry, DatasetEntry, DatasetEntry]: ...

def load_datasets(
    split_name: str,
    load_external: bool = False,
    use_normalization: bool = False,
    clustering: Optional[str] = None, 
    clustering_params: Optional[Dict] = None,
    sample_pos: Optional[str] = None, 
    sample_pos_params: Optional[Dict] = None,
) -> Union[Tuple[DatasetEntry, DatasetEntry], Tuple[DatasetEntry, DatasetEntry, DatasetEntry]]:

    dataset_folder_size, dataset_folder_name, dataset_name_str = split_name.split("/")
    dataset_path_name = Path(dataset_name_str)

    internal = load_internal_datasets(
        dataset_folder_size=dataset_folder_size,
        dataset_folder_name=dataset_folder_name,
        dataset_name=dataset_path_name,
        regex=True,
    )

    training_set: DatasetEntry = internal.train
    valid_set: DatasetEntry = internal.valid
    test_set: Optional[DatasetEntry] = None

    if load_external:
        external = load_external_datasets(
            dataset_folder_size=dataset_folder_size,
            dataset_folder_name=dataset_folder_name,
        )
        test_set = external.test
            
    if use_normalization:
        mean, std = compute_normalization_stats(
            train=training_set 
        )

        training_set = apply_normalization(
            split=training_set,
            mean=mean,
            std=std
        )

        valid_set = apply_normalization(
            split=valid_set,
            mean=mean,
            std=std
        )

        if test_set is not None:
            test_set = apply_normalization(
                split=test_set,
                mean=mean,
                std=std
            )
    
    if load_external:
        assert test_set is not None, \
            "Internal logic error: current_test_set should be populated if load_external is True."
        return training_set, valid_set, test_set
    else:
        return training_set, valid_set