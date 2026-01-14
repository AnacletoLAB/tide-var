import sys
from pathlib import Path

project_root = (Path(__file__).resolve().parents[2])
sys.path.insert(0, str(project_root))

import os
from pathlib import Path
from typing import List, Tuple, Union, Optional
import shutil
import math
from dataclasses import dataclass

import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset

from tremm.utils.downloader import download_data

FILE_ID = "1bsaFUh6dmgWD60HUIJieaO2BzxHo_B7E"

@dataclass 
class DatasetEntry(Dataset):
    X: torch.Tensor
    y: torch.Tensor
    
    def __len__(self):
            return self.X.size(0)
            
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

@dataclass
class DatasetSplit:
    train: DatasetEntry
    test: DatasetEntry
    valid: Optional[DatasetEntry] = None
    
    def __iter__(self):
        yield self.train
        if self.valid is not None:
            yield self.valid
        if self.test is not None:
            yield self.test

@dataclass
class ExternalFold:
    train: DatasetEntry
    test: DatasetEntry
    
    def __iter__(self):
        yield self.train
        yield self.test

@dataclass
class NExternalFold:
    """External fold with global validation set for n_kfold datasets."""
    train: DatasetEntry        # Available for internal folds (after global_valid is removed)
    test: DatasetEntry         # External test set 
    global_valid: DatasetEntry # Global validation set for global weight optimization
    
    def __iter__(self):
        yield self.train
        yield self.test
        yield self.global_valid

@dataclass
class InternalFold:
    train: DatasetEntry
    valid: DatasetEntry

    def __iter__(self):
        yield self.train
        yield self.valid

def txt_to_arrow(input_data_path: Path, arrow_data_path: Path, files: List[str]) -> None:
    for file in files:
        has_header = file == "data" or file == "mini_data"

        df = pl.read_csv(
            input_data_path / f"{file}.txt",
            separator=" ", 
            has_header=has_header,
            infer_schema_length=100 
        )

        df.write_ipc(f"{arrow_data_path}/{file}.arrow")

def read_arrow_files(
    arrow_data_path: Path,
    files: List[str],
    num_rows: Optional[int] = None,
    sample_n_random_rows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    seed: int = 42,
) -> List[pl.DataFrame]:
    dfs = []
    labels_df = None

    for file in files:
        df = pl.read_ipc(arrow_data_path / f"{file}.arrow")
        if usecols is not None and file == "data":
            df = df.select(usecols)
        if num_rows is not None:
            df = df.head(num_rows)
        if file == "labels" and sample_n_random_rows is not None:
            labels_df = df
        dfs.append(df)

    if sample_n_random_rows is not None and labels_df is not None:
        labels_df = labels_df.with_row_index()
        label_1_indices = labels_df.filter(pl.col(labels_df.columns[1]) == 1).to_numpy()[:, 0]

        remaining_rows_to_sample = sample_n_random_rows - len(label_1_indices)

        all_indices = np.arange(labels_df.height)
        non_label_1_indices = np.setdiff1d(all_indices, label_1_indices)

        np.random.seed(seed)
        sampled_indices = np.random.choice(non_label_1_indices, size=remaining_rows_to_sample, replace=False)
        sample_idx = np.concatenate([label_1_indices, sampled_indices])
        dfs = [df[sample_idx] for df in dfs]
    return dfs
    
def check_raw_data(input_data_path: Path):
    folder_files = {"data.names.txt", "data.txt", "folds.txt", "labels.txt"}
    
    def get_existing_files(path: Path):
        return {f for f in os.listdir(path) if f not in {"README", "__init__.py"}}
    
    if not input_data_path.exists():
        download_data(file_id=FILE_ID, destination=input_data_path)
    
    existing_files = get_existing_files(input_data_path)
    if folder_files != existing_files:
        for item in os.listdir(input_data_path):
            if item in {"__init__.py", "README"}:
                continue
            item_path = input_data_path / item
            if item_path.is_file():
                item_path.unlink()
            elif item_path.is_dir():
                shutil.rmtree(item_path)
        
        download_data(file_id=FILE_ID, destination=input_data_path)
        
        existing_files = get_existing_files(input_data_path)
        if folder_files != existing_files:
            raise Exception("Raw data files are incorrect after re-downloading.")
    
    print("Raw data check passed.")

class MendelianDataset(Dataset):
    def __init__(self, input_data_path: Path,
                 arrow_data_path: Path,
                 files: List[str],
                 num_rows: Optional[int] = None,
                 sample_n_random_rows: Optional[int] = None,
                 usecols: Optional[List[str]] = None,
                 seed: int = 42,
                 device: str = 'cpu') -> None:
        check_raw_data(input_data_path=input_data_path)

        if not os.path.exists(arrow_data_path / f"{files[0]}.arrow"):
            txt_to_arrow(input_data_path=input_data_path, arrow_data_path=arrow_data_path, files=files)

        x_tmp, y_tmp, self.sample_names = read_arrow_files(arrow_data_path=arrow_data_path,
                                                           files=files,
                                                           num_rows=num_rows,
                                                           sample_n_random_rows=sample_n_random_rows,
                                                           usecols=usecols,
                                                           seed=seed)

        self.x_data = x_tmp.to_torch(dtype=pl.Float32).to(device)
        self.y_data = y_tmp.to_torch(dtype=pl.Int8).squeeze().to(device)
        self.names_features = x_tmp.columns

        column = self.sample_names.columns[0]
        self.chromosomes = (
            self.sample_names[column]
            .str.extract(r"chr([a-zA-Z0-9]+)\.", 1)
            .to_numpy()
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x_data)

    def __getitem__(self, idx: int) -> dict:
        feats = self.x_data[idx]
        label = self.y_data[idx]
        name = self.sample_names[idx]
        return {'features': feats, 'label': label, 'name': name}
        
    def get_chromosomes(self) -> np.ndarray:
        """Return the array of chromosomes."""
        return self.chromosomes

    def train_valid_test_split(
            self,
            split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            positive_ratios: Tuple[float, float, float] = (0.5, 0.25, 0.25),
            shuffle: bool = True,
            random_seed: int = 42,
    ) -> DatasetSplit:
        train_ratio, valid_ratio, test_ratio = split_ratios

        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "split_ratios must sum to 1."
        assert abs(sum(positive_ratios) - 1.0) < 1e-6, "positive_ratios must sum to 1."

        positive_indices = np.where(self.y_data.cpu() == 1)[0]
        negative_indices = np.where(self.y_data.cpu() == 0)[0]

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(positive_indices)
            np.random.shuffle(negative_indices)

        total_size = len(self.y_data)
        total_pos = len(positive_indices)

        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)

        train_pos_size = int(total_pos * positive_ratios[0])
        valid_pos_size = int(total_pos * positive_ratios[1])

        train_neg_size = train_size - train_pos_size
        valid_neg_size = valid_size - valid_pos_size

        train_pos_indices = positive_indices[:train_pos_size]
        valid_pos_indices = positive_indices[train_pos_size:train_pos_size + valid_pos_size]
        test_pos_indices = positive_indices[train_pos_size + valid_pos_size:]

        train_neg_indices = negative_indices[:train_neg_size]
        valid_neg_indices = negative_indices[train_neg_size:train_neg_size + valid_neg_size]
        test_neg_indices = negative_indices[train_neg_size + valid_neg_size:]

        train_indices = np.concatenate([train_pos_indices, train_neg_indices])
        valid_indices = np.concatenate([valid_pos_indices, valid_neg_indices])
        test_indices = np.concatenate([test_pos_indices, test_neg_indices])

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(train_indices)
            np.random.shuffle(valid_indices)
            np.random.shuffle(test_indices)
        
        return DatasetSplit(
            train=DatasetEntry(
                X=self.x_data[train_indices],
                y=self.y_data[train_indices]
            ),
            valid=DatasetEntry(
                X=self.x_data[valid_indices],
                y=self.y_data[valid_indices]
            ),
            test=DatasetEntry(
                X=self.x_data[test_indices],
                y=self.y_data[test_indices]
            ),
        )
                
    def kfold_split(
            self,
            ext_test_ratio: float = 0.2,
            int_valid_ratio: float = 0.2,
            shuffle: bool = True,
            random_seed: int = 42,
    ) -> Tuple[ExternalFold, List[InternalFold]]:
        X, y = self.x_data, self.y_data        
        n_ext = int(round(1 / ext_test_ratio))
        n_int = int(round(1 / int_valid_ratio))
        total_n_positives = torch.where(y == 1)[0].numel()

        external_kf = StratifiedKFold(n_splits=n_ext, shuffle=shuffle, random_state=random_seed)
        external_train_index, external_test_index = next(external_kf.split(X=X.cpu(), y=y.cpu()))
        
        external_train_X, external_train_y = X[external_train_index], y[external_train_index]
        external_test_X, external_test_y = X[external_test_index], y[external_test_index]

        n_positives_in_ext_train = torch.where(external_train_y == 1)[0].numel()
        n_positives_in_ext_test = torch.where(external_test_y == 1)[0].numel()
        expected_ext_train_ratio = 1 - ext_test_ratio
        actual_external_train_ratio = round(n_positives_in_ext_train / total_n_positives, 2)
        actual_external_test_ratio = round(n_positives_in_ext_test / total_n_positives, 2)

        assert math.isclose(
            actual_external_train_ratio, expected_ext_train_ratio, rel_tol=1e-2,
        ), f"External train ratios are incorrect, actual ratio: {actual_external_train_ratio}, " \
            f"expected ratio: {expected_ext_train_ratio}"
        print(f"Actual external train ratio: {actual_external_train_ratio}, expected: {expected_ext_train_ratio}")

        assert math.isclose(
            actual_external_test_ratio, ext_test_ratio, rel_tol=1e-2,
        ), f"External test ratios are incorrect, actual ratio: {actual_external_test_ratio}, " \
            f"expected ratio: {ext_test_ratio}"
        print(f"Actual external test ratio: {actual_external_test_ratio}, expected: {ext_test_ratio}")
        
        external_fold = ExternalFold(
            train=DatasetEntry(
                X=external_train_X,
                y=external_train_y,
            ),
            test=DatasetEntry(
                X=external_test_X,
                y=external_test_y,
            )
        )

        internal_folds = []  
        internal_kf = StratifiedKFold(n_splits=n_int, shuffle=shuffle, random_state=random_seed)

        for j, (internal_train_index, internal_valid_index) in enumerate(
            internal_kf.split(X=external_train_X.cpu(), y=external_train_y.cpu())
        ):
            internal_train_X = external_train_X[internal_train_index]
            internal_train_y = external_train_y[internal_train_index]
            internal_valid_X = external_train_X[internal_valid_index]
            internal_valid_y = external_train_y[internal_valid_index]

            n_positives_in_int_train = torch.where(internal_train_y == 1)[0].numel()
            n_positives_in_int_valid = torch.where(internal_valid_y == 1)[0].numel()
            expected_int_train_ratio = 1 - int_valid_ratio
            actual_internal_train_ratio = round(n_positives_in_int_train / n_positives_in_ext_train, 2)
            actual_internal_valid_ratio = round(n_positives_in_int_valid / n_positives_in_ext_train, 2)
            
            assert math.isclose(
                actual_internal_train_ratio, expected_int_train_ratio, rel_tol=1e-2,
            ), f"Internal train ratios are incorrect, actual ratio: {actual_internal_train_ratio}, " \
                f"expected ratio: {expected_int_train_ratio}"
            print(f"{j}. Actual internal train ratio: {actual_internal_train_ratio}, expected: {expected_int_train_ratio}")
            assert math.isclose(
                actual_internal_valid_ratio, ext_test_ratio, rel_tol=1e-2,
            ), f"Internal valid ratios are incorrect, actual ratio: {actual_internal_valid_ratio}, " \
                f"expected ratio: {int_valid_ratio}"
            print(f"   Actual internal valid ratio: {actual_internal_valid_ratio}, expected: {int_valid_ratio}")
            print()
            
            internal_fold = InternalFold(
                train=DatasetEntry(
                    X=internal_train_X,
                    y=internal_train_y,
                ),
                valid=DatasetEntry(
                    X=internal_valid_X,
                    y=internal_valid_y,
                )
            )
            internal_folds.append(internal_fold)                    
            
        return external_fold, internal_folds

    def n_kfold_split(
            self,
            ext_test_ratio: float = 0.2,
            global_valid_ratio: float = 0.1,
            num_internal_folds: int = 5,
            int_valid_ratio: float = 0.2,
            shuffle: bool = True,
            random_seed: int = 42,
    ) -> Tuple[NExternalFold, List[InternalFold]]:
        """
        Create n_kfold splits with global validation set for global weight optimization.
        
        Args:
            ext_test_ratio: Ratio for external test set (default 0.2 = 20%)
            global_valid_ratio: Ratio of external train to reserve for global validation (default 0.1 = 10%)
            num_internal_folds: Number of internal folds to create (default 5)
            int_valid_ratio: Ratio for internal validation within each fold (default 0.2 = 20%)
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (n_external_fold, list_of_internal_folds)
            
        Structure:
            Full Dataset (100%)
            ├── External Test (20%)
            └── External Train (80%)
                ├── Global Validation (8% of full = 10% of external train)
                └── Available for Internal Folds (72% of full = 90% of external train)
                    └── Split into {num_internal_folds} folds, each with {int_valid_ratio*100}% validation
        """
        X, y = self.x_data, self.y_data
        n_ext = int(round(1 / ext_test_ratio))
        n_int = num_internal_folds  # Use direct parameter instead of deriving from ratio
        total_n_positives = torch.where(y == 1)[0].numel()

        print(f"Creating n_kfold split with global validation:")
        print(f"  External test ratio: {ext_test_ratio} ({ext_test_ratio*100:.1f}%)")
        print(f"  Global validation ratio: {global_valid_ratio} of external train ({(1-ext_test_ratio)*global_valid_ratio:.1f}% of full data)")
        print(f"  Number of internal folds: {num_internal_folds}")
        print(f"  Internal validation ratio per fold: {int_valid_ratio} ({int_valid_ratio*100:.1f}%)")

        # Step 1: Split into external train and test
        external_kf = StratifiedKFold(n_splits=n_ext, shuffle=shuffle, random_state=random_seed)
        external_train_index, external_test_index = next(external_kf.split(X=X.cpu(), y=y.cpu()))
        
        external_train_X, external_train_y = X[external_train_index], y[external_train_index]
        external_test_X, external_test_y = X[external_test_index], y[external_test_index]

        # Step 2: Split external train into global validation and available for internal folds
        n_global_valid = int(round(1 / global_valid_ratio))
        global_valid_kf = StratifiedKFold(n_splits=n_global_valid, shuffle=shuffle, random_state=random_seed + 1)
        available_for_internal_index, global_valid_index = next(global_valid_kf.split(X=external_train_X.cpu(), y=external_train_y.cpu()))
        
        available_for_internal_X = external_train_X[available_for_internal_index]
        available_for_internal_y = external_train_y[available_for_internal_index]
        global_valid_X = external_train_X[global_valid_index]
        global_valid_y = external_train_y[global_valid_index]

        # Verify ratios
        n_positives_in_ext_test = torch.where(external_test_y == 1)[0].numel()
        n_positives_in_global_valid = torch.where(global_valid_y == 1)[0].numel()
        n_positives_in_available = torch.where(available_for_internal_y == 1)[0].numel()
        
        actual_external_test_ratio = round(n_positives_in_ext_test / total_n_positives, 3)
        actual_global_valid_ratio = round(n_positives_in_global_valid / total_n_positives, 3)
        actual_available_ratio = round(n_positives_in_available / total_n_positives, 3)
        
        print(f"Actual ratios:")
        print(f"  External test: {actual_external_test_ratio:.3f} (expected {ext_test_ratio:.3f})")
        print(f"  Global validation: {actual_global_valid_ratio:.3f} (expected {(1-ext_test_ratio)*global_valid_ratio:.3f})")
        print(f"  Available for internal: {actual_available_ratio:.3f} (expected {(1-ext_test_ratio)*(1-global_valid_ratio):.3f})")

        # Create NExternalFold
        n_external_fold = NExternalFold(
            train=DatasetEntry(
                X=available_for_internal_X,
                y=available_for_internal_y,
            ),
            test=DatasetEntry(
                X=external_test_X,
                y=external_test_y,
            ),
            global_valid=DatasetEntry(
                X=global_valid_X,
                y=global_valid_y,
            )
        )

        # Step 3: Create internal folds from available data
        internal_folds = []  
        internal_kf = StratifiedKFold(n_splits=n_int, shuffle=shuffle, random_state=random_seed + 2)

        n_positives_in_available = torch.where(available_for_internal_y == 1)[0].numel()

        for j, (internal_train_index, internal_valid_index) in enumerate(
            internal_kf.split(X=available_for_internal_X.cpu(), y=available_for_internal_y.cpu())
        ):
            internal_train_X = available_for_internal_X[internal_train_index]
            internal_train_y = available_for_internal_y[internal_train_index]
            internal_valid_X = available_for_internal_X[internal_valid_index]
            internal_valid_y = available_for_internal_y[internal_valid_index]

            n_positives_in_int_train = torch.where(internal_train_y == 1)[0].numel()
            n_positives_in_int_valid = torch.where(internal_valid_y == 1)[0].numel()
            expected_int_train_ratio = 1 - int_valid_ratio
            actual_internal_train_ratio = round(n_positives_in_int_train / n_positives_in_available, 2)
            actual_internal_valid_ratio = round(n_positives_in_int_valid / n_positives_in_available, 2)
            
            assert math.isclose(
                actual_internal_train_ratio, expected_int_train_ratio, rel_tol=1e-2,
            ), f"Internal train ratios are incorrect, actual ratio: {actual_internal_train_ratio}, " \
                f"expected ratio: {expected_int_train_ratio}"
            print(f"{j}. Actual internal train ratio: {actual_internal_train_ratio}, expected: {expected_int_train_ratio}")
            assert math.isclose(
                actual_internal_valid_ratio, int_valid_ratio, rel_tol=1e-2,
            ), f"Internal valid ratios are incorrect, actual ratio: {actual_internal_valid_ratio}, " \
                f"expected ratio: {int_valid_ratio}"
            print(f"   Actual internal valid ratio: {actual_internal_valid_ratio}, expected: {int_valid_ratio}")
            print()
            
            internal_fold = InternalFold(
                train=DatasetEntry(
                    X=internal_train_X,
                    y=internal_train_y,
                ),
                valid=DatasetEntry(
                    X=internal_valid_X,
                    y=internal_valid_y,
                )
            )
            internal_folds.append(internal_fold)                    
            
        return n_external_fold, internal_folds
                    
    def do_chrom_aware_holdout(self, chrom: Union[str, Tuple[str, ...]] = '1') -> Tuple[Subset, Subset]:
        # Ensure chrom is iterable; if a single chromosome, make it a tuple
        if isinstance(chrom, str):
            chrom = (chrom,)

        # Collect indices of samples that belong to the specified chromosome(s)
        ind_test = np.concatenate([np.where(self.chromosomes == c)[0] for c in chrom])

        # Define test and train subsets
        test_data = Subset(self, ind_test)
        total_samples = self.__len__()
        ind_train = np.setdiff1d(np.arange(total_samples), ind_test, assume_unique=True)
        train_data = Subset(self, ind_train)

        return train_data, test_data