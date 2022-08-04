from __future__ import annotations

from enum import unique, Enum
from math import floor
from typing import Dict, List, Tuple, TYPE_CHECKING

from fast_denser.neural_networks_torch.transformers import BaseTransformer

import numpy as np
from torch.utils.data import Subset
from torchvision import datasets

if TYPE_CHECKING:
    from torch.utils.data import Dataset

__all__ = ['DatasetType', 'load_dataset']


@unique
class DatasetType(Enum):
    EVO_TRAIN = "evo_train"
    EVO_VALIDATION = "evo_validation"
    EVO_TEST = "evo_test"
    TEST = "test"


def load_dataset(dataset_name: str,
                 train_transformer: BaseTransformer,
                 test_transformer: BaseTransformer) -> Dict[DatasetType, Dataset]:
    train_data: Dataset
    train_data_for_fitness: Dataset
    test_data: Dataset
    if dataset_name == "fashion-mnist":
        train_data, train_data_for_fitness, test_data = _load_fashion_mnist(train_transformer, test_transformer)
    elif dataset_name == "mnist":
        train_data, train_data_for_fitness, test_data = _load_mnist(train_transformer, test_transformer)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    #print(type(train_data))
    #print(train_data.targets)
    #print(train_data.data)
    #import sys
    #sys.exit(0)
    train_size: int = len(train_data) # type: ignore
    aug_factor: int = 1 # 4 is thrown randomly
    train_indices: List[int] = list(range(train_size)) * aug_factor 
    real_train_size: int = train_size * aug_factor
    np.random.shuffle(train_indices)
    evo_train_end_idx: int = floor(0.7 * train_size)
    evo_validation_start_idx: int = evo_train_end_idx
    evo_validation_end_idx: int = evo_train_end_idx + floor(0.2 * train_size)
    evo_test_start_idx: int = evo_validation_end_idx
    
    evo_train_subset: Subset = Subset(train_data, train_indices[:evo_train_end_idx])
    evo_validation_subset: Subset = Subset(train_data,
                                           train_indices[evo_validation_start_idx:evo_validation_end_idx])
    evo_test_subset: Subset = Subset(train_data_for_fitness, train_indices[evo_test_start_idx:])
    
    return {
        DatasetType.EVO_TRAIN: evo_train_subset,
        DatasetType.EVO_VALIDATION: evo_validation_subset,
        DatasetType.EVO_TEST: evo_test_subset,
        DatasetType.TEST: test_data
    }


def _load_fashion_mnist(train_transformer: BaseTransformer,
                        test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    train_data_for_fitness = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )

    return train_data, train_data_for_fitness, test_data

def _load_mnist(train_transformer: BaseTransformer,
                test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    train_data_for_fitness = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )
    return train_data, train_data_for_fitness, test_data
