from __future__ import annotations

from enum import unique, Enum
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from fast_denser.neural_networks_torch.transformers import BaseTransformer

from sklearn.model_selection import train_test_split
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
                 test_transformer: BaseTransformer,
                 enable_stratify: bool) -> Dict[DatasetType, Dataset]:
    train_data: Dataset
    test_data: Dataset
    if dataset_name == "fashion-mnist":
        train_data, test_data = _load_fashion_mnist(train_transformer, test_transformer)
    elif dataset_name == "mnist":
        train_data, test_data = _load_mnist(train_transformer, test_transformer)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    targets: List[Any] = train_data.targets # type: ignore
    train_indices: List[int] = list(range(len(targets))) # * aug_factor

    stratify: Any 
    stratify = targets if enable_stratify is True else None
    train_idx, test_idx = train_test_split(train_indices,
                                           test_size=0.2,
                                           shuffle=True,
                                           stratify=stratify)
    stratify = targets[train_idx] if enable_stratify is True else None
    train_idx, valid_idx = train_test_split(train_idx,
                                            test_size=0.25,
                                            shuffle=True,
                                            stratify=stratify) # 0.25 x 0.8 = 0.2
    
    evo_train_subset: Subset = Subset(train_data, train_idx)
    evo_validation_subset: Subset = Subset(train_data, valid_idx)
    evo_test_subset: Subset = Subset(train_data, test_idx)

    return {
        DatasetType.EVO_TRAIN: evo_train_subset,
        DatasetType.EVO_VALIDATION: evo_validation_subset,
        DatasetType.EVO_TEST: evo_test_subset,
        DatasetType.TEST: test_data
    }


def _load_fashion_mnist(train_transformer: BaseTransformer,
                        test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset]:
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )

    return train_data, test_data

def _load_mnist(train_transformer: BaseTransformer,
                test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset]:
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )
    return train_data, test_data
