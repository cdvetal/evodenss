from __future__ import annotations

from enum import unique, Enum
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Union

from fast_denser.misc.proportions import ProportionsIndexes, ProportionsFloat
from fast_denser.neural_networks_torch.transformers import BaseTransformer

from sklearn.model_selection import train_test_split
from torch import Tensor
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
                 enable_stratify: bool,
                 proportions: Union[ProportionsFloat, ProportionsIndexes]) -> Dict[DatasetType, Subset]:

    train_data: Dataset
    test_data: Dataset
    if dataset_name == "fashion-mnist":
        train_data, test_data = _load_fashion_mnist(train_transformer, test_transformer)
    elif dataset_name == "mnist":
        train_data, evo_test_data, test_data = _load_mnist(train_transformer, test_transformer)
    elif dataset_name == "cifar10":
        train_data, evo_test_data, test_data = _load_cifar10(train_transformer, test_transformer)
    elif dataset_name == "cifar100":
        train_data, evo_test_data, test_data = _load_cifar100(train_transformer, test_transformer)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    subset_dict: Dict[DatasetType, Subset] = {}
    # In case we have received the indexes for each subset already
    if type(proportions) is ProportionsIndexes:
        for k in proportions.keys():
            if k == DatasetType.EVO_TEST:
                subset_dict[k] = Subset(evo_test_data, proportions[k])
            else:
                subset_dict[k] = Subset(train_data, proportions[k])
        subset_dict[DatasetType.TEST] = Subset(test_data, list(range(len(test_data.targets)))) # type: ignore
        return subset_dict

    else:
        # otherwise we'll do it based on the proportions that were asked
        assert type(proportions) == ProportionsFloat
        targets: Any = train_data.targets # type: ignore
        targets_tensor: Tensor = targets if type(targets) == "torch.Tensor" else Tensor(targets)
        train_indices: List[int] = list(range(len(targets))) # * aug_factor

        stratify: Any = targets_tensor if enable_stratify is True else None
        train_idx, test_idx = train_test_split(train_indices,
                                            test_size=proportions[DatasetType.EVO_TEST],
                                            shuffle=True,
                                            stratify=stratify)
        subset_dict[DatasetType.EVO_TEST] = Subset(evo_test_data, test_idx)

        if DatasetType.EVO_VALIDATION in proportions.keys():
            stratify = targets_tensor[train_idx] if enable_stratify is True else None
            real_validation_proportion: float = \
                proportions[DatasetType.EVO_VALIDATION]/(1 - proportions[DatasetType.EVO_TEST])
            train_idx, valid_idx = train_test_split(train_idx,
                                                    test_size=real_validation_proportion,
                                                    shuffle=True,
                                                    stratify=stratify)
            subset_dict[DatasetType.EVO_VALIDATION] = Subset(train_data, valid_idx)
        
        print(len(train_idx), len(test_idx))
        subset_dict[DatasetType.EVO_TRAIN] = Subset(train_data, train_idx)
        subset_dict[DatasetType.TEST] = Subset(test_data, list(range(len(test_data.targets)))) # type: ignore
        #
        #evo_train_subset: Subset = Subset(train_data, train_idx)
        #evo_validation_subset: Subset = Subset(train_data, valid_idx)
        #evo_test_subset: Subset = Subset(train_data, test_idx)
        #
        #return {
        #    DatasetType.EVO_TRAIN: evo_train_subset,
        #    DatasetType.EVO_VALIDATION: evo_validation_subset,
        #    DatasetType.EVO_TEST: evo_test_subset,
        #    DatasetType.TEST: test_data
        #}
        #train_idx, test_idx = train_test_split(train_indices,
        #                                       test_size=0.2,
        #                                       shuffle=True,
        #                                       stratify=stratify)
        #evo_train_subset: Subset = Subset(train_data, train_idx)
        #evo_test_subset: Subset = Subset(train_data, test_idx)

        #{
        #    DatasetType.EVO_TRAIN: evo_train_subset,
        #    DatasetType.EVO_TEST: evo_test_subset,
        #    DatasetType.TEST: test_data
        #}

        return subset_dict

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
                test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    evo_test_data = datasets.MNIST(
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
    return train_data, evo_test_data, test_data

def _load_cifar10(train_transformer: BaseTransformer,
                  test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    evo_test_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )

    return train_data, evo_test_data, test_data

def _load_cifar100(train_transformer: BaseTransformer,
                  test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    evo_test_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )

    return train_data, evo_test_data, test_data
