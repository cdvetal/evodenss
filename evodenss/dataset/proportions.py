from __future__ import annotations
from typing import Any, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from evodenss.dataset.dataset_loader import DatasetType

class ProportionsFloat:

    def __init__(self, proportions_dict: dict[DatasetType, float]):
        self.proportions_dict: dict[DatasetType, float] = proportions_dict

    def keys(self) -> Any:
        return self.proportions_dict.keys()

    def __getitem__(self, key: DatasetType) -> float:
        return self.proportions_dict[key]

    def __contains__(self, key: DatasetType) -> bool:
        return key in self.proportions_dict.keys()

    def __iter__(self) -> Iterator[DatasetType]:
        for key in self.proportions_dict:
            yield key

    def __len__(self) -> int:
        return len(self.proportions_dict)


class ProportionsIndexes:

    def __init__(self, proportions_dict: dict[DatasetType, list[int]]):
        self.proportions_dict: dict[DatasetType, list[int]] = proportions_dict

    def keys(self) -> Any:
        return self.proportions_dict.keys()

    def __getitem__(self, key: DatasetType) -> list[int]:
        return self.proportions_dict[key]

    def __contains__(self, key: DatasetType) -> bool:
        return key in self.proportions_dict.keys()

    def __iter__(self) -> Iterator[DatasetType]:
        for key in self.proportions_dict:
            yield key

    def __len__(self) -> int:
        return len(self.proportions_dict)
