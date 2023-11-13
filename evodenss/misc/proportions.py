from __future__ import annotations
from typing import Any, Dict, Iterator, List, TYPE_CHECKING

if TYPE_CHECKING:
    from evodenss.networks.torch.dataset_loader import DatasetType

class ProportionsFloat:
    
    def __init__(self, proportions_dict: Dict[DatasetType, float]):
        self.proportions_dict: Dict[DatasetType, float] = proportions_dict

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
    
    def __init__(self, proportions_dict: Dict[DatasetType, List[int]]):
        self.proportions_dict: Dict[DatasetType, List[int]] = proportions_dict

    def keys(self) -> Any:
        return self.proportions_dict.keys()

    def __getitem__(self, key: DatasetType) -> List[int]:
        return self.proportions_dict[key]

    def __contains__(self, key: DatasetType) -> bool:
        return key in self.proportions_dict.keys()

    def __iter__(self) -> Iterator[DatasetType]:
        for key in self.proportions_dict:
            yield key

    def __len__(self) -> int:
        return len(self.proportions_dict)