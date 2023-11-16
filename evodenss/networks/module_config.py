from dataclasses import dataclass
from typing import List

@dataclass
class ModuleConfig:
    min_expansions: int
    max_expansions: int
    initial_network_structure: List[int]
    levels_back: int

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModuleConfig):
            return self.__dict__ == other.__dict__
        return False
