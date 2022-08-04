from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from fast_denser.evolution import Individual
    from fast_denser.neural_networks_torch.evaluators import BaseEvaluator


@dataclass
class Checkpoint:
    run: int
    random_state: Any
    numpy_random_state: Dict[str, Any]
    torch_random_state: Tensor
    last_processed_generation: int
    total_epochs: int
    best_fitness: Optional[float]
    evaluator: BaseEvaluator
    population: Optional[List[Individual]] = field(default=None)
    parent: Optional[Individual] = field(default=None)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Checkpoint):
            return self.__dict__ == other.__dict__ 
        return False
