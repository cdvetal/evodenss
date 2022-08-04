from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from typing import Any, Iterator, List

@dataclass
class EvaluationMetrics:
    is_valid_solution: bool
    fitness: float
    n_trainable_parameters: int
    n_layers: int
    training_time_spent: float
    validation_losses: List[float]
    n_epochs: int

    @classmethod
    def default(cls) -> "EvaluationMetrics":
        return EvaluationMetrics(
            is_valid_solution=False,
            fitness=-1,
            n_trainable_parameters=-1,
            n_layers=0,
            training_time_spent=0.0,
            validation_losses=[],
            n_epochs=0
        )

    @classmethod
    def list_fields(cls) -> List[str]:
        return [field.name for field in fields(cls)]

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self)) 
