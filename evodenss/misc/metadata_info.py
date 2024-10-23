from __future__ import annotations

from typing import Any, Optional, Sequence

from pydantic import BaseModel
from torch.utils.data import Subset

from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetType
from evodenss.networks.phenotype_parser import Optimiser, Pretext
from evodenss.train.learning_parameters import LearningParams


class TrainingInfo(BaseModel):
    dataset_name: str
    train_indices: Sequence[int]
    validation_indices: Sequence[int]
    test_indices: Sequence[int]
    optimiser_name: str
    optimiser_parameters: dict[str, Any]
    batch_size: int
    early_stop: Optional[int]
    trained_epochs: int


class PretextTrainingInfo(TrainingInfo, BaseModel):
    pretext_algorithm_name: str
    pretext_algorithm_params: dict[str, Any]


class MetadataInfo(BaseModel):
    pretext_training_info: Optional[PretextTrainingInfo]
    downstream_training_info: Optional[TrainingInfo]

    @classmethod
    def new_instance(cls,
                     dataset_name: str,
                     dataset: dict[DatasetType, Subset[ConcreteDataset]],
                     optimiser: Optimiser,
                     learning_params: LearningParams,
                     pretext_task: Optional[Pretext]) -> 'MetadataInfo':
        pretext_training_info: Optional[PretextTrainingInfo] = None
        downstream_training_info: Optional[TrainingInfo] = None
        if pretext_task is not None:
            pretext_training_info = PretextTrainingInfo(
                dataset_name=dataset_name,
                train_indices=dataset[DatasetType.PRETEXT_TRAIN].indices,
                validation_indices=[],
                test_indices=dataset[DatasetType.EVO_TEST].indices,
                optimiser_name=optimiser.optimiser_type,
                optimiser_parameters=optimiser.optimiser_parameters,
                batch_size=learning_params.batch_size,
                early_stop=learning_params.early_stop,
                pretext_algorithm_name=pretext_task.pretext_type.value,
                pretext_algorithm_params=pretext_task.pretext_parameters,
                trained_epochs=0
            )
        else:
            downstream_training_info = TrainingInfo(
                dataset_name=dataset_name,
                train_indices=dataset[DatasetType.DOWNSTREAM_TRAIN].indices,
                validation_indices=dataset[DatasetType.VALIDATION].indices,
                test_indices=dataset[DatasetType.EVO_TEST].indices,
                optimiser_name=optimiser.optimiser_type,
                optimiser_parameters=optimiser.optimiser_parameters,
                batch_size=learning_params.batch_size,
                early_stop=learning_params.early_stop,
                trained_epochs=0
            )
        return cls(pretext_training_info=pretext_training_info,
                   downstream_training_info=downstream_training_info)
