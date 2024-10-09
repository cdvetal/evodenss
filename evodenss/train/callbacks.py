from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
from time import time
from typing import Any, Dict, TYPE_CHECKING

import torch

from evodenss.misc.constants import METADATA_FILENAME, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.networks.evolved_networks import BarlowTwinsNetwork


if TYPE_CHECKING:
    from evodenss.misc.metadata_info import MetadataInfo
    from evodenss.train.trainers import Trainer


logger = logging.getLogger(__name__)


class Callback(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def on_train_begin(self, trainer: Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_train_end(self, trainer: Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_epoch_begin(self, trainer: Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer) -> None:
        raise NotImplementedError()


# Needs tweak to save individual with lowest validation error
class ModelCheckpointCallback(Callback):

    def __init__(self,
                 model_saving_dir: str,
                 metadata_info: MetadataInfo,
                 model_filename: str=MODEL_FILENAME,
                 weights_filename: str=WEIGHTS_FILENAME,
                 metadata_filename: str=METADATA_FILENAME) -> None:
        self.model_saving_dir: str = model_saving_dir
        self.model_filename: str = model_filename
        self.metadata_filename: str = metadata_filename
        self.weights_filename: str = weights_filename
        self.metadata_info: MetadataInfo = metadata_info


    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        if isinstance(trainer.model, BarlowTwinsNetwork):
            assert self.metadata_info.pretext_training_info is not None
            self.metadata_info.pretext_training_info.trained_epochs = trainer.trained_epochs
        else:
            assert self.metadata_info.downstream_training_info is not None
            self.metadata_info.downstream_training_info.trained_epochs = trainer.trained_epochs
        torch.save(trainer.model, os.path.join(self.model_saving_dir, self.model_filename))
        torch.save(trainer.model.state_dict(), os.path.join(self.model_saving_dir, self.weights_filename))
        with open(os.path.join(self.model_saving_dir, self.metadata_filename), 'w', encoding='utf-8') as f:
            json.dump(self.metadata_info.model_dump(),
                      f,
                      ensure_ascii=False,
                      indent=4)

    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        pass

    def _build_structured_metadata_json(self,
                                        metadata_info: Dict[str, Any],
                                        trained_epochs: int) -> Dict[str, Any]:
        return {
            'dataset': {
                'name': metadata_info['dataset_name'],
                'pretext': {
                    'train': metadata_info.get("pretext_train"),
                    'validation': metadata_info.get("pretext_validation"),
                    'test': metadata_info.get("pretext_test"),
                },
                'downstream':
                {
                    'train': metadata_info.get("downstream_train"),
                    'validation': metadata_info.get("downstream_validation"),
                    'test': metadata_info.get("downstream_test")
                }
            },
            'learning': {
                'pretext': {
                    'algorithm': {
                        'name': metadata_info.get("pretext_algorithm"),
                        'params': metadata_info.get("pretext_algorithm_params")
                    },
                    'optimiser': {
                        'name': metadata_info.get("pretext_optimiser"),
                        'params': metadata_info.get("pretext_optimiser_params")
                    },
                    'batch_size': metadata_info.get("pretext_batch_size")
                },
                'downstream':
                {
                    'optimiser': {
                        'name': metadata_info.get("downstream_optimiser"),
                        'params': metadata_info.get("downstream_optimiser_params")
                    },
                    'batch_size':  metadata_info.get("downstream_batch_size")
                }
            },
            'trained_pretext_epochs': metadata_info.get("trained_pretext_epochs", trained_epochs),
            'trained_downstream_epochs': trained_epochs if "trained_pretext_epochs" in metadata_info.keys() else None
        }



class TimedStoppingCallback(Callback):

    def __init__(self, max_seconds: float) -> None:
        self.start_time: float = 0.0
        self.max_seconds: float = max_seconds

    def on_train_begin(self, trainer: Trainer) -> None:
        self.start_time = time()

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        if time() - self.start_time > self.max_seconds:
            trainer.stop_training = True


class EarlyStoppingCallback(Callback):

    def __init__(self, patience: int) -> None:
        self.patience: int = patience
        self.best_score: float = 999999999.9
        self.counter: int = 0

    def on_train_begin(self, trainer: Trainer) -> None:
        self.counter = 0

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        if trainer.validation_loss[-1] >= self.best_score:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}. "
                         f"Best score {self.best_score}, current: {trainer.validation_loss[-1]}")
            if self.counter >= self.patience:
                trainer.stop_training = True
        else:
            self.best_score = trainer.validation_loss[-1]
            self.counter = 0
