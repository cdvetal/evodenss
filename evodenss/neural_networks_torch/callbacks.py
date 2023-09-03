from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from time import time
from typing import TYPE_CHECKING

from evodenss.misc.constants import MODEL_FILENAME, WEIGHTS_FILENAME

import torch

if TYPE_CHECKING:
    from evodenss.neural_networks_torch.trainers import Trainer


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
                 model_filename: str=MODEL_FILENAME,
                 weights_filename: str=WEIGHTS_FILENAME) -> None:
        self.model_saving_dir: str = model_saving_dir
        self.model_filename: str = model_filename
        self.weights_filename: str = weights_filename

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        torch.save(trainer.model, os.path.join(self.model_saving_dir, self.model_filename))
        torch.save(trainer.model.state_dict(), os.path.join(self.model_saving_dir, self.weights_filename))
        
    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        pass


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