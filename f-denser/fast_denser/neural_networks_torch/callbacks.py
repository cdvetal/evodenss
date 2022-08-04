from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
import sys
from time import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fast_denser.neural_networks_torch.trainers import Trainer


logger = logging.getLogger(__name__)


class Callback(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def on_train_end(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer) -> None:
        pass


# Needs tweak to save individual with lowest validation error
class ModelCheckpointCallback(Callback):

    def __init__(self, model_saving_dir: str) -> None:
        self.model_saving_dir = model_saving_dir

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        torch.save(trainer.model, os.path.join(self.model_saving_dir, 'model.pth'))
        torch.save(trainer.model.state_dict(), os.path.join(self.model_saving_dir, 'weights.pth'))

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
        self.best_score: float = 999999.9
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
            logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}. "
                         f"Best score {self.best_score}, current: {trainer.validation_loss[-1]}")
            if self.counter >= self.patience:
                trainer.stop_training = True
        else:
            self.best_score = trainer.validation_loss[-1]
            self.counter = 0