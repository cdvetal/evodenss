from dataclasses import dataclass
import logging
import time
import traceback
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from evodenss.dataset.dataset_loader import DatasetType
from evodenss.misc.enums import Device
from evodenss.misc.utils import InvalidNetwork
from evodenss.networks.phenotype_parser import Optimiser
from evodenss.train.callbacks import Callback
from evodenss.train.lars import LARS
from evodenss.train.learning_parameters import LearningParams


if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingInfo:
    torch_model: nn.Module
    optimiser: Optimiser
    learning_params: LearningParams
    loaders_dict: dict[DatasetType, DataLoader]
    starting_epoch: int
    n_trainable_parameters: int
    n_layers: int


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 optimiser: optim.Optimizer,
                 train_data_loader: DataLoader,
                 validation_data_loader: Optional[DataLoader],
                 loss_function: Any,
                 n_epochs: int,
                 initial_epoch: int,
                 device: Device,
                 callbacks: list[Callback]=[],
                 representation_model: Optional[nn.Module]=None,
                 scheduler: Optional['LRScheduler']=None) -> None:
        self.model: nn.Module = model
        self.optimiser: optim.Optimizer = optimiser
        self.loss_function: Any = loss_function
        self.train_data_loader: DataLoader = train_data_loader
        self.validation_data_loader: Optional[DataLoader] = validation_data_loader
        self.n_epochs: int = n_epochs
        self.initial_epoch: int = initial_epoch
        self.device: Device = device
        self.callbacks: list[Callback] = callbacks
        self.stop_training: bool = False
        self.trained_epochs: int = 0
        self.loss_values: dict[str, list[float]] = {}
        self.validation_loss: list[float] = []
        self.representation_model: Optional[nn.Module] = representation_model

        self.scheduler: Optional['LRScheduler'] = scheduler

        # cuda stuff
        torch.cuda.empty_cache()
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = True

    def _call_on_train_begin_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_train_begin(self)

    def _call_on_train_end_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_train_end(self)

    def _call_on_epoch_begin_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_epoch_begin(self)

    def _call_on_epoch_end_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_epoch_end(self)

    def train(self) -> None:
        #assert self.validation_data_loader is not None

        logger.debug("Initiating supervised training")
        self.loss_values = { "train_loss": [], "val_loss": [] }
        try:
            epoch: int = self.initial_epoch
            n_batches_train: int = len(self.train_data_loader)
            n_batches_validation: Optional[int]
            if self.validation_data_loader is not None:
                n_batches_validation = len(self.validation_data_loader)
            self.model.train()
            self._call_on_train_begin_callbacks()

            while epoch < self.n_epochs and self.stop_training is False:
                logger.debug(f"Starting Downstream Epoch {epoch}")
                self._call_on_epoch_begin_callbacks()
                start = time.time() # noqa: F841
                total_loss = torch.zeros(size=(1,), device=self.device.value)
                for i, data in enumerate(self.train_data_loader, 0):
                    inputs, labels = data[0].to(self.device.value, non_blocking=True), \
                        data[1].to(self.device.value, non_blocking=True)
                    if isinstance(self.optimiser, LARS):
                        self.optimiser.adjust_learning_rate(n_batches_train, self.n_epochs, i)
                    # zero the parameter gradients
                    self.optimiser.zero_grad()
                    if self.representation_model is not None:
                        inputs = self.representation_model(inputs)
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    total_loss += loss/n_batches_train
                    loss.backward()
                    self.optimiser.step()
                end = time.time() # noqa: F841
                #logger.info(f"[{round(end-start, 2)}s] TRAIN epoch {epoch} -- loss: {total_loss}")
                self.loss_values["train_loss"].append(round(float(total_loss.data), 3))
                logger.debug(f"Loss: {round(float(total_loss.data), 3)}")
                logger.debug("=============================================================")

                if self.validation_data_loader is not None:
                    with torch.no_grad():
                        self.model.eval()
                        total_loss = torch.zeros(size=(1,), device=self.device.value)
                        for i, data in enumerate(self.validation_data_loader, 0):
                            inputs, labels = data[0].to(self.device.value, non_blocking=True), \
                                data[1].to(self.device.value, non_blocking=True)
                            if self.representation_model is not None:
                                inputs = self.representation_model(inputs)
                            outputs = self.model(inputs)
                            total_loss += self.loss_function(outputs, labels)/n_batches_validation
                        self.loss_values["val_loss"].append(round(float(total_loss.data), 3))
                        self.validation_loss.append(float(total_loss.data)) # Used for early stopping criteria
                    self.model.train()
                    end = time.time() # noqa: F841
                    #logger.info(f"[{round(end-start, 2)}s] VALIDATION epoch {epoch} -- loss: {total_loss}")
                if self.scheduler is not None:
                    self.scheduler.step()
                epoch += 1
                self._call_on_epoch_end_callbacks()

            self._call_on_train_end_callbacks()
            self.trained_epochs = epoch - self.initial_epoch
        except RuntimeError as e:
            logger.warning(traceback.format_exc())
            raise InvalidNetwork(str(e)) from e

    def barlow_twins_train(self) -> None:
        assert self.validation_data_loader is None
        self.loss_values = {
            "train_loss_diagonal": [],
            "train_loss_offdiagonal": [],
            "train_loss_complete": []
            #"val_loss_diagonal": [],
            #"val_loss_offdiagonal": [],
            #"val_loss_complete": []
        }

        try:
            epoch: int = self.initial_epoch
            n_batches_train: int = len(self.train_data_loader)
            #n_batches_validation: int = len(self.validation_data_loader)

            self._call_on_train_begin_callbacks()
            scaler = torch.cuda.amp.GradScaler()
            while epoch < self.n_epochs and self.stop_training is False:
                self.model.train()
                start = time.time() # noqa: F841
                total_loss = 0
                total_diagonal_loss = 0
                total_offdiagonal_loss = 0

                self._call_on_epoch_begin_callbacks()
                for step, ((y_a, y_b), _) in enumerate(self.train_data_loader, start=epoch * n_batches_train):
                    inputs_a = y_a.to(self.device.value, non_blocking=True)
                    inputs_b = y_b.to(self.device.value, non_blocking=True)

                    if isinstance(self.optimiser, LARS):
                        self.optimiser.adjust_learning_rate(n_batches_train, self.n_epochs, step)
                    self.optimiser.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast():
                        z_a = self.model.forward(inputs_a)
                        z_b = self.model.forward(inputs_b)
                        loss, diagonal_loss, off_diagonal_loss = self.loss_function(z_a, z_b)
                        total_loss += loss.item()
                        total_diagonal_loss += diagonal_loss.item()
                        total_offdiagonal_loss += off_diagonal_loss.item()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimiser)
                    scaler.update()

                end = time.time() # noqa: F841
                #logger.info(f"[{round(end-start, 2)}s] TRAIN epoch {epoch} -- loss: {total_loss/n_batches_train}")
                self.loss_values["train_loss_diagonal"].append(round(total_diagonal_loss/n_batches_train, 3))
                self.loss_values["train_loss_offdiagonal"].append(round(total_offdiagonal_loss/n_batches_train, 3))
                self.loss_values["train_loss_complete"].append(round(total_loss/n_batches_train, 3))
                logger.debug(f"Epoch: {epoch} Loss: {round(total_loss/n_batches_train, 3)}")
                logger.debug("----------------")
                #self.model.eval()
                #with torch.no_grad():
                #    start = time.time()
                #    total_loss = 0
                #    total_diagonal_loss = 0
                #    total_offdiagonal_loss = 0
                #    for _, ((y_a, y_b), _) in enumerate(self.validation_data_loader, 0):
                #        inputs_a = y_a.to(self.device.value, non_blocking=True)
                #        inputs_b = y_b.to(self.device.value, non_blocking=True)
                #        with torch.cuda.amp.autocast():
                #            all_loss_components = self.model.forward(inputs_a, inputs_b, batch_size)
                #            total_loss += all_loss_components[-1].item()
                #            total_diagonal_loss += all_loss_components[0].item()
                #            total_offdiagonal_loss += all_loss_components[1].item()
                #
                #    end = time.time()
                #    logger.debug(f"[{round(end-start, 2)}s] VALIDATION epoch {epoch} -- "
                #                 f"loss: {total_loss/n_batches_validation}")
                #    self.validation_loss.append(total_loss/n_batches_validation) # Used for early stopping criteria
                #    self.loss_values["val_loss_diagonal"].append(round(total_diagonal_loss/n_batches_validation, 3))
                #    self.loss_values["val_loss_offdiagonal"].append(
                #        round(total_offdiagonal_loss/n_batches_validation, 3)
                #    )
                #    self.loss_values["val_loss_complete"].append(round(total_loss/n_batches_validation, 3))
                #logger.debug("=============================================================")
                epoch += 1
                self._call_on_epoch_end_callbacks()
            self._call_on_train_end_callbacks()
            self.trained_epochs = epoch - self.initial_epoch
        except RuntimeError as e:
            logger.warning(traceback.format_exc())
            raise InvalidNetwork(str(e)) from e
