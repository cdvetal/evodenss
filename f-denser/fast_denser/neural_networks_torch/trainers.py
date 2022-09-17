import logging
from typing import Any, Dict, List

import torch

from fast_denser.misc.enums import Device
from fast_denser.misc.utils import InvalidNetwork
from fast_denser.neural_networks_torch.callbacks import Callback

from torch import nn, optim
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 optimiser: optim.Optimizer,
                 train_data_loader: DataLoader,
                 validation_data_loader: DataLoader,
                 loss_function: Any,
                 n_epochs: int,
                 initial_epoch: int,
                 device: Device,
                 callbacks: List[Callback]=[]) -> None:
        self.model: nn.Module = model
        self.optimiser: optim.Optimizer = optimiser
        self.loss_function: Any = loss_function
        self.train_data_loader: DataLoader = train_data_loader
        self.validation_data_loader: DataLoader = validation_data_loader
        self.n_epochs: int = n_epochs
        self.initial_epoch: int = initial_epoch
        self.device: Device = device
        self.callbacks: List[Callback] = callbacks
        self.stop_training: bool = False
        self.trained_epochs: int = 0
        self.loss_values: Dict[str, List[float]] = {}
        self.validation_loss: List[float] = []

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
        self.loss_values = { "train_loss": [], "val_loss": [] }
        try:
            epoch: int = self.initial_epoch
            n_batches_train: int = len(self.train_data_loader)
            n_batches_validation: int = len(self.validation_data_loader)
            self.model.train()

            self._call_on_train_begin_callbacks()

            while epoch < self.n_epochs and self.stop_training is False:

                self._call_on_epoch_begin_callbacks()
                total_loss = torch.zeros(size=(1,), device=self.device.value)
                for i, data in enumerate(self.train_data_loader, 0):
                    inputs, labels = data[0].to(self.device.value, non_blocking=True), data[1].to(self.device.value, non_blocking=True)
                    # zero the parameter gradients
                    self.optimiser.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    total_loss += loss/n_batches_train
                    loss.backward()
                    self.optimiser.step()
                self.loss_values["train_loss"].append(float(total_loss.data))
                
                with torch.no_grad():
                    self.model.eval()
                    total_loss = torch.zeros(size=(1,), device=self.device.value)
                    for i, data in enumerate(self.validation_data_loader, 0):
                        inputs, labels = data[0].to(self.device.value, non_blocking=True), data[1].to(self.device.value, non_blocking=True)
                        outputs = self.model(inputs)
                        total_loss += self.loss_function(outputs, labels)/n_batches_validation
                    self.loss_values["val_loss"].append(float(total_loss.data))
                    self.validation_loss.append(float(total_loss.data)) # Used for early stopping criteria
                self.model.train()
                epoch += 1

                self._call_on_epoch_end_callbacks()

            self._call_on_train_end_callbacks()
            self.trained_epochs = epoch
        except RuntimeError as e:
            print(e)
            import traceback
            print(traceback.format_exc())
            raise InvalidNetwork(str(e))

    def barlow_twins_train(self, batch_size: int) -> None:
        self.loss_values = {
            "train_loss_diagonal": [],
            "train_loss_offdiagonal": [],
            "train_loss_complete": [],
            "val_loss_diagonal": [],
            "val_loss_offdiagonal": [],
            "val_loss_complete": []
        }
        try:
            epoch: int = self.initial_epoch
            n_batches_train: int = len(self.train_data_loader)
            n_batches_validation: int = len(self.validation_data_loader)
            self.model.train()

            self._call_on_train_begin_callbacks()
            scaler = torch.cuda.amp.GradScaler()
            # torch.autograd.set_detect_anomaly(True)
            print(epoch, self.n_epochs, self.stop_training)
            while epoch < self.n_epochs and self.stop_training is False:
                total_loss = torch.zeros(size=(1,), device=self.device.value)
                total_diagonal_loss = torch.zeros(size=(1,), device=self.device.value)
                total_offdiagonal_loss = torch.zeros(size=(1,), device=self.device.value)


                self._call_on_epoch_begin_callbacks()
                for i, ((y_a, y_b), _) in enumerate(self.train_data_loader, 0):
                    inputs_a = y_a.to(self.device.value, non_blocking=True)
                    inputs_b = y_b.to(self.device.value, non_blocking=True)
                
                    # TODO: Add adjust learning rate function from original code
                    # zero the parameter gradients
                    self.optimiser.zero_grad()
                    with torch.cuda.amp.autocast():
                        all_loss_components = self.model.forward(inputs_a, inputs_b, batch_size)
                 
                    total_loss += all_loss_components[-1]/n_batches_train
                    total_diagonal_loss += all_loss_components[0]/n_batches_train
                    total_offdiagonal_loss += all_loss_components[1]/n_batches_train

                    scaler.scale(all_loss_components[-1]).backward()
                    scaler.step(self.optimiser)
                    scaler.update()

                #print(f"train Epoch {epoch}: {total_diagonal_loss.data}, " \
                #      f"{total_offdiagonal_loss.data}, " 
                #      f"{total_loss.data}")

                self.loss_values["train_loss_diagonal"].append(float(total_diagonal_loss.data))
                self.loss_values["train_loss_offdiagonal"].append(float(total_offdiagonal_loss.data))
                self.loss_values["train_loss_complete"].append(float(total_loss.data))

                # print(f"Epoch {epoch} loss: {total_loss}")
                
                self.model.eval()
                with torch.no_grad():
                    total_loss = torch.zeros(size=(1,), device=self.device.value)
                    total_diagonal_loss = torch.zeros(size=(1,), device=self.device.value)
                    total_offdiagonal_loss = torch.zeros(size=(1,), device=self.device.value)
                    for _, ((y_a, y_b), _) in enumerate(self.validation_data_loader, 0):
                        inputs_a = y_a.to(self.device.value, non_blocking=True)
                        inputs_b = y_b.to(self.device.value, non_blocking=True)
                        with torch.cuda.amp.autocast():
                            all_loss_components = self.model.forward(inputs_a, inputs_b, batch_size)
                            complete_loss = all_loss_components[-1]
                        total_loss += complete_loss/n_batches_validation
                        total_diagonal_loss += all_loss_components[0]/n_batches_validation
                        total_offdiagonal_loss += all_loss_components[1]/n_batches_validation

                    #print(f"validation Epoch {epoch}: {total_diagonal_loss.data}, " \
                    #      f"{total_offdiagonal_loss.data}, " 
                    #      f"{total_loss.data}")

                    self.validation_loss.append(float(total_loss.data)) # Used for early stopping criteria
                    self.loss_values["val_loss_diagonal"].append(float(total_diagonal_loss.data))
                    self.loss_values["val_loss_offdiagonal"].append(float(total_offdiagonal_loss.data))
                    self.loss_values["val_loss_complete"].append(float(total_loss.data))
                self.model.train()
                epoch += 1

                self._call_on_epoch_end_callbacks()
            self._call_on_train_end_callbacks()
            self.trained_epochs = epoch - self.initial_epoch
        except RuntimeError as e:
            raise InvalidNetwork(str(e)) from e
