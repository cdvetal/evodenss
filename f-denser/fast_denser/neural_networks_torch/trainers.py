import logging
from typing import Any, List

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
        try:
            epoch: int = self.initial_epoch
            self.model.train()

            self._call_on_train_begin_callbacks()

            while epoch < self.n_epochs and self.stop_training is False:

                self._call_on_epoch_begin_callbacks()

                for i, data in enumerate(self.train_data_loader, 0):
                    inputs, labels = data[0].to(self.device.value, non_blocking=True), data[1].to(self.device.value, non_blocking=True)
                    # zero the parameter gradients
                    self.optimiser.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimiser.step()
                epoch += 1

                self.model.eval()
                with torch.no_grad():
                    total_loss = torch.zeros(size=(1,), device=self.device.value)
                    for i, data in enumerate(self.validation_data_loader, 0):
                        inputs, labels = data[0].to(self.device.value, non_blocking=True), data[1].to(self.device.value, non_blocking=True)
                        outputs = self.model(inputs)
                        total_loss += self.loss_function(outputs, labels)
                    self.validation_loss.append(float(total_loss.data))
                self.model.train()

                self._call_on_epoch_end_callbacks()

            self._call_on_train_end_callbacks()
            self.trained_epochs = epoch
        except RuntimeError as e:
            raise InvalidNetwork(str(e))

    def barlow_twins_train(self, batch_size: int) -> None:
        try:
            epoch: int = self.initial_epoch
            self.model.train()

            self._call_on_train_begin_callbacks()
            losses = []
            scaler = torch.cuda.amp.GradScaler()
            while epoch < self.n_epochs and self.stop_training is False:
                total_loss = torch.zeros(size=(1,), device=self.device.value)
                self._call_on_epoch_begin_callbacks()
                for i, ((y_a, y_b), _) in enumerate(self.train_data_loader, 0):
                    inputs_a = y_a.to(self.device.value, non_blocking=True)
                    inputs_b = y_b.to(self.device.value, non_blocking=True)
                    # TODO: Add adjust learning rate function from original code
                    # zero the parameter gradients
                    self.optimiser.zero_grad()
                    with torch.cuda.amp.autocast():
                        loss = self.model.forward(inputs_a, inputs_b, batch_size)
                    total_loss += loss
                    #loss.backward()
                    #self.optimiser.step()
                    loss = scaler.scale(loss).backward()
                    scaler.step(self.optimiser)
                    scaler.update()
                losses.append(total_loss)
                    
                # print(f"Epoch {epoch} loss: {total_loss}")
                epoch += 1

                '''
                self.model.eval()
                with torch.no_grad():
                    total_loss = torch.zeros(size=(1,), device=self.device.value)
                    for i, ((y_a, y_b), _) in enumerate(self.validation_data_loader, 0):
                        inputs_a = y_a.to(self.device.value, non_blocking=True)
                        inputs_b = y_b.to(self.device.value, non_blocking=True)
                        loss = self.model.forward(inputs_a, inputs_b, batch_size)
                        total_loss += loss
                    self.validation_loss.append(float(total_loss.data))
                self.model.train()
                '''
                self._call_on_epoch_end_callbacks()
            print(losses)
            self._call_on_train_end_callbacks()
            self.trained_epochs = epoch
        except RuntimeError as e:
            raise InvalidNetwork(str(e))
