import logging
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from evodenss.config.pydantic import get_config
from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetType
from evodenss.misc.enums import Device, AdversarialTrainingType
from evodenss.misc.utils import InvalidNetwork
from evodenss.networks.phenotype_parser import Optimiser
from evodenss.train.callbacks import Callback
from evodenss.train.lars import LARS
from evodenss.train.learning_parameters import LearningParams

import copy
if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingInfo:
    torch_model: nn.Module
    optimiser: Optimiser
    learning_params: LearningParams
    loaders_dict: dict[DatasetType, DataLoader[ConcreteDataset]]
    starting_epoch: int
    n_trainable_parameters: int
    n_layers: int


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 optimiser: optim.Optimizer,
                 train_data_loader: DataLoader[ConcreteDataset],
                 validation_data_loader: Optional[DataLoader[ConcreteDataset]],
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
        self.train_data_loader: DataLoader[ConcreteDataset] = train_data_loader
        self.validation_data_loader: Optional[DataLoader[ConcreteDataset]] = validation_data_loader
        self.n_epochs: int = n_epochs
        self.warm_up_stage: int = max(1, self.n_epochs // 3)
        self.adversarial_training: AdversarialTrainingType = AdversarialTrainingType.NONE
        self.pgd_epsilon: float = 8.0 / 255.0
        self.pgd_alpha: float = 2.0 / 255.0
        self.pgd_steps_pretext: int = 5
        self.pgd_steps_downstream: int = 20
        self.adv_loss_weight: float = 1.0
        self.initial_epoch: int = initial_epoch
        self.device: Device = device
        self.callbacks: list[Callback] = callbacks
        self.stop_training: bool = False
        self.trained_epochs: int = 0
        self.loss_values: dict[str, list[float]] = {}
        self.validation_loss: list[float] = []
        self.representation_model: Optional[nn.Module] = representation_model

        self.scheduler: Optional['LRScheduler'] = scheduler

        try:
            adversarial_config = get_config().network.learning.adversarial
            if adversarial_config is not None:
                self.adversarial_training = adversarial_config.training_type
                self.pgd_epsilon = adversarial_config.pgd_epsilon
                self.pgd_alpha = adversarial_config.pgd_alpha
                self.pgd_steps_pretext = adversarial_config.pgd_steps_pretext
                self.pgd_steps_downstream = adversarial_config.pgd_steps_downstream
                self.adv_loss_weight = adversarial_config.loss_weight
        except Exception:
            logger.debug("Config not initialized while creating Trainer; using default adversarial settings")

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

    def _pgd_attack_pretext(self,
                        original_inputs: torch.Tensor,
                        epsilon: float,
                        alpha: float,
                        steps: int) -> torch.Tensor:
        
        # PGD attack over input space [0, 1] for pretext objective RANDOM START
        x_orig = original_inputs.clone().detach()
        x_adv = x_orig.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0.0, 1.0)
        
        self.model.eval()
        with torch.no_grad():
            z_orig = self.model.forward(x_orig)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                z_adv = self.model.forward(x_adv)
                adv_loss, _, _ = self.loss_function(z_adv, z_orig)
            grad = torch.autograd.grad(adv_loss, x_adv, only_inputs=True)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + perturbation, min=0.0, max=1.0).detach()
        self.model.train()
        
        return x_adv

    def _pgd_attack_downstream(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            epsilon: float,
            alpha: float,
            steps: int
    ) -> torch.Tensor:

        x_orig = inputs.clone().detach()

        # random start inside epsilon-ball
        x_adv = x_orig.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.eval()

        for _ in range(steps):
            x_adv.requires_grad_(True)

            with torch.enable_grad():
                logits = self.model(x_adv)
                loss = self.loss_function(logits, targets)

            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

            # PGD step
            x_adv = x_adv.detach() + alpha * grad.sign()

            # project back to epsilon ball
            perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + perturbation, min=0.0, max=1.0).detach()
        
        self.model.train()
        return x_adv

    def _weight_adv(self, epoch: int) -> float:

        if self.warm_up_stage <= 0:
            return 1.0

        if epoch < self.warm_up_stage:
            return 0.0

        denom = max(1, self.n_epochs - self.warm_up_stage)
        weight = min(1.0, (epoch - self.warm_up_stage) / denom)
        logger.debug(f"Adversarial loss weight: {weight} at epoch {epoch}")
        return weight
    
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
            self.trained_epochs = 0
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

                    if self.adversarial_training == AdversarialTrainingType.DOWNSTREAM:
                        #logger.debug("adversarial training")
                        adv_inputs = self._pgd_attack_downstream(
                            inputs,
                            labels,
                            epsilon=self.pgd_epsilon,
                            alpha=self.pgd_alpha,
                            steps=self.pgd_steps_downstream
                        )
                        adv_outputs = self.model(adv_inputs)
                        adv_loss = self.loss_function(adv_outputs, labels)

                        loss = loss + self.adv_loss_weight * adv_loss 

                    total_loss += loss.detach() / n_batches_train
                    loss.backward()
                    self.optimiser.step()
                end = time.time() # noqa: F841
                #logger.info(f"[{round(end-start, 2)}s] TRAIN epoch {epoch} -- loss: {total_loss}")
                self.loss_values["train_loss"].append(round(float(total_loss.data), 3))
                logger.debug(f"Loss: {round(float(total_loss.data), 3)}")
                logger.debug("=============================================================")

                if self.validation_data_loader is not None and len(self.validation_data_loader) > 0:
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
                self.trained_epochs += 1
                self._call_on_epoch_end_callbacks()

            self._call_on_train_end_callbacks()
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
                pretext_adversarial_weight = self._weight_adv(epoch)
                for step, ((orig, y_a, y_b), _) in enumerate(self.train_data_loader, start=epoch * n_batches_train):
                    inputs_orig = orig.to(self.device.value, non_blocking=True)
                    inputs_a = y_a.to(self.device.value, non_blocking=True)
                    inputs_b = y_b.to(self.device.value, non_blocking=True)

                    if isinstance(self.optimiser, LARS):
                        self.optimiser.adjust_learning_rate(n_batches_train, self.n_epochs, step)
                    self.optimiser.zero_grad(set_to_none=True)

                    adv_orig = self._pgd_attack_pretext(
                        inputs_orig,
                        epsilon=self.pgd_epsilon,
                        alpha=self.pgd_alpha,
                        steps=self.pgd_steps_pretext
                    ) if self.adversarial_training != AdversarialTrainingType.NONE else None 
                    
                    with torch.cuda.amp.autocast():
                        z_a = self.model.forward(inputs_a)
                        z_b = self.model.forward(inputs_b)
                        z_original = self.model.forward(inputs_orig)

                        #clean loss
                        loss, diagonal_loss, off_diagonal_loss = self.loss_function(z_a, z_b)

                        #adversarial loss, if activated on the pretext task
                        if self.adversarial_training != AdversarialTrainingType.NONE:
                            z_adv = self.model.forward(adv_orig)
                            loss_adv, diagonal_loss_adv, off_diagonal_loss_adv = self.loss_function(z_adv, z_original) 
                        else:
                            loss_adv, diagonal_loss_adv, off_diagonal_loss_adv = (torch.zeros_like(loss), torch.zeros_like(diagonal_loss), torch.zeros_like(off_diagonal_loss))
                        
                        total_pretext_loss = loss + pretext_adversarial_weight*loss_adv
                        
                        total_loss += total_pretext_loss.item()
                        total_diagonal_loss += (diagonal_loss.item() + pretext_adversarial_weight*diagonal_loss_adv.item())
                        total_offdiagonal_loss += (off_diagonal_loss.item() + pretext_adversarial_weight*off_diagonal_loss_adv.item())
                    
                    scaler.scale(total_pretext_loss).backward()
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
