from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from fast_denser.misc.enums import Device
from fast_denser.misc.evaluation_metrics import EvaluationMetrics
from fast_denser.misc.utils import InputLayerId, LayerId, InvalidNetwork
from fast_denser.misc.constants import INPUT_DIMENSIONS
from fast_denser.misc.phenotype_parser import parse_phenotype, Layer, Optimiser
from fast_denser.neural_networks_torch.callbacks import EarlyStoppingCallback, \
    ModelCheckpointCallback, TimedStoppingCallback
from fast_denser.neural_networks_torch.dataset_loader import DatasetType, load_dataset
from fast_denser.neural_networks_torch.trainers import Trainer
from fast_denser.neural_networks_torch.transformers import BaseTransformer, LegacyTransformer, \
    BarlowTwinsTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

if TYPE_CHECKING:
    from fast_denser.neural_networks_torch import LearningParams
    from torchmetrics.metric import Metric

__all__ = ['create_evaluator', 'Evaluator', 'LegacyEvaluator', 'BarlowTwinsEvaluator']


logger = logging.getLogger(__name__)

def create_evaluator(dataset_name: str,
                     fitness_metric: Callable,
                     run: int,
                     learning_type: str,
                     is_gpu_run: bool,
                     train_augmentation_params: Dict[str, Any],
                     test_augmentation_params: Dict[str, Any]) -> 'BaseEvaluator':
    user_chosen_device: Device = Device.GPU if is_gpu_run is True else Device.CPU
    train_transformer: Optional[BaseTransformer]
    test_transformer: Optional[BaseTransformer]
    # Create Transformer instance
    if learning_type == 'self-supervised':
        # We say that none one them can be None because we assume there will be resize involved.
        # If we end up with different tensor sizes in train and test, the training will break
        assert train_augmentation_params is not None
        train_transformer = BarlowTwinsTransformer(train_augmentation_params)
        assert test_augmentation_params is not None
        test_transformer = LegacyTransformer(test_augmentation_params)
        return BarlowTwinsEvaluator(dataset_name,
                                    fitness_metric,
                                    run,
                                    user_chosen_device,
                                    train_transformer,
                                    test_transformer)
    else:
        train_augmentation_params = {} if train_augmentation_params is None else train_augmentation_params
        train_transformer = LegacyTransformer(train_augmentation_params)
        test_augmentation_params = {} if test_augmentation_params is None else test_augmentation_params
        test_transformer = LegacyTransformer(test_augmentation_params)
        return LegacyEvaluator(dataset_name,
                               fitness_metric,
                               run,
                               user_chosen_device,
                               train_transformer,
                               test_transformer)


class BaseEvaluator(ABC):
    def __init__(self,
                 fitness_metric: Callable,
                 seed: int,
                 user_chosen_device: Device,
                 dataset: Dict[DatasetType, Dataset]) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        self.fitness_metric: Callable[..., float] = fitness_metric
        self.seed: int = seed
        self.user_chosen_device: Device = user_chosen_device
        self.dataset = dataset


    @staticmethod
    def _adapt_model_to_device(torch_model: nn.Module, device: Device) -> None:
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)


    @staticmethod
    def _calculate_invalid_network_fitness() -> float:
        return -1.0


    def _get_data_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(self.dataset[DatasetType.EVO_TRAIN],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=8,
                                  drop_last=True,
                                  pin_memory=True)

        validation_loader = DataLoader(self.dataset[DatasetType.EVO_VALIDATION],
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=8,
                                       drop_last=True,
                                       pin_memory=True)

        test_loader = DataLoader(self.dataset[DatasetType.EVO_TEST],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=8,
                                 drop_last=True)
        return train_loader, validation_loader, test_loader
    

    def _decide_device(self) -> Device:
        if self.user_chosen_device == Device.CPU:
            return Device.CPU
        else:
            if torch.cuda.is_available() is False:
                logger.warning(f"User chose training in {self.user_chosen_device.name} but CUDA is not available. "
                               f"Defaulting training to {Device.CPU.name}")
                return Device.CPU
            else:
                return Device.GPU

    @abstractmethod
    def evaluate(self,
                 phenotype: str,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int,
                 input_size: Any=(32, 32, 3)) -> EvaluationMetrics:
        pass


    def compute_fitness(self, model: nn.Module, data_loader: DataLoader, metric: Metric, device: Device) -> float:
        model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(device.value, non_blocking=True), data[1].to(device.value, non_blocking=True)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                accuracy_test = metric(predicted, labels)
        accuracy_test = metric.compute()
        return float(accuracy_test.data)


    def testing_performance(self, model_dir: str) -> float:
        torch_model: nn.Module = torch.load(os.path.join(model_dir, "model.pth"))
        torch_model.load_state_dict(torch.load(os.path.join(model_dir, "weights.pth")))
        torch_model.eval()

        device: Device = self._decide_device()
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)

        test_loader: DataLoader = DataLoader(self.dataset[DatasetType.TEST],
                                             batch_size=64,
                                             shuffle=True)

        metric = Accuracy().to(device.value, non_blocking=True)
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device.value, non_blocking=True), data[1].to(device.value, non_blocking=True)
                outputs = torch_model(images)
                _, predicted = torch.max(outputs.data, 1)
                accuracy_test = metric(predicted, labels)
        accuracy_test = metric.compute()

        return float(accuracy_test.data)



class LegacyEvaluator(BaseEvaluator):

    def __init__(self,
                 dataset_name: str,
                 fitness_metric: Callable,
                 seed: int,
                 user_chosen_device: Device,
                 train_transformer: BaseTransformer,
                 test_transformer: BaseTransformer) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        dataset: Dict[DatasetType, Dataset] = load_dataset(dataset_name, train_transformer, test_transformer)
        super().__init__(fitness_metric, seed, user_chosen_device, dataset)


    def evaluate(self,
                 phenotype: str,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int,
                 input_size: Any=INPUT_DIMENSIONS) -> EvaluationMetrics: #pragma: no cover

        from fast_denser.neural_networks_torch.model_builder import ModelBuilder

        layers: List[Layer]
        optimiser: Optimiser
        layers_connections: Dict[LayerId, List[InputLayerId]]
        device: Device
        torch_model: nn.Module
        fitness_value: float
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        print(phenotype)
        layers, layers_connections, optimiser = parse_phenotype(phenotype)
        print(f"Reuse parents weights: {reuse_parent_weights}, Directory: {parent_dir}")
        try:
            if reuse_parent_weights is True \
                    and parent_dir is not None \
                    and len(os.listdir(parent_dir)) > 0:
                torch_model = torch.load(os.path.join(parent_dir, "model.pth"))
                torch_model.load_state_dict(torch.load(os.path.join(parent_dir, "weights.pth")))
            else:
                if reuse_parent_weights is True:
                    num_epochs = 0
                model_builder: ModelBuilder = ModelBuilder(layers, layers_connections, input_size)
                torch_model = model_builder.assemble_network(type(self))

            device = self._decide_device()
            self._adapt_model_to_device(torch_model, device)

            learning_params: LearningParams = ModelBuilder.assemble_optimiser(
                torch_model.parameters(),
                optimiser
            )
            trainable_params_count: int = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

            train_loader: DataLoader
            validation_loader: DataLoader
            test_loader: DataLoader
            train_loader, validation_loader, test_loader = self._get_data_loaders(learning_params.batch_size)

            trainer = Trainer(model=torch_model,
                            optimiser=learning_params.torch_optimiser,
                            loss_function=nn.CrossEntropyLoss(),
                            train_data_loader=train_loader,
                            validation_data_loader=validation_loader,
                            n_epochs=learning_params.epochs,
                            initial_epoch=num_epochs,
                            device=device,
                            callbacks=[EarlyStoppingCallback(patience=learning_params.early_stop),
                                        ModelCheckpointCallback(model_saving_dir),
                                        TimedStoppingCallback(max_seconds=train_time)])
            trainer.train()
            fitness_value = self.compute_fitness(model=torch_model,
                                                data_loader=test_loader,
                                                metric=Accuracy().to(device.value, non_blocking=True),
                                                device=device)    
            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                n_trainable_parameters=trainable_params_count,
                n_layers=len(layers),
                n_epochs=trainer.trained_epochs,
                validation_losses=trainer.validation_loss,
                training_time_spent=time()-start
            )
        except InvalidNetwork as e:
            logger.warning(f"Invalid model. Fitness will be computed as invalid individual. Reason: {e.message}")
            return EvaluationMetrics.default()

class BarlowTwinsEvaluator(BaseEvaluator):


    def __init__(self,
                 dataset_name: str,
                 fitness_metric: Callable,
                 seed: int,
                 user_chosen_device: Device,
                 train_transformer: BaseTransformer,
                 test_transformer: BaseTransformer) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        dataset: Dict[DatasetType, Dataset] = load_dataset(dataset_name, train_transformer, test_transformer)
        super().__init__(fitness_metric, seed, user_chosen_device, dataset)


    #def compute_fitness_temp(self, model: nn.Module, data_loader: DataLoader, metric: Metric, device: Device, batch_size: int) -> float:
    #    model.eval()
    #    total_loss = torch.zeros(1,)
    #    # since we're not training, we don't need to calculate the gradients for our outputs
    #    counter = 0
    #    for i, ((y_a, y_b), _) in enumerate(data_loader, 0):
    #        counter += 1
    #        inputs_a = y_a.to(device.value)
    #        inputs_b = y_b.to(device.value)
    #        loss = model.forward(inputs_a, inputs_b, batch_size)
    #        total_loss += loss
    #    return -float(total_loss.data)/counter

    def evaluate(self,
                 phenotype: str,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int,
                 input_size: Any=INPUT_DIMENSIONS) -> EvaluationMetrics: #pragma: no cover

        from fast_denser.neural_networks_torch.model_builder import ModelBuilder

        layers: List[Layer]
        optimiser: Optimiser
        layers_connections: Dict[LayerId, List[InputLayerId]]
        device: Device
        torch_model: Optional[nn.Module]
        fitness_value: float
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        print(phenotype)
        layers, layers_connections, optimiser = parse_phenotype(phenotype)
        print(f"Reuse parents weights: {reuse_parent_weights}, Directory: {parent_dir}")
        try:
            if reuse_parent_weights is True \
                    and parent_dir is not None \
                    and len(os.listdir(parent_dir)) > 0:
                torch_model = torch.load(os.path.join(parent_dir, "model.pth"))
                assert torch_model is not None
                torch_model.load_state_dict(torch.load(os.path.join(parent_dir, "weights.pth")))
            else:
                if reuse_parent_weights is True:
                    num_epochs = 0
                model_builder: ModelBuilder = ModelBuilder(layers, layers_connections, input_size)
                torch_model = model_builder.assemble_network(type(self))
            
                if torch_model is None:
                    logger.warning("Invalid model. Fitness will be computed as invalid individual")
                    fitness_value = self._calculate_invalid_network_fitness()
                    return EvaluationMetrics.default()

            device = self._decide_device()
            self._adapt_model_to_device(torch_model, device)

            learning_params: LearningParams = ModelBuilder.assemble_optimiser(
                torch_model.parameters(),
                optimiser
            )
            trainable_params_count: int = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

            assert learning_params.batch_size > 1, \
                "batch size should be > 1, otherwise the BatchNorm1D layer won't work"

            train_loader: DataLoader
            validation_loader: DataLoader
            test_loader: DataLoader
            train_loader, validation_loader, test_loader = self._get_data_loaders(learning_params.batch_size)

            trainer = Trainer(model=torch_model,
                            optimiser=learning_params.torch_optimiser,
                            loss_function=nn.CrossEntropyLoss(),
                            train_data_loader=train_loader,
                            validation_data_loader=validation_loader,
                            n_epochs=learning_params.epochs,
                            initial_epoch=num_epochs,
                            device=device,
                            callbacks=[ModelCheckpointCallback(model_saving_dir),
                                       TimedStoppingCallback(max_seconds=train_time)])
                                       #EarlyStoppingCallback(patience=learning_params.early_stop)])
            trainer.barlow_twins_train(learning_params.batch_size)
            fitness_value = self.compute_fitness(model=torch_model,
                                                data_loader=test_loader,
                                                metric=Accuracy().to(device.value, non_blocking=True),
                                                device=device)
            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                n_trainable_parameters=trainable_params_count,
                n_layers=len(layers),
                n_epochs=trainer.trained_epochs,
                validation_losses=trainer.validation_loss,
                training_time_spent=time()-start
            )
        except InvalidNetwork as e:
            logger.warning(f"Invalid model. Fitness will be computed as invalid individual. Reason: {e.message}")
            return EvaluationMetrics.default()
