from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
import signal
from time import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from evodenss.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device, FitnessMetricName
from evodenss.misc.evaluation_metrics import EvaluationMetrics
from evodenss.misc.fitness_metrics import *
from evodenss.misc.proportions import ProportionsIndexes, ProportionsFloat
from evodenss.misc.utils import InputLayerId, LayerId, InvalidNetwork
from evodenss.misc.phenotype_parser import parse_phenotype, Layer, Optimiser
from evodenss.neural_networks_torch.callbacks import EarlyStoppingCallback, \
    ModelCheckpointCallback, TimedStoppingCallback
from evodenss.neural_networks_torch.dataset_loader import DatasetType, load_dataset
from evodenss.neural_networks_torch.trainers import Trainer
from evodenss.neural_networks_torch.transformers import BaseTransformer, LegacyTransformer, \
    BarlowTwinsTransformer
from evodenss.neural_networks_torch.evolved_networks import EvaluationBarlowTwinsNetwork

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:
    from evodenss.neural_networks_torch import LearningParams

__all__ = ['create_evaluator', 'BaseEvaluator', 'BarlowTwinsEvaluator', 'LegacyEvaluator']


logger = logging.getLogger(__name__)

def setup(n_nodes: int, n_gpus_per_node: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    signal.signal(signal.SIGINT, cleanup) # type: ignore
    signal.signal(signal.SIGKILL, cleanup) # type: ignore

    # initialize the process group
    dist.init_process_group("nccl", rank=n_nodes, world_size=n_gpus_per_node)

def cleanup() -> None:
    dist.destroy_process_group()

def create_fitness_metric(metric_name: FitnessMetricName,
                          evaluator_type: type['BaseEvaluator'],
                          batch_size: Optional[int]=None,
                          loss_function: Optional[Any]=None) -> FitnessMetric:
    fitness_metric: FitnessMetric
    if metric_name.value not in FitnessMetricName.enum_values():
        raise ValueError(f"Invalid fitness metric retrieved from the config: [{metric_name}]")
    #print(evaluator_type, metric_name)
    if metric_name is FitnessMetricName.ACCURACY:
        fitness_metric = AccuracyMetric()
    elif metric_name is FitnessMetricName.LOSS: 
        if evaluator_type is LegacyEvaluator:
            assert loss_function is not None
            fitness_metric = LossMetric(loss_function)
        elif evaluator_type is BarlowTwinsEvaluator:
            assert batch_size is not None
            fitness_metric = BTLossMetric(batch_size)
    else:
        raise ValueError(f"Unexpected evaluator type: [{evaluator_type}]")
    return fitness_metric

def create_evaluator(dataset_name: str,
                     fitness_metric_name: FitnessMetricName,
                     run: int,
                     learning_type: str,
                     is_gpu_run: bool,
                     train_augmentation_params: Dict[str, Any],
                     supervised_train_augmentation_params: Dict[str, Any], # only exists in self supervised
                     test_augmentation_params: Dict[str, Any],
                     train_set_percentage: int) -> 'BaseEvaluator':
    user_chosen_device: Device = Device.GPU if is_gpu_run is True else Device.CPU
    train_transformer: Optional[BaseTransformer]
    test_transformer: Optional[BaseTransformer]
    # Create Transformer instance
    if learning_type == 'self-supervised':
        # We say that none one them can be None because we assume there will be resize involved.
        # If we end up with different tensor sizes in train and test, the training will break
        assert train_augmentation_params is not None
        train_transformer = BarlowTwinsTransformer(train_augmentation_params)
        assert supervised_train_augmentation_params is not None
        supervised_train_transformer = LegacyTransformer(supervised_train_augmentation_params)
        assert test_augmentation_params is not None
        supervised_test_transformer = LegacyTransformer(test_augmentation_params)
         # same transformations but to be used only in the test set when we want the evolution to be guided by the loss
        test_transformer = BarlowTwinsTransformer(train_augmentation_params)
        return BarlowTwinsEvaluator(dataset_name,
                                    fitness_metric_name,
                                    run,
                                    user_chosen_device,
                                    train_transformer,
                                    supervised_train_transformer,
                                    test_transformer,
                                    supervised_test_transformer,
                                    train_set_percentage)
    else:
        train_augmentation_params = {} if train_augmentation_params is None else train_augmentation_params
        train_transformer = LegacyTransformer(train_augmentation_params)
        test_augmentation_params = {} if test_augmentation_params is None else test_augmentation_params
        test_transformer = LegacyTransformer(test_augmentation_params)
        return LegacyEvaluator(dataset_name,
                               fitness_metric_name,
                               run,
                               user_chosen_device,
                               train_transformer,
                               test_transformer,
                               train_set_percentage)


class BaseEvaluator(ABC):
    def __init__(self,
                 fitness_metric_name: FitnessMetricName,
                 seed: int,
                 user_chosen_device: Device,
                 dataset: Dict[DatasetType, Subset]) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        self.fitness_metric_name: FitnessMetricName = fitness_metric_name
        self.seed: int = seed
        self.user_chosen_device: Device = user_chosen_device
        self.dataset = dataset


    @staticmethod
    def _adapt_model_to_device(torch_model: nn.Module, device: Device) -> None:
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)


    @staticmethod
    def _calculate_invalid_network_fitness(metric_name: FitnessMetricName,
                                           evaluator_type: type['BaseEvaluator']) -> Fitness:
        if metric_name.value not in FitnessMetricName.enum_values():
            raise ValueError(f"Invalid fitness metric retrieved from the config: [{metric_name}]")
        if metric_name is FitnessMetricName.ACCURACY:
            return AccuracyMetric.worst_fitness()
        elif metric_name is FitnessMetricName.LOSS:
            if evaluator_type is BarlowTwinsEvaluator:
                return BTLossMetric.worst_fitness()
            elif evaluator_type is LegacyEvaluator:
                return LossMetric.worst_fitness()
            else:
                raise ValueError(f"Unexpected evaluator type: [{evaluator_type}]")
        else:
            raise ValueError(f"Invalid fitness metric") 


    def _get_data_loaders(self,
                          dataset: Dict[DatasetType, Subset],
                          batch_size: int) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:

        g = torch.Generator()
        g.manual_seed(0)

        #during bt training if the the last batch has 1 element, training breaks at last batch norm.
        #therefore, we drop the last batch
        train_loader = DataLoader(dataset[DatasetType.EVO_TRAIN],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True,
                                  generator=g)

        validation_loader: Optional[DataLoader]
        if DatasetType.EVO_VALIDATION in dataset.keys():
            validation_loader = DataLoader(dataset[DatasetType.EVO_VALIDATION],
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           drop_last=False,
                                           pin_memory=True,
                                           generator=g)
        else:
            validation_loader = None

        
        test_loader = DataLoader(dataset[DatasetType.EVO_TEST],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=False,
                                 pin_memory=True,
                                 generator=g)

        return train_loader, validation_loader, test_loader
    

    def _decide_device(self) -> Device:
        if self.user_chosen_device == Device.CPU:
            return Device.CPU
        else:
            if torch.cuda.is_available() is False and torch.backends.mps.is_available() is False:
                logger.warning(f"User chose training in {self.user_chosen_device.name} but CUDA/MPS is not available. "
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
                 num_epochs: int) -> EvaluationMetrics:
        raise NotImplementedError()


    def testing_performance(self, model_dir: str) -> float:
        model_filename: str
        weights_filename: str
        if type(self) is BarlowTwinsEvaluator:
            model_filename = f"complete_{MODEL_FILENAME}"
            weights_filename = f"complete_{WEIGHTS_FILENAME}"
        elif type(self) is LegacyEvaluator:
            model_filename = MODEL_FILENAME
            weights_filename = WEIGHTS_FILENAME
        else:
            raise ValueError("Unexpected evaluator")
        
        torch_model: nn.Module = torch.load(os.path.join(model_dir, model_filename))
        torch_model.load_state_dict(torch.load(os.path.join(model_dir, weights_filename)))
        torch_model.eval()

        device: Device = self._decide_device()
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)

        test_set = self.dataset[DatasetType.TEST]
        assert test_set is not None
        test_loader: DataLoader = DataLoader(test_set, batch_size=64, shuffle=True)
        metric = AccuracyMetric(batch_size=64)
        return metric.compute_metric(torch_model, test_loader, device)



class LegacyEvaluator(BaseEvaluator):

    def __init__(self,
                 dataset_name: str,
                 fitness_metric_name: FitnessMetricName,
                 seed: int,
                 user_chosen_device: Device,
                 train_transformer: BaseTransformer,
                 test_transformer: BaseTransformer,
                 train_set_percentage: int) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        self.dataset_name: str = dataset_name
        dataset: Dict[DatasetType, Subset] = load_dataset(dataset_name,
                                                          train_transformer,
                                                          test_transformer,
                                                          enable_stratify=True,
                                                          proportions=ProportionsFloat({
                                                               DatasetType.EVO_TRAIN: 0.7,
                                                               DatasetType.EVO_VALIDATION: 0.2,
                                                               DatasetType.EVO_TEST: 0.1
                                                          }),
                                                          downstream_train_percentage=train_set_percentage)
        super().__init__(fitness_metric_name, seed, user_chosen_device, dataset)


    def evaluate(self,
                 phenotype: str,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics: #pragma: no cover

        from evodenss.neural_networks_torch.model_builder import ModelBuilder

        layers: List[Layer]
        optimiser: Optimiser
        layers_connections: Dict[LayerId, List[InputLayerId]]
        device: Device = self._decide_device()
        torch_model: nn.Module
        fitness_value: Fitness
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        logger.info(phenotype)
        layers, layers_connections, optimiser = parse_phenotype(phenotype)
        #print(f"Reuse parents weights: {reuse_parent_weights}, Directory: {parent_dir}")
        try:
            input_size: Tuple[int, int, int] = DATASETS_INFO[self.dataset_name]["expected_input_dimensions"]
            model_builder: ModelBuilder = ModelBuilder(layers, layers_connections, device, input_size)
            torch_model = model_builder.assemble_network(type(self))
            if reuse_parent_weights is True \
                    and parent_dir is not None \
                    and len(os.listdir(parent_dir)) > 0:
                torch_model.load_state_dict(torch.load(os.path.join(parent_dir, WEIGHTS_FILENAME)))
            else:
                if reuse_parent_weights is True:
                    num_epochs = 0
                

            device = self._decide_device()
            self._adapt_model_to_device(torch_model, device)

            learning_params: LearningParams = ModelBuilder.assemble_optimiser(
                torch_model.parameters(),
                optimiser
            )
            trainable_params_count: int = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

            train_loader: DataLoader
            validation_loader: Optional[DataLoader]
            test_loader: DataLoader
            train_loader, validation_loader, test_loader = \
                self._get_data_loaders(self.dataset, learning_params.batch_size)

            assert validation_loader is not None
            #assert learning_params.early_stop is not None

            loss_function = nn.CrossEntropyLoss()
            trainer = Trainer(model=torch_model,
                              optimiser=learning_params.torch_optimiser,
                              loss_function=loss_function,
                              train_data_loader=train_loader,
                              validation_data_loader=validation_loader,
                              n_epochs=learning_params.epochs,
                              initial_epoch=num_epochs,
                              device=device,
                              callbacks=[ModelCheckpointCallback(model_saving_dir),
                                         TimedStoppingCallback(max_seconds=train_time),
                                         EarlyStoppingCallback(patience=learning_params.early_stop)])
            trainer.train()
            fitness_metric: FitnessMetric = create_fitness_metric(self.fitness_metric_name,
                                                                  type(self),
                                                                  loss_function=loss_function)
            fitness_value = Fitness(fitness_metric.compute_metric(torch_model, test_loader, device),
                                    type(fitness_metric))
            accuracy: Optional[float]
            if fitness_metric is AccuracyMetric:
                accuracy = None
            else:
                accuracy = AccuracyMetric().compute_metric(torch_model, test_loader, device)
            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                accuracy=accuracy,
                n_trainable_parameters=trainable_params_count,
                n_layers=len(layers),
                n_epochs=trainer.trained_epochs,
                losses=trainer.loss_values,
                training_time_spent=time()-start,
                total_epochs_trained=num_epochs+trainer.trained_epochs,
                max_epochs_reached=num_epochs+trainer.trained_epochs >= learning_params.epochs
            )
        except InvalidNetwork as e:
            logger.warning(f"Invalid model. Fitness will be computed as invalid individual. Reason: {e.message}")
            fitness_value = self._calculate_invalid_network_fitness(self.fitness_metric_name, type(self))
            return EvaluationMetrics.default(fitness_value)



class BarlowTwinsEvaluator(BaseEvaluator):

    def __init__(self,
                 dataset_name: str,
                 fitness_metric_name: FitnessMetricName,
                 seed: int,
                 user_chosen_device: Device,
                 train_transformer: BaseTransformer,
                 supervised_train_transformer: BaseTransformer,
                 test_transformer: BaseTransformer,
                 supervised_test_transformer: BaseTransformer,
                 train_set_percentage: int) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        self.dataset_name: str = dataset_name

        # Pretext task uses EVO TRAIN from Pairwise Dataset
        # Downstream task uses EVO TRAIN from dataset
        # and measures accuracy using EVO TEST from dataset
        # We need to ensure that test set from the downstream task
        # does not overlap with the train set from the pretext task
        self.pairwise_dataset: Dict[DatasetType, Subset] = load_dataset(
            dataset_name,
            train_transformer,
            test_transformer,
            enable_stratify=True,
            proportions=ProportionsFloat({
                DatasetType.EVO_TRAIN: 0.7,
                DatasetType.EVO_TEST: 0.3
            }),
            downstream_train_percentage=None # Pairwise dataset is only used for the pretext task
        )
        
        dataset: Dict[DatasetType, Subset] = load_dataset(
            dataset_name,
            supervised_train_transformer,
            supervised_test_transformer,
            enable_stratify=True,
            proportions=ProportionsIndexes({
                DatasetType.EVO_TRAIN: list(self.pairwise_dataset[DatasetType.EVO_TRAIN].indices),
                DatasetType.EVO_TEST: list(self.pairwise_dataset[DatasetType.EVO_TEST].indices)
            }),
            downstream_train_percentage=train_set_percentage
        )
        super().__init__(fitness_metric_name, seed, user_chosen_device, dataset)


    def evaluate(self,
                 phenotype: str,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics: #pragma: no cover

        from evodenss.neural_networks_torch.model_builder import ModelBuilder

        layers: List[Layer]
        optimiser: Optimiser
        layers_connections: Dict[LayerId, List[InputLayerId]]
        device: Device = self._decide_device()
        torch_model: Optional[nn.Module]
        fitness_value: Fitness
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        logger.info(phenotype)
        layers, layers_connections, optimiser = parse_phenotype(phenotype)
        try:
            input_size = DATASETS_INFO[self.dataset_name]["expected_input_dimensions"]
            model_builder: ModelBuilder = ModelBuilder(layers, layers_connections, device, input_size)
            torch_model = model_builder.assemble_network(type(self))
            if reuse_parent_weights is True \
                    and parent_dir is not None \
                    and len(os.listdir(parent_dir)) > 0:
                torch_model.load_state_dict(torch.load(os.path.join(parent_dir, WEIGHTS_FILENAME)))
            else:
                if reuse_parent_weights is True:
                    num_epochs = 0

            torch_model.to(device.value)
            logger.debug(torch_model)

            learning_params: LearningParams = ModelBuilder.assemble_optimiser(
                torch_model.parameters(),
                optimiser
            )
            trainable_params_count: int = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

            assert learning_params.batch_size > 1, \
                "batch size should be > 1, otherwise the BatchNorm1D layer won't work"

            train_loader: DataLoader
            pairwise_test_loader: DataLoader
            normal_test_loader: DataLoader
            train_loader, _, pairwise_test_loader = \
                self._get_data_loaders(self.pairwise_dataset, learning_params.batch_size)

            trainer = Trainer(model=torch_model,
                              optimiser=learning_params.torch_optimiser,
                              loss_function=nn.CrossEntropyLoss(),
                              train_data_loader=train_loader,
                              validation_data_loader=None,
                              n_epochs=learning_params.epochs,
                              initial_epoch=num_epochs,
                              device=device,
                              callbacks=[ModelCheckpointCallback(model_saving_dir),
                                         TimedStoppingCallback(max_seconds=train_time)])
                                         #EarlyStoppingCallback(patience=learning_params.early_stop)])
            
            trainer.barlow_twins_train(learning_params.batch_size)
            fitness_metric: FitnessMetric = create_fitness_metric(self.fitness_metric_name,
                                                                  type(self),
                                                                  batch_size=learning_params.batch_size)

            accuracy: Optional[float]
            # Second train for the last layer
            train_loader, validation_loader, normal_test_loader = \
                self._get_data_loaders(self.dataset, learning_params.batch_size)
            n_classes: int = DATASETS_INFO[self.dataset_name]["classes"]
            
            complete_model: EvaluationBarlowTwinsNetwork = EvaluationBarlowTwinsNetwork(torch_model, n_classes, device)
            complete_model.to(device.value, non_blocking=True)
            #print(list(map(lambda x: x[0], complete_model.named_parameters())))
            relevant_index: int = complete_model.relevant_index
            params_to_tune = [param for name, param in complete_model.named_parameters()
                              if name in {f'final_layer.{relevant_index}.weight', f'final_layer.{relevant_index}.bias'}]
            
            last_layer_trainer = Trainer(model=complete_model,
                                         optimiser=torch.optim.Adam(params_to_tune, lr=1e-3, weight_decay=1e-6),
                                         loss_function=nn.CrossEntropyLoss(),
                                         train_data_loader=train_loader,
                                         validation_data_loader=validation_loader,
                                         n_epochs=30,
                                         initial_epoch=0,
                                         device=device,
                                         callbacks=[ModelCheckpointCallback(model_saving_dir,
                                                                            model_filename=f"complete_{MODEL_FILENAME}",
                                                                            weights_filename=f"complete_{WEIGHTS_FILENAME}")])
            last_layer_trainer.train()
            if type(fitness_metric) is AccuracyMetric:
                fitness_value = Fitness(
                    fitness_metric.compute_metric(complete_model, normal_test_loader, device),
                    type(fitness_metric)
                )
                accuracy = None
            else:
                try:
                    fitness_value = Fitness(
                        fitness_metric.compute_metric(torch_model, pairwise_test_loader, device),
                        type(fitness_metric)
                    )
                except ValueError as e:
                    logger.warning(f"Error computing fitness, individual will be considered invalid. Reason: {traceback.format_exc()}")
                    fitness_value = self._calculate_invalid_network_fitness(self.fitness_metric_name, type(self))
                accuracy = AccuracyMetric().compute_metric(complete_model, normal_test_loader, device)

            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                accuracy=accuracy,
                n_trainable_parameters=trainable_params_count,
                n_layers=len(layers),
                n_epochs=trainer.trained_epochs,
                losses=trainer.loss_values,
                training_time_spent=time()-start,
                total_epochs_trained=num_epochs+trainer.trained_epochs,
                max_epochs_reached=num_epochs+trainer.trained_epochs >= learning_params.epochs
            )
        except InvalidNetwork as e:
            logger.warning(f"Invalid model, error during evaluation. Fitness will be computed as invalid individual. Reason: {traceback.format_exc()}")
            fitness_value = self._calculate_invalid_network_fitness(self.fitness_metric_name, type(self))
            return EvaluationMetrics.default(fitness_value)
