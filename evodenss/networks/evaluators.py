from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import logging
import os
from time import time
import traceback
from typing import Any, Optional, TYPE_CHECKING, cast

import torch
from torch import Generator, nn, Size
from torch.utils.data import DataLoader, Subset

from evodenss.config.pydantic import LearningType, PriorRepresentationsConfig, get_config, get_fitness_extra_params
from evodenss.misc.constants import DATASETS_INFO, DEFAULT_SEED, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device, DownstreamMode, FitnessMetricName
from evodenss.metrics.evaluation_metrics import EvaluationMetrics
from evodenss.metrics.fitness_metrics import AccuracyMetric, DownstreamAccuracyMetric, Fitness, \
    FitnessMetric, KNNAccuracyMetric
from evodenss.misc.metadata_info import MetadataInfo
from evodenss.misc.utils import InvalidNetwork
from evodenss.networks.evolved_networks import BarlowTwinsNetwork
from evodenss.networks.phenotype_parser import Pretext, parse_phenotype, Optimiser
from evodenss.networks.transformers import LegacyTransformer
from evodenss.train.callbacks import Callback, EarlyStoppingCallback, \
    ModelCheckpointCallback, TimedStoppingCallback
from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetProcessor, DatasetType
from evodenss.train.losses import BarlowTwinsLoss
from evodenss.train.trainers import Trainer

if TYPE_CHECKING:
    from evodenss.networks.phenotype_parser import ParsedNetwork
    from evodenss.train.learning_parameters import LearningParams
    from evodenss.networks.model_builder import ModelBuilder

__all__ = ['BaseEvaluator', 'BarlowTwinsEvaluator', 'LegacyEvaluator']


logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    def __init__(self,
                 dataset_name: str,
                 user_chosen_device: Device) -> None:
        self.dataset_name: str = dataset_name
        self.user_chosen_device: Device = user_chosen_device

    @staticmethod
    def create_evaluator(dataset_name: str,
                         is_gpu_run: bool) -> 'BaseEvaluator':
        user_chosen_device: Device = Device.GPU if is_gpu_run is True else Device.CPU
        learning_type: LearningType = get_config().network.learning.learning_type
        if learning_type == LearningType.self_supervised:
            return BarlowTwinsEvaluator(dataset_name, user_chosen_device)
        else:
            prior_representation_config: Optional[PriorRepresentationsConfig] = \
                get_config().network.prior_representations
            representation_model: Optional[nn.Module]
            representation_model_training_mode: Optional[DownstreamMode]
            representation_model, representation_model_training_mode = \
                BaseEvaluator.load_representation_model(prior_representation_config)
            return LegacyEvaluator(dataset_name,
                                   user_chosen_device,
                                   representation_model,
                                   representation_model_training_mode)

    @staticmethod
    def load_representation_model(
            prior_representation_config: Optional[PriorRepresentationsConfig]
            ) -> tuple[Optional[nn.Module], Optional[DownstreamMode]]:
        if prior_representation_config is None:
            return None, None
        model: nn.Module = torch.load(prior_representation_config.representations_model_path)
        model.load_state_dict(torch.load(prior_representation_config.representations_weights_path))
        return model, prior_representation_config.training_mode


    @staticmethod
    def _adapt_model_to_device(torch_model: nn.Module, device: Device) -> None:
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)
        torch.compile(torch_model, mode="reduce-overhead")


    @staticmethod
    def _calculate_invalid_network_fitness(metric_name: FitnessMetricName) -> Fitness:
        if metric_name.value not in FitnessMetricName.enum_values():
            raise ValueError(f"Invalid fitness metric retrieved from the config: [{metric_name}]")
        if metric_name is FitnessMetricName.ACCURACY:
            return AccuracyMetric.worst_fitness()
        elif metric_name is FitnessMetricName.DOWNSTREAM_ACCURACY:
            return DownstreamAccuracyMetric.worst_fitness()
        elif metric_name is FitnessMetricName.KNN_ACCURACY:
            return KNNAccuracyMetric.worst_fitness()
        else:
            raise ValueError("Invalid fitness metric")


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
                 dataset: dict[DatasetType, Subset],
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics:
        raise NotImplementedError()


    def _build_callbacks(self,
                         model_saving_dir: str,
                         metadata_info: MetadataInfo,
                         train_time: float,
                         early_stop: int | None) -> list[Callback]:
        callbacks: list[Callback] = [ModelCheckpointCallback(model_saving_dir, metadata_info),
                                     TimedStoppingCallback(max_seconds=train_time)]
        if early_stop is not None:
            return callbacks + [EarlyStoppingCallback(patience=early_stop)]
        return callbacks


    def prepare_torch_model(self,
                            model_builder: 'ModelBuilder',
                            pretext_task: Optional[Pretext],
                            reuse_parent_weights: bool,
                            parent_dir: Optional[str],
                            device: Device) -> tuple[nn.Module, bool]:
        restart_train: bool = False
        torch_model = model_builder.assemble_network(type(self), pretext_task)
        if reuse_parent_weights is True \
                and parent_dir is not None \
                and len(os.listdir(parent_dir)) > 0:
            torch_model.load_state_dict(torch.load(os.path.join(parent_dir, WEIGHTS_FILENAME)))
        else:
            if reuse_parent_weights is True:
                restart_train = True
        self._adapt_model_to_device(torch_model, device)
        return torch_model, restart_train


    def testing_performance(self,
                            dataset: dict[DatasetType, Subset],
                            model_dir: str,
                            fitness_metric_name: FitnessMetricName,
                            **kwargs: Any) -> float:
        model_filename: str
        weights_filename: str
        fitness_used: FitnessMetricName = get_config().evolutionary.fitness.metric_name
        if isinstance(self, BarlowTwinsEvaluator):
            # using get_config we do the comparison based on the metric we chose to guide the evolution process
            # bear in mind that `fitness_metric_name` may contain any alternative metric evaluations.
            # this allows us to get some individual whose fitness was obtained via knn evaluation
            # and check how would it be evaluated if it was to be evaluated with an extra linear layer
            if fitness_used == FitnessMetricName.DOWNSTREAM_ACCURACY:
                model_filename = f"complete_{MODEL_FILENAME}"
                weights_filename = f"complete_{WEIGHTS_FILENAME}"
            elif fitness_used == FitnessMetricName.KNN_ACCURACY:
                model_filename = MODEL_FILENAME
                weights_filename = WEIGHTS_FILENAME
            else:
                raise ValueError("Unexpected fitness metric name")
        elif isinstance(self, LegacyEvaluator):
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

        test_set = dataset[DatasetType.TEST]
        assert test_set is not None
        batch_size: int = 64
        test_loader: DataLoader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        loaders_to_use: dict[DatasetType, DataLoader] = {DatasetType.EVO_TEST: test_loader}
        if fitness_metric_name in [FitnessMetricName.DOWNSTREAM_ACCURACY, FitnessMetricName.KNN_ACCURACY]:
            g = Generator()
            g.manual_seed(DEFAULT_SEED)
            if fitness_metric_name != fitness_used:
                # This can only happen if the fitness used to evolve is based on knn evaluation
                # and we also want to provide what would be the fitness using linear evaluation
                dataset_copy = deepcopy(cast(ConcreteDataset, dataset[DatasetType.DOWNSTREAM_TRAIN].dataset))
                dataset_copy.transform = LegacyTransformer(
                    {
                        'random_crop': {
                            'size': 32,
                            'padding': 4
                        },
                        'random_horizontal_flip': {
                            'probability': 0.5
                        }
                    })
                kwargs['dataset'] = dataset_copy
            else:
                kwargs['dataset'] = dataset
            kwargs['dataset_name'] = self.dataset_name
            loaders_to_use[DatasetType.DOWNSTREAM_TRAIN] = \
                DataLoader(dataset[DatasetType.DOWNSTREAM_TRAIN],
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4,
                           drop_last=False,
                           pin_memory=True,
                           generator=g)
            if fitness_metric_name == FitnessMetricName.DOWNSTREAM_ACCURACY:
                fitness_metric_name = FitnessMetricName.ACCURACY
        else:
            assert isinstance(self, LegacyEvaluator)
            kwargs['representation_model'] = self.representation_model
        metric: FitnessMetric = FitnessMetric.create_fitness_metric(fitness_metric_name, **kwargs)
        return metric.compute_metric(torch_model, loaders_to_use, device)



class LegacyEvaluator(BaseEvaluator):

    def __init__(self,
                 dataset_name: str,
                 user_chosen_device: Device,
                 representation_model: Optional[nn.Module],
                 representation_model_training_mode: Optional[DownstreamMode]) -> None:
        self.dataset_name: str = dataset_name
        self.representation_model: Optional[nn.Module] = representation_model
        self.representation_model_training_mode: Optional[DownstreamMode] = representation_model_training_mode
        print(self.representation_model_training_mode)

        super().__init__(dataset_name, user_chosen_device)

    def get_representation_model_shape(self, input_size: Size) -> Size:
        assert self.representation_model is not None
        self.representation_model.to(self.user_chosen_device.value)
        if isinstance(self.representation_model, BarlowTwinsNetwork):
            self.representation_model.projector_model = nn.Identity()
        fake_input = torch.rand(1, *(input_size)).to(self.user_chosen_device.value)
        fake_output = self.representation_model(fake_input)
        return cast(Size, fake_output.data.shape[1:])

    def evaluate(self,
                 phenotype: str,
                 dataset: dict[DatasetType, Subset],
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics:
        from evodenss.networks.model_builder import ModelBuilder

        optimiser: Optimiser
        device: Device = self._decide_device()
        fitness_value: Fitness
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        logger.info(phenotype)
        parsed_network, _, optimiser, _ = parse_phenotype(phenotype)
        dataset_input_size: Size = \
            Size(list(DATASETS_INFO[self.dataset_name]["expected_input_dimensions"]))
        input_size: Size
        if self.representation_model is not None:
            input_size = self.get_representation_model_shape(dataset_input_size)
        else:
            input_size = dataset_input_size
        model_builder: ModelBuilder = ModelBuilder(parsed_network, None, device, input_size)
        try:
            torch_model: nn.Module
            restart_train: bool
            torch_model, restart_train = \
                self.prepare_torch_model(model_builder, None, reuse_parent_weights, parent_dir, device)
            
            training_parameters: list[nn.Parameter] = list(torch_model.parameters())
            if self.representation_model is not None:
                assert self.representation_model_training_mode is not None

                required_grad: bool = False \
                    if self.representation_model_training_mode == DownstreamMode.freeze else True
                for _, param in self.representation_model.named_parameters():
                    param.requires_grad_(required_grad)
                if required_grad is True:
                    training_parameters += list(self.representation_model.parameters())
            
            learning_params: LearningParams = ModelBuilder.assemble_optimiser(
                training_parameters,
                optimiser
            )
            loaders_dict: dict[DatasetType, DataLoader] = DatasetProcessor.get_data_loaders(
                dataset,
                [DatasetType.DOWNSTREAM_TRAIN, DatasetType.VALIDATION, DatasetType.EVO_TEST, DatasetType.TEST],
                learning_params.batch_size)
            metadata_info: MetadataInfo = \
                MetadataInfo.new_instance(self.dataset_name, dataset, optimiser, learning_params, None)
            start_epoch: int = num_epochs if restart_train is False else 0
            trainer = Trainer(model=torch_model,
                              optimiser=learning_params.torch_optimiser,
                              loss_function=nn.CrossEntropyLoss(),
                              train_data_loader=loaders_dict[DatasetType.DOWNSTREAM_TRAIN],
                              validation_data_loader=loaders_dict[DatasetType.VALIDATION],
                              n_epochs=learning_params.epochs,
                              initial_epoch=start_epoch,
                              device=device,
                              callbacks=self._build_callbacks(model_saving_dir,
                                                              metadata_info,
                                                              train_time,
                                                              learning_params.early_stop),
                              representation_model=self.representation_model)
            trainer.train()
            fitness_metric = FitnessMetric.create_fitness_metric(
                get_config().evolutionary.fitness.metric_name,
                dataset_name=self.dataset_name,
                dataset=dataset,
                model_saving_dir=model_saving_dir,
                representation_model=self.representation_model,
                **get_fitness_extra_params()
            )
            fitness_value = Fitness(fitness_metric.compute_metric(torch_model, loaders_dict, device),
                                    type(fitness_metric))
            accuracy: Optional[float]
            if isinstance(fitness_metric, AccuracyMetric) is True:
                accuracy = None
            else:
                accuracy = AccuracyMetric(self.representation_model).compute_metric(torch_model, loaders_dict, device)
            torch_model.requires_grad_(True)
            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                accuracy=accuracy,
                n_trainable_parameters=sum(p.numel() for p in torch_model.parameters() if p.requires_grad),
                n_layers=len(parsed_network.layers),
                n_layers_projector=-1,
                n_epochs=trainer.trained_epochs,
                losses=trainer.loss_values,
                training_time_spent=time()-start,
                total_epochs_trained=start_epoch+trainer.trained_epochs,
                max_epochs_reached=start_epoch+trainer.trained_epochs >= learning_params.epochs
            )
        except InvalidNetwork as e:
            logger.warning(f"Invalid model. Fitness will be computed as invalid individual. Reason: {e.message}")
            fitness_value = self._calculate_invalid_network_fitness(get_config().evolutionary.fitness.metric_name)
            return EvaluationMetrics.default(fitness_value)



class BarlowTwinsEvaluator(BaseEvaluator):

    def __init__(self,
                 dataset_name: str,
                 user_chosen_device: Device) -> None:
        super().__init__(dataset_name, user_chosen_device)

    def evaluate(self,
                 phenotype: str,
                 dataset: dict[DatasetType, Subset],
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics:
        from evodenss.networks.model_builder import ModelBuilder

        parsed_network: ParsedNetwork
        parsed_projector_network: ParsedNetwork
        optimiser: Optimiser
        pretext_task: Optional[Pretext]

        device: Device = self._decide_device()
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        logger.info(phenotype)
        parsed_network, parsed_projector_network, optimiser, pretext_task = parse_phenotype(phenotype)
        input_size: tuple[int, int, int] = DATASETS_INFO[self.dataset_name]["expected_input_dimensions"]
        model_builder: ModelBuilder = \
            ModelBuilder(parsed_network, parsed_projector_network, device, Size(list(input_size)))
        assert pretext_task is not None
        try:
            torch_model: nn.Module
            restart_train: bool
            torch_model, restart_train = \
                self.prepare_torch_model(model_builder, pretext_task, reuse_parent_weights, parent_dir, device)
            learning_params: LearningParams = \
                ModelBuilder.assemble_optimiser(list(torch_model.parameters()), optimiser)
            assert learning_params.batch_size > 1, \
                "batch size should be > 1, otherwise the BatchNorm1D layer won't work"
            loaders_dict: dict[DatasetType, DataLoader] = DatasetProcessor.get_data_loaders(
                dataset,
                [DatasetType.PRETEXT_TRAIN, DatasetType.DOWNSTREAM_TRAIN, DatasetType.EVO_TEST, DatasetType.TEST],
                learning_params.batch_size)
            metadata_info: MetadataInfo = \
                MetadataInfo.new_instance(self.dataset_name, dataset, optimiser, learning_params, pretext_task)
            start_epoch: int = num_epochs if restart_train is False else 0
            trainer = Trainer(model=torch_model,
                              optimiser=learning_params.torch_optimiser,
                              loss_function=BarlowTwinsLoss(**pretext_task.pretext_parameters),
                              train_data_loader=loaders_dict[DatasetType.PRETEXT_TRAIN],
                              validation_data_loader=None,
                              n_epochs=learning_params.epochs,
                              initial_epoch=start_epoch,
                              device=device,
                              callbacks=self._build_callbacks(model_saving_dir,
                                                              metadata_info,
                                                              train_time,
                                                              learning_params.early_stop))

            trainer.barlow_twins_train()

            fitness_metric = FitnessMetric.create_fitness_metric(
                get_config().evolutionary.fitness.metric_name,
                dataset_name=self.dataset_name,
                dataset=dataset,
                model_saving_dir=model_saving_dir,
                **get_fitness_extra_params())
            fitness_value: Fitness
            if isinstance(fitness_metric, AccuracyMetric) or \
                isinstance(fitness_metric, DownstreamAccuracyMetric) or \
                    isinstance(fitness_metric, KNNAccuracyMetric):
                fitness_value = Fitness(
                    fitness_metric.compute_metric(torch_model, loaders_dict, device),
                    type(fitness_metric)
                )
                accuracy = None
            else:
                raise ValueError(f"fitness type not supported for {fitness_metric}")
            torch_model.requires_grad_(True)
            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                accuracy=accuracy,
                n_trainable_parameters=sum(p.numel() for p in torch_model.parameters() if p.requires_grad),
                n_layers=len(parsed_network.layers),
                n_layers_projector=-1 if parsed_projector_network is None else len(parsed_projector_network.layers),
                n_epochs=trainer.trained_epochs,
                losses=trainer.loss_values,
                training_time_spent=time()-start,
                total_epochs_trained=num_epochs+trainer.trained_epochs,
                max_epochs_reached=num_epochs+trainer.trained_epochs >= learning_params.epochs
            )
        except InvalidNetwork:
            logger.warning(f"Invalid model, error during evaluation. Fitness will be computed as invalid individual."
                           f" Reason: {traceback.format_exc()}")
            fitness_value = self._calculate_invalid_network_fitness(get_config().evolutionary.fitness.metric_name)
            return EvaluationMetrics.default(fitness_value)
