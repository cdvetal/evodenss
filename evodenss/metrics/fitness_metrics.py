from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetType
from evodenss.misc.constants import DATASETS_INFO, METADATA_FILENAME, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device, DownstreamMode, FitnessMetricName, OptimiserType
from evodenss.misc.metadata_info import MetadataInfo
from evodenss.misc.utils import InvalidNetwork
from evodenss.networks.evolved_networks import BarlowTwinsNetwork, EvaluationBarlowTwinsNetwork
from evodenss.networks.phenotype_parser import Optimiser
from evodenss.train.callbacks import Callback, ModelCheckpointCallback
from evodenss.train.learning_parameters import LearningParams
from evodenss.train.trainers import Trainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Subset


class Fitness:

    def __init__(self, value: float, metric: type[FitnessMetric]) -> None:
        self.value: float = value
        self.metric: type[FitnessMetric] = metric

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fitness):
            return self.__dict__ == other.__dict__
        return False

    def __lt__(self, other: Fitness) -> bool:
        return self.metric.worse_than(self, other)

    def __gt__(self, other: Fitness) -> bool:
        return self.metric.better_than(self, other)

    def __leq__(self, other: Fitness) -> bool:
        return self.metric.worse_or_equal_than(self, other)

    def __geq__(self, other: Fitness) -> bool:
        return self.metric.better_or_equal_than(self, other)

    def __str__(self) -> str:
        return str(round(self.value, 5))

    def __repr__(self) -> str:
        return self.__str__()


class FitnessMetric(ABC):

    @staticmethod
    def create_fitness_metric(
            metric_name: FitnessMetricName,
            **kwargs: Any) -> FitnessMetric:
        fitness_metric: FitnessMetric
        if metric_name == FitnessMetricName.ACCURACY:
            class_kwargs = \
                {k: kwargs.get(k, None) for k in AccuracyMetric.__init__.__code__.co_varnames[1:]}
            fitness_metric = AccuracyMetric(**class_kwargs)
        elif metric_name == FitnessMetricName.KNN_ACCURACY:
            # get parameter names from contructor and exclude the first one (self)
            class_kwargs = \
                {k: kwargs.get(k, None) for k in KNNAccuracyMetric.__init__.__code__.co_varnames[1:]}
            fitness_metric = KNNAccuracyMetric(**class_kwargs)
        elif metric_name == FitnessMetricName.DOWNSTREAM_ACCURACY:
            # get parameter names from contructor and exclude the first one (self)
            class_kwargs = \
                {k: kwargs.get(k, None) for k in DownstreamAccuracyMetric.__init__.__code__.co_varnames[1:]}
            fitness_metric = DownstreamAccuracyMetric(**class_kwargs)
        else:
            raise ValueError(f"Unknown fitness metric name found: {metric_name.value}")
        return fitness_metric

    @abstractmethod
    def compute_metric(self,
                       model: nn.Module,
                       loaders_dict: dict[DatasetType, DataLoader[ConcreteDataset]],
                       device: Device) -> float:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worst_fitness(cls) -> Fitness:
        raise NotImplementedError()


class AccuracyMetric(FitnessMetric):

    def __init__(self, representation_model: Optional[nn.Module]=None) -> None:
        super().__init__()
        self.representation_model = representation_model

    def compute_metric(self,
                       model: nn.Module,
                       loaders_dict: dict[DatasetType, DataLoader[ConcreteDataset]],
                       device: Device) -> float:
        model.eval()
        correct_guesses: float = 0
        size: int = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in loaders_dict[DatasetType.EVO_TEST]:
                inputs, labels = data[0].to(device.value, non_blocking=True), \
                    data[1].to(device.value, non_blocking=True)
                if self.representation_model is not None:
                    inputs = self.representation_model(inputs)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct_guesses += (predicted == labels).float().sum().item()
                size += len(labels)
        return correct_guesses/size

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(-1.0, cls)


class DownstreamAccuracyMetric(FitnessMetric):

    def __init__(self,
                 dataset_name: str,
                 dataset: dict[DatasetType, Subset[ConcreteDataset]],
                 batch_size: int,
                 downstream_mode: DownstreamMode,
                 downstream_epochs: int,
                 optimiser_type: OptimiserType,
                 optimiser_parameters: dict[str, str],
                 model_saving_dir: Optional[str]=None) -> None:
        super().__init__()
        self.dataset_name: str = dataset_name
        self.dataset: dict[DatasetType, Subset[ConcreteDataset]] = dataset
        self.model_saving_dir: Optional[str] = model_saving_dir
        self.downstream_mode: DownstreamMode = downstream_mode
        self.optimiser_type: OptimiserType = optimiser_type
        self.optimiser_parameters: dict[str, str] = optimiser_parameters
        self.optimiser_parameters['batch_size'] = str(batch_size)
        self.optimiser_parameters['epochs'] = str(downstream_epochs)


    def _get_parameters_to_tune(self,
                                model: EvaluationBarlowTwinsNetwork) -> list[nn.Parameter]:
        model_parameters: Iterator[tuple[str, nn.Parameter]] = model.named_parameters()
        params_to_tune: list[nn.Parameter]
        if self.downstream_mode == DownstreamMode.freeze:
            params_to_tune = [param for name, param in model_parameters
                if name in {f'final_layer.{model.last_fc_layer_index}.weight',
                            f'final_layer.{model.last_fc_layer_index}.bias'}
            ]
            model.barlow_twins_trained_model.requires_grad_(False)
            model.final_layer.requires_grad_(True)
        elif self.downstream_mode == DownstreamMode.finetune:
            params_to_tune = [param for _, param in model_parameters]
            model.barlow_twins_trained_model.requires_grad_(True)
            model.final_layer.requires_grad_(True)
        else:
            raise ValueError(f"downstream mode not supported: [{self.downstream_mode}]")
        return params_to_tune


    def compute_metric(self,
                       model: nn.Module,
                       loaders_dict: dict[DatasetType, DataLoader[ConcreteDataset]],
                       device: Device) -> float:
        from evodenss.networks.model_builder import ModelBuilder
        n_classes: int = DATASETS_INFO[self.dataset_name]["classes"]

        complete_model: EvaluationBarlowTwinsNetwork
        if isinstance(model, BarlowTwinsNetwork) is True:
            complete_model = EvaluationBarlowTwinsNetwork(model, n_classes, device)
            complete_model.to(device.value, non_blocking=True)
            params_to_tune: list[nn.Parameter] = \
                self._get_parameters_to_tune(complete_model)

            downstream_optimiser = Optimiser(self.optimiser_type, self.optimiser_parameters)
            downstream_learning_params: LearningParams = \
                ModelBuilder.assemble_optimiser(params_to_tune, downstream_optimiser)
            callbacks: list[Callback] = []

            if self.model_saving_dir is not None:
                metadata_info = \
                    MetadataInfo.new_instance(self.dataset_name, self.dataset, downstream_optimiser,
                                              downstream_learning_params, None)
                callbacks.append(ModelCheckpointCallback(
                    self.model_saving_dir,
                    model_filename=f"complete_{MODEL_FILENAME}",
                    weights_filename=f"complete_{WEIGHTS_FILENAME}",
                    metadata_filename=f"complete_{METADATA_FILENAME}",
                    metadata_info=metadata_info)
                )
            last_layer_trainer = Trainer(model=complete_model,
                                         optimiser=downstream_learning_params.torch_optimiser,
                                         loss_function=nn.CrossEntropyLoss(),
                                         train_data_loader=loaders_dict[DatasetType.DOWNSTREAM_TRAIN],
                                         validation_data_loader=loaders_dict.get(DatasetType.VALIDATION, None),
                                         n_epochs=downstream_learning_params.epochs,
                                         initial_epoch=0,
                                         device=device,
                                         callbacks=callbacks)
            last_layer_trainer.train()
        elif isinstance(model, EvaluationBarlowTwinsNetwork) is True:
            complete_model = cast(EvaluationBarlowTwinsNetwork, model)
            complete_model.to(device.value, non_blocking=True)
        else:
            raise TypeError(f"Invalid model type: {type(model)}")
        complete_model.eval()
        correct_guesses: float = 0
        size: int = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in loaders_dict[DatasetType.EVO_TEST]:
                inputs, labels = data[0].to(device.value, non_blocking=True), \
                    data[1].to(device.value, non_blocking=True)
                outputs = complete_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct_guesses += (predicted == labels).float().sum().item()
                size += len(labels)
        return correct_guesses/size

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(-1.0, cls)


class KNNAccuracyMetric(FitnessMetric):

    def __init__(self, dataset_name: str, k: int, t: float) -> None:
        super().__init__()
        self.dataset_name: str = dataset_name
        self.k: int = k
        self.t: float = t

    def knn_predict(self,
                    feature: Tensor,
                    feature_bank: Tensor,
                    feature_labels: Tensor,
                    knn_k: int,
                    knn_t: float) -> Tensor:
        """
        Helper method to run kNN predictions on features based on a feature bank

        Args:
            feature: Tensor of shape [N, D] consisting of N D-dimensional features
            feature_bank: Tensor of a database of features used for kNN
            feature_labels: Labels for the features in our feature_bank
            classes: Number of classes (e.g. 10 for CIFAR-10)
            knn_k: Number of k neighbors used for kNN
        """

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices) # [B, K]

        #sim_weight = (sim_weight / knn_t).exp() # we do a reweighting of the similarities
        # counts for each class
        n_classes: int = DATASETS_INFO[self.dataset_name]["classes"]
        one_hot_label = torch.zeros(feature.size(0) * knn_k, n_classes, device=sim_labels.device)
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0) # [B*K, C]

        # weighted score ---> [B, C]
        pred_scores = \
            torch.sum(one_hot_label.view(feature.size(0), -1, n_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def compute_metric(self,
                       model: nn.Module,
                       loaders_dict: dict[DatasetType, DataLoader[ConcreteDataset]],
                       device: Device) -> float:
        try:
            model.eval()
            # cast(BarlowTwinsNetwork, model).projector_model = nn.Identity()
            correct_guesses: float = 0
            size: int = 0
            feature_bank: list[Tensor] = []
            labels_bank: list[Tensor] = []
            feature: Tensor
            # we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                # generate feature bank
                for inputs, labels in loaders_dict[DatasetType.DOWNSTREAM_TRAIN]:
                    feature = model(inputs.to(device.value, non_blocking=True))
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)
                    labels_bank.append(labels.to(device.value, non_blocking=True))

                feature_bank_tensor: Tensor = torch.cat(feature_bank, dim=0).t().contiguous() # [D, N]
                # [N]
                labels_bank_tensor: Tensor = torch.cat(labels_bank, dim=0).t().contiguous() # [D, N]
                # loop test data to predict the label by weighted knn search
                for data in loaders_dict[DatasetType.EVO_TEST]:
                    inputs, labels = data[0].to(device.value, non_blocking=True), \
                        data[1].to(device.value, non_blocking=True)
                    
                    feature = model(inputs)
                    feature = F.normalize(feature, dim=1)
                    pred_labels: Tensor = self.knn_predict(
                        feature, feature_bank_tensor, labels_bank_tensor, self.k, self.t
                    )
                    predicted = torch.flatten(torch.transpose(pred_labels[:,:1], 0, 1))
                    correct_guesses += (predicted == labels).float().sum().item()
                    size += len(labels)
            return correct_guesses/size
        except Exception as e:
            raise InvalidNetwork(str(e)) from e

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(-1.0, cls)
