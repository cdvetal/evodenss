from __future__ import annotations
from abc import ABC, abstractmethod

import math
from sys import float_info
from typing import Any, TYPE_CHECKING, Optional

from fast_denser.misc.enums import Device

import torch

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

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
    
    def __init__(self, batch_size: Optional[int]=None, loss_function: Any=None) -> None:
        self.batch_size: Optional[int] = batch_size
        self.loss_function: Any = loss_function

    @abstractmethod
    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
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

    def __init__(self, batch_size: Optional[int]=None, loss_function: Any=None) -> None:
        super().__init__(batch_size, loss_function)

    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
        model.eval()
        correct_guesses: float = 0
        size: int = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(device.value, non_blocking=True), data[1].to(device.value, non_blocking=True)
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

'''
class BTAccuracyMetric(FitnessMetric):

    def __init__(self, batch_size: Optional[int]=None, loss_function: Any=None) -> None:
        super().__init__(batch_size, loss_function)

    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
        print("computing accuracy bt")
        metric: Metric = Accuracy().to(device.value, non_blocking=True)
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
'''

class LossMetric(FitnessMetric):

    def __init__(self, loss_function: Any, batch_size: Optional[int]=None) -> None:
        super().__init__(batch_size, loss_function)

    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
        pass

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(float_info.max, cls)

class BTLossMetric(FitnessMetric):

    def __init__(self, batch_size: int, loss_function: Any=None) -> None:
        super().__init__(batch_size, loss_function)

    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
        model.eval()
        total_loss: float
        n_batches: int = len(data_loader)
        with torch.no_grad():
            total_loss_tensor = torch.zeros(size=(1,), device=device.value)
            for i, ((y_a, y_b), _) in enumerate(data_loader, 0):
                inputs_a = y_a.to(device.value, non_blocking=True)
                inputs_b = y_b.to(device.value, non_blocking=True)
                with torch.cuda.amp.autocast():
                    all_loss_components = model.forward(inputs_a, inputs_b, self.batch_size)
                    complete_loss = all_loss_components[-1]
                total_loss_tensor += complete_loss/n_batches
        total_loss = float(total_loss_tensor.data)
        if math.isinf(total_loss) is True or math.isnan(total_loss):
            raise ValueError(f"Invalid loss (inf or NaN): {total_loss}")
        else:
            return total_loss

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(float_info.max, cls)