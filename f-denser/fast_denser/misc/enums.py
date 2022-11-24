from enum import unique, Enum
from typing import Any, List

import torch

class ExtendedEnum(Enum):
    @classmethod
    def enum_values(cls) -> List[Any]:
        return list(map(lambda c: c.value, cls)) # type: ignore

@unique
class Entity(ExtendedEnum):
    LAYER = "layer"
    OPTIMISER = "learning"

@unique
class ProjectorUsage(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"

@unique
class Device(Enum):
    CPU = "cpu"
    GPU = "mps" if torch.backends.mps.is_available() else "cuda:0"

@unique
class LayerType(ExtendedEnum):
    CONV = "conv"
    BATCH_NORM = "batch_norm"
    POOL_AVG = "pool_avg"
    POOL_MAX = "pool_max"
    FC = "fc"
    DROPOUT = "dropout"
    IDENTITY = "identity"

@unique
class OptimiserType(Enum):
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LARS = "lars"

@unique
class ActivationType(Enum):
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"

@unique
class TransformOperation(ExtendedEnum):
    COLOR_JITTER = "color_jitter"
    NORMALIZE = "normalize"
    PADDING = "padding"
    RANDOM_CROP = "random_crop"
    HORIZONTAL_FLIPPING = "random_horizontal_flip"
    RANDOM_RESIZED_CROP = "random_resized_crop"
    RANDOM_GRAYSCALE = "random_grayscale"
    GAUSSIAN_BLUR = "gaussian_blur"
    SOLARIZE = "random_solarize"
    CENTER_CROP = "center_crop"
    RESIZE = "resize"

@unique
class FitnessMetricName(ExtendedEnum):
    LOSS = "loss"
    ACCURACY = "accuracy"

@unique
class DatasetType(Enum):
    EVO_TRAIN = "evo_train"
    EVO_VALIDATION = "evo_validation"
    EVO_TEST = "evo_test"
    TEST = "test"
