from enum import unique, Enum
from typing import Any

import torch


@unique
class AttributeType(Enum):
    INT = "int"
    INT_POWER2 = "int_power2"
    INT_POWER2_INV = "inv_power2"
    FLOAT = "float"


class ExtendedEnum(Enum):
    @classmethod
    def enum_values(cls) -> list[Any]:
        return list(map(lambda c: c.value, cls))


@unique
class Entity(ExtendedEnum):
    LAYER = "layer"
    OPTIMISER = "learning"
    PROJECTOR_LAYER = "projector_layer"
    PRETEXT_TASK = "pretext"


@unique
class Device(Enum):
    CPU = "cpu"
    GPU = "mps" if torch.backends.mps.is_available() else "cuda:0"


@unique
class LayerType(ExtendedEnum):
    CONV = "conv"
    BATCH_NORM = "batch_norm"
    BATCH_NORM_PROJ = "batch_norm_proj"
    POOL_AVG = "pool_avg"
    POOL_MAX = "pool_max"
    FC = "fc"
    DROPOUT = "dropout"
    IDENTITY = "identity"
    RELU_AGG = "relu_agg"


@unique
class OptimiserType(str, Enum):
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LARS = "lars"


@unique
class PretextType(Enum):
    BT = "bt"


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


class MutationType(ExtendedEnum):
    TRAIN_LONGER = "train_longer"
    ADD_LAYER = "add_layer"
    REUSE_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    DSGE_TOPOLOGICAL = "dsge_topological"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    DSGE_NON_TOPOLOGICAL = "dsge_non_topological"

    def __str__(self) -> str:
        return self.value


@unique
class FitnessMetricName(ExtendedEnum):
    KNN_ACCURACY = "knn_accuracy"
    DOWNSTREAM_ACCURACY = "downstream_accuracy"
    ACCURACY = "accuracy"


class Goal(Enum):
    maximise = "maximise"
    minimise = "minimise"


class DownstreamMode(Enum):
    freeze = "freeze"
    finetune = "finetune"


class LearningType(Enum):
    supervised = "supervised"
    self_supervised = "self-supervised"
