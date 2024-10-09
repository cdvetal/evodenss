from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import TYPE_CHECKING

from evodenss.misc.enums import LayerType


if TYPE_CHECKING:
    from evodenss.networks.phenotype_parser import Layer


@dataclass
class Dimensions:
    channels: int
    height: int
    width: int

    @classmethod
    def from_layer(cls, layer: Layer, input_dimensions: "Dimensions") -> "Dimensions":
        out_channels: int
        height: int
        width: int
        kernel_size: int
        padding_w: int
        padding_h: int
        if layer.layer_type == LayerType.CONV:
            out_channels = layer.layer_parameters['out_channels']
            if layer.layer_parameters['padding'] == "same":
                height = input_dimensions.height
                width = input_dimensions.width
            elif layer.layer_parameters['padding'] == "valid":
                kernel_size = layer.layer_parameters['kernel_size']
                height = ceil((input_dimensions.height - kernel_size + 1) / layer.layer_parameters['stride'])
                width = ceil((input_dimensions.width - kernel_size + 1) / layer.layer_parameters['stride'])
            elif isinstance(layer.layer_parameters['padding'], tuple):
                padding_h = layer.layer_parameters['padding'][0]
                padding_w = layer.layer_parameters['padding'][1]
                kernel_size_h: int = layer.layer_parameters['kernel_size'][0]
                kernel_size_w: int = layer.layer_parameters['kernel_size'][1]
                height = ceil((input_dimensions.height - kernel_size_h + 1) / \
                              layer.layer_parameters['stride']) + padding_h * 2
                width = ceil((input_dimensions.width - kernel_size_w + 1) / \
                             layer.layer_parameters['stride']) + padding_w * 2
            return cls(out_channels, height, width)
        elif layer.layer_type in [LayerType.POOL_AVG, LayerType.POOL_MAX]:
            assert isinstance(layer.layer_parameters['padding'], str) is True
            out_channels = input_dimensions.channels
            kernel_size = layer.layer_parameters['kernel_size']
            if layer.layer_parameters['padding'] == "valid":
                padding_w = padding_h = 0
            elif layer.layer_parameters['padding'] == "same":
                paddings: tuple[int, int, int, int] = \
                    input_dimensions.compute_adjusting_padding(layer.layer_parameters['kernel_size'],
                                                               layer.layer_parameters['stride'])
                padding_w = paddings[2] + paddings[3]
                padding_h = paddings[0] + paddings[1]
            kernel_w: int
            kernel_h: int
            if isinstance(kernel_size, int):
                kernel_w = kernel_h = kernel_size
            elif isinstance(kernel_size, tuple):
                kernel_h = kernel_size[0]
                kernel_w = kernel_size[1]
            height = ceil((input_dimensions.height - kernel_h + 1) / layer.layer_parameters['stride']) + padding_h
            width = ceil((input_dimensions.width - kernel_w + 1) / layer.layer_parameters['stride']) + padding_w
            return cls(out_channels, height, width)
        elif layer.layer_type in [LayerType.BATCH_NORM, LayerType.DROPOUT, LayerType.IDENTITY, LayerType.RELU_AGG]:
            return input_dimensions
        elif layer.layer_type == LayerType.BATCH_NORM_PROJ:
            return cls(input_dimensions.flatten(), height=1, width=1)
        elif layer.layer_type == LayerType.FC:
            return cls(layer.layer_parameters['out_features'], height=1, width=1)
        else:
            raise ValueError(f"Can't create Dimensions object for layer [{layer.layer_type}]")


    def compute_adjusting_padding(self,
                                  kernel_size: int,
                                  stride: int) -> tuple[int, int, int, int]:
        padding_w: float = (self.width - (self.width - kernel_size + 1) / stride) / 2
        padding_h: float = (self.height - (self.height - kernel_size + 1) / stride) / 2
        padding_left: int = ceil(padding_w)
        padding_right: int = floor(padding_w)
        padding_top: int = ceil(padding_h)
        padding_bottom: int = floor(padding_h)
        return (padding_left, padding_right, padding_top, padding_bottom)

    def flatten(self) -> int:
        return self.channels * self.height * self.width

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.channels, self.height, self.width))
