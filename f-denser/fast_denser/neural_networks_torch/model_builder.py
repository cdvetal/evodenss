from __future__ import annotations

from collections import Counter
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING


from fast_denser.misc.constants import SEPARATOR_CHAR
from fast_denser.misc.enums import ActivationType, Device, LayerType, OptimiserType, ProjectorUsage
from fast_denser.misc.phenotype_parser import Layer
from fast_denser.misc.utils import InvalidNetwork, InputLayerId, LayerId
from fast_denser.neural_networks_torch import Dimensions, LearningParams
from fast_denser.neural_networks_torch.evolved_networks import BarlowTwinsNetwork, \
    EvolvedNetwork, LegacyNetwork
from fast_denser.neural_networks_torch.lars import LARS
from fast_denser.neural_networks_torch.evaluators import *

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import Size, nn, optim, Tensor

if TYPE_CHECKING:
    from fast_denser.misc.phenotype_parser import Optimiser


logger = logging.getLogger(__name__)

class ModelBuilder():

    def __init__(self,
                 layers: List[Layer],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 device: Device,
                 input_shape: Size) -> None:
        self.layers: List[Layer] = layers
        self.layers_connections: Dict[LayerId, List[InputLayerId]] = layers_connections
        self.layer_shapes: Dict[InputLayerId, Dimensions] = {
            InputLayerId(-1): Dimensions(channels=input_shape[0],
                                         height=input_shape[1],
                                         width=input_shape[2])
        }
        self.layer_type_counts: Counter = Counter([])
        self.device = device


    @classmethod
    def assemble_optimiser(cls, model_parameters: Iterable[Tensor], optimiser: Optimiser) -> LearningParams:
        early_stop: Optional[int] = optimiser.optimiser_parameters.pop("early_stop", None)
        batch_size: int = optimiser.optimiser_parameters.pop("batch_size")
        epochs: int = optimiser.optimiser_parameters.pop("epochs")
        torch_optimiser: optim.Optimizer
        if optimiser.optimiser_type == OptimiserType.RMSPROP:
            torch_optimiser = optim.RMSprop(params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.GRADIENT_DESCENT:
            torch_optimiser = optim.SGD(params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.ADAM:
            optimiser.optimiser_parameters["betas"] = (
                optimiser.optimiser_parameters.pop("beta1"),
                optimiser.optimiser_parameters.pop("beta2")
            )
            torch_optimiser = optim.Adam(params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.LARS:
            param_weights = []
            param_biases = []
            for param in model_parameters:
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)
            parameters = [{'params': param_weights}, {'params': param_biases}]
            torch_optimiser = LARS(parameters,
                                   batch_size=batch_size,
                                   eta=0.001,
                                   weight_decay_filter=True,
                                   lars_adaptation_filter=True,
                                   **optimiser.optimiser_parameters)
            #weight_decay, momentum, 
        else:
            raise ValueError(f"Invalid optimiser name found: {optimiser.optimiser_type}")
        return LearningParams(
            early_stop=early_stop,
            batch_size=batch_size,
            epochs=epochs,
            torch_optimiser=torch_optimiser
        )


    def assemble_network(self, evaluation_type: type[BaseEvaluator]) -> EvolvedNetwork:
        layer_to_add: nn.Module
        torch_layers: List[Tuple[str, nn.Module]] = []
        collected_extra_torch_layers: List[Tuple[str, nn.Module]] = []
        self.layer_type_counts.update([layer_type for layer_type in LayerType])

        try:
            for i, l in enumerate(self.layers):
                layer_name: str = f"{l.layer_type.value}{SEPARATOR_CHAR}{self.layer_type_counts[l.layer_type]}"
                self.layer_type_counts.update([l.layer_type])

                inputs_shapes: Dict[InputLayerId, Dimensions] = \
                    { input_id: self.layer_shapes[input_id] for input_id in self.layers_connections[LayerId(i)] }
                minimum_extra_id: int = len(self.layers) + len(collected_extra_torch_layers)
                extra_layers: Dict[InputLayerId, Layer] = self._get_layers_to_fix_shape_mismatches(inputs_shapes, minimum_extra_id)
                for input_id, extra_layer in extra_layers.items():

                    extra_layer_name: str = f"{extra_layer.layer_type.value}_{self.layer_type_counts[extra_layer.layer_type]}"
                    self.layer_type_counts.update([extra_layer.layer_type])

                    # Add the new connection in between
                    self.layers_connections[LayerId(extra_layer.layer_id)] = [input_id]
                    # Fix the old ones
                    self.layers_connections[LayerId(i)].remove(input_id)
                    self.layers_connections[LayerId(i)].append(InputLayerId(extra_layer.layer_id))
                    
                    layer_to_add = self._create_torch_layer(extra_layer, extra_layer_name)
                    collected_extra_torch_layers.append((extra_layer_name, layer_to_add))
                    
                layer_to_add = self._create_torch_layer(l, layer_name)
                torch_layers.append((layer_name, layer_to_add))

            if evaluation_type is LegacyEvaluator:
                return LegacyNetwork(torch_layers + collected_extra_torch_layers, self.layers_connections)
            elif evaluation_type is BarlowTwinsEvaluator:
                return BarlowTwinsNetwork(torch_layers + collected_extra_torch_layers,
                                          self.layers_connections,
                                          self.layer_shapes,
                                          ProjectorUsage.EXPLICIT,
                                          self.device)
            else:
                raise ValueError(f"Unexpected network type: {evaluation_type}")
        except InvalidNetwork as e:
            raise e


    def _get_layers_to_fix_shape_mismatches(self,
                                            inputs_shapes: Dict[InputLayerId, Dimensions],
                                            minimum_extra_id: int) -> Dict[InputLayerId, Layer]:
        new_layers_dict: Dict[InputLayerId, Layer] = {}
        
        # max_channels: int = min(list(map(lambda x: x.channels, inputs_shapes.values())))
        min_height: int = min(list(map(lambda x: x.height, inputs_shapes.values())))
        min_width: int = min(list(map(lambda x: x.width, inputs_shapes.values())))
        layer_id: int = minimum_extra_id
        for input_id, input_shape in inputs_shapes.items():
            if input_shape.height > min_height or input_shape.width > min_width :
                logging.warning("Shape mismatch found")
                new_layers_dict[input_id] = Layer(layer_id=LayerId(layer_id),
                                                  layer_type=LayerType.POOL_MAX,
                                                  layer_parameters={'kernel_size_fix': str((input_shape.height-min_height + 1,
                                                                                            input_shape.width-min_width + 1)),
                                                                    'padding': 'valid',
                                                                    'stride': '1'})
                layer_id += 1
        return new_layers_dict


    def _create_activation_layer(self, activation: ActivationType) -> nn.Module:
        if activation == ActivationType.RELU:
            return nn.ReLU()
        elif activation == ActivationType.SIGMOID:
            return nn.Sigmoid()
        elif activation == ActivationType.SOFTMAX:
            return nn.Softmax()
        else:
            raise ValueError(f"Unexpected activation function found: {activation}")


    def _create_torch_layer(self, layer: Layer, layer_name: str) -> nn.Module:
        layer_to_add: nn.Module        
        inputs_shapes = {input_id: self.layer_shapes[input_id]
                         for input_id in self.layers_connections[layer.layer_id]}
        expected_input_dimensions: Optional[Dimensions]
        #if len(set(inputs_shapes.values())) == 1:
        first_input = list(inputs_shapes.values())[0]
        if len(inputs_shapes) > 1:
            total_channels: int = sum([x.channels for x in list(inputs_shapes.values())])
            expected_input_dimensions = Dimensions(total_channels, first_input.height, first_input.width)
        else:
            # in this case all inputs will have the same dimensions, just take the first one...
            expected_input_dimensions = first_input
        #else:
        #    expected_input_dimensions = None
        
        #if expected_input_dimensions is None:
        #    raise InvalidNetwork(f"Invalid network found. Layer [{layer.layer_id}] has inputs with different dimensions: {inputs_shapes}")

        self.layer_shapes[InputLayerId(layer.layer_id)] = Dimensions.from_layer(layer, expected_input_dimensions)

        if layer.layer_type == LayerType.CONV:
            layer_to_add = self._build_convolutional_layer(layer, expected_input_dimensions)
        elif layer.layer_type == LayerType.BATCH_NORM:
            layer_to_add = self._build_batch_norm_layer(layer, expected_input_dimensions)
        elif layer.layer_type == LayerType.POOL_AVG:
            layer_to_add = self._build_avg_pooling_layer(layer, expected_input_dimensions)
        elif layer.layer_type == LayerType.POOL_MAX:
            layer_to_add = self._build_max_pooling_layer(layer, expected_input_dimensions)
        elif layer.layer_type == LayerType.FC:
            layer_to_add = self._build_dense_layer(layer, layer_name, expected_input_dimensions)
        elif layer.layer_type == LayerType.DROPOUT:
            layer_to_add = self._build_dropout_layer(layer, expected_input_dimensions)
        elif layer.layer_type == LayerType.IDENTITY:
            layer_to_add = nn.Identity()
        #print("== shapes: ", self.layer_shapes)
        #print("=== ", layer_to_add, layer.layer_id)        
        return layer_to_add

    def _build_convolutional_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
                #m.bias.data.fill_(0.01)
        
        layer_to_add: nn.Module
        activation: ActivationType = ActivationType(layer.layer_parameters.pop("act"))

        assert layer.layer_parameters['padding'] in ['valid', 'same']

        # TODO: put bias = False if all next layers connected to this layer are batch norm
        
        # If padding = same torch does not support strided convolutions.
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        if layer.layer_parameters['padding'] == "same":
            layer.layer_parameters['stride'] = 1
        if activation == ActivationType.LINEAR:
            layer_to_add = nn.Conv2d(**layer.layer_parameters, in_channels=input_dimensions.channels, device=self.device.value)
            #torch.nn.init.xavier_uniform(layer_to_add.weight)
        else:
            layer_to_add = nn.Sequential(nn.Conv2d(**layer.layer_parameters, in_channels=input_dimensions.channels, device=self.device.value),
                                         self._create_activation_layer(activation))
            layer_to_add.apply(init_weights)
        return layer_to_add

    def _build_batch_norm_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        layer_to_add = nn.BatchNorm2d(**layer.layer_parameters, num_features=input_dimensions.channels, device=self.device.value)
        #torch.nn.init.xavier_uniform(layer_to_add.weight)
        #layer_to_add.bias.data.fill_(0.01)
        return layer_to_add

    def _build_avg_pooling_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        torch_layers_to_add: List[nn.Module] = []
        padding_type: str = layer.layer_parameters.pop("padding")
        layer.layer_parameters['padding'] = 0
        torch_layers_to_add.append(nn.AvgPool2d(**layer.layer_parameters))
        if padding_type == "same":
            # https://github.com/pytorch/pytorch/issues/3298
            padding_to_apply: Tuple[int, int, int, int] = \
                input_dimensions.compute_adjusting_padding(layer.layer_parameters['kernel_size'],
                                                           layer.layer_parameters['stride'])
            torch_layers_to_add.append(nn.ZeroPad2d(padding_to_apply))
        return nn.Sequential(*torch_layers_to_add)

    def _build_max_pooling_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        torch_layers_to_add: List[nn.Module] = []
        padding_type: str = layer.layer_parameters.pop("padding")
        layer.layer_parameters['padding'] = 0
        torch_layers_to_add.append(nn.MaxPool2d(**layer.layer_parameters))
        if padding_type == "same":
            # https://github.com/pytorch/pytorch/issues/3298
            padding_to_apply: Tuple[int, int, int, int] = \
                input_dimensions.compute_adjusting_padding(layer.layer_parameters['kernel_size'],
                                                           layer.layer_parameters['stride'])
            torch_layers_to_add.append(nn.ZeroPad2d(padding_to_apply))
        return nn.Sequential(*torch_layers_to_add)

    def _build_dropout_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        layer.layer_parameters['p'] = min(0.5, layer.layer_parameters.pop("rate"))
        layer_to_add = nn.Dropout(**layer.layer_parameters)
        return layer_to_add

    def _build_dense_layer(self, layer: Layer, layer_name: str, input_dimensions: Dimensions) -> nn.Sequential:
        activation = ActivationType(layer.layer_parameters.pop("act"))
        torch_layers_to_add: List[nn.Module] = []
        if layer_name == f"{LayerType.FC.value}{SEPARATOR_CHAR}1":
            torch_layers_to_add.append(nn.Flatten())
            torch_layers_to_add.append(nn.Linear(**layer.layer_parameters, in_features=input_dimensions.flatten(), device=self.device.value))
            #torch.nn.init.xavier_uniform(torch_layers_to_add[1].weight)
            #torch_layers_to_add[1].bias.data.fill_(0.01)
        else:
            torch_layers_to_add.append(nn.Linear(**layer.layer_parameters, in_features=input_dimensions.channels, device=self.device.value))
            #torch.nn.init.xavier_uniform(torch_layers_to_add[0].weight)
            #torch_layers_to_add[0].bias.data.fill_(0.01)
        if activation != ActivationType.LINEAR:
            torch_layers_to_add.append(self._create_activation_layer(activation))
        
        return nn.Sequential(*torch_layers_to_add)
