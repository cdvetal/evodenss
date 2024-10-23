from dataclasses import dataclass
from itertools import chain, dropwhile, takewhile
from typing import Any, Optional, cast

from evodenss.misc.enums import Entity, LayerType, OptimiserType, PretextType
from evodenss.misc.utils import InputLayerId, LayerId


class Layer:

    def __init__(self,
                 layer_id: LayerId,
                 layer_type: LayerType,
                 layer_parameters: dict[str, str]) -> None:
        self.layer_id: LayerId = layer_id
        self.layer_type: LayerType = layer_type
        self.layer_parameters: dict[str, Any] = dict(self._convert(k, v) for k,v in layer_parameters.items())

    def _convert(self, key: str, value: str) -> tuple[str, Any]:
        if key == "bias":
            return key, value.title() == "True"
        elif key in ["rate"]:
            return key, float(value)
        elif key in ["out_channels", "out_features", "kernel_size", "stride"]:
            return key, int(value)
        elif key in ["act", "padding"]:
            return key, value
        elif key == "input":
            return key, list(map(int, value))
        elif key == "kernel_size_fix":
            return "kernel_size", tuple(map(int, value[1:-1].split(',')))
        elif key == "padding_fix":
            return "padding", tuple(map(int, value[1:-1].split(',')))
        else:
            raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Layer):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        return f"Layer [{self.layer_type}] with id [{self.layer_id}] and params: {self.layer_parameters}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ParsedNetwork:
    layers: list[Layer]
    layers_connections: dict[LayerId, list[InputLayerId]]

    # It gets the layer id that corresponds to the final/output layer
    def get_output_layer_id(self) -> LayerId:
        keyset: set[int] = cast(set[int], set(self.layers_connections.keys()))
        values_set: set[int] = cast(set[int], set(list(chain(*self.layers_connections.values()))))
        result: set[int] = keyset.difference(values_set)
        assert len(result) == 1
        return LayerId(list(result)[0])
    
    def is_empty(self) -> bool:
        return not self.layers and not self.layers_connections


class Optimiser:

    def __init__(self,
                 optimiser_type: OptimiserType,
                 optimiser_parameters: dict[str, str]) -> None:
        self.optimiser_type: OptimiserType = optimiser_type
        self.optimiser_parameters: dict[str, Any] = {
            k: self._convert(k, v) for k,v in optimiser_parameters.items()
        }

    def _convert(self, key: str, value: str) -> Any:
        if key == "nesterov":
            return value.title() == "True"
        elif key in ["lr", "lr_weights", "lr_biases", "alpha", "weight_decay", "momentum", "beta1", "beta2"]:
            return float(value)
        elif key in ["early_stop", "batch_size", "epochs"]:
            return int(value)
        else:
            raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Optimiser):
            return self.__dict__ == other.__dict__
        return False


class Pretext:

    def __init__(self,
                 pretext_type: PretextType,
                 pretext_parameters: dict[str, str]) -> None:
        self.pretext_type: PretextType = pretext_type
        self.pretext_parameters: dict[str, Any] = {
            k: self._convert(k, v) for k,v in pretext_parameters.items()
        }

    def _convert(self, key: str, value: str) -> Any:
        if key in ["lamb"]:
            return float(value)
        else:
            raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Pretext):
            return self.__dict__ == other.__dict__
        return False

def parse_phenotype(phenotype: str) -> tuple[ParsedNetwork, ParsedNetwork, Optimiser, Optional[Pretext]]:
    phenotype_as_list: list[list[str]] = \
        list(map(lambda x: x.split(":"), phenotype.split(" ")))

    optimiser: Optimiser
    pretext_task: Optional[Pretext] = None
    layers: list[Layer] = []
    layers_connections: dict[LayerId, list[InputLayerId]] = {}
    projector_layers: list[Layer] = []
    projector_layers_connections: dict[LayerId, list[InputLayerId]] = {}
    layer_id: int = 0
    projector_layer_id: int = 0
    while phenotype_as_list:
        entity: Entity = Entity(phenotype_as_list[0][0])
        name: str = phenotype_as_list[0][1]
        entity_parameters: dict[str, str] = {
            kv[0]: kv[1]
            for kv in takewhile(lambda kv: kv[0] not in Entity.enum_values(),
                                phenotype_as_list[1:])
        }
        phenotype_as_list = list(dropwhile(lambda kv: kv[0] not in Entity.enum_values(),
                                           phenotype_as_list[1:]))
        input_info: list[InputLayerId]
        if entity == Entity.LAYER:
            input_info = \
                list(map(lambda x: InputLayerId(int(x)), entity_parameters.pop("input").split(",")))
            layers.append(Layer(LayerId(layer_id),
                                layer_type=LayerType(name),
                                layer_parameters=entity_parameters))
            layers_connections[LayerId(layer_id)] = input_info
            layer_id += 1
        elif entity == Entity.PROJECTOR_LAYER:
            input_info = \
                list(map(lambda x: InputLayerId(int(x)), entity_parameters.pop("input").split(",")))
            projector_layers.append(Layer(LayerId(projector_layer_id),
                                          layer_type=LayerType(name),
                                          layer_parameters=entity_parameters))
            projector_layers_connections[LayerId(projector_layer_id)] = input_info
            projector_layer_id += 1
        elif entity == Entity.OPTIMISER:
            optimiser = Optimiser(optimiser_type=OptimiserType(name),
                                  optimiser_parameters=entity_parameters)
        elif entity == Entity.PRETEXT_TASK:
            pretext_task = Pretext(pretext_type=PretextType(name),
                                   pretext_parameters=entity_parameters)

    return ParsedNetwork(layers, layers_connections), \
        ParsedNetwork(projector_layers, projector_layers_connections), \
        optimiser, \
        pretext_task
