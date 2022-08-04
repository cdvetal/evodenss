from itertools import takewhile, dropwhile
from typing import Any, Dict, List, Optional, Tuple, Union

from fast_denser.misc.enums import Entity, LayerType, OptimiserType
from fast_denser.misc.utils import InputLayerId, LayerId

class Layer:

    def __init__(self,
                 layer_id: LayerId,
                 layer_type: LayerType,
                 layer_parameters: Dict[str, str]) -> None:
        self.layer_id: LayerId = layer_id
        self.layer_type: LayerType = layer_type
        self.layer_parameters: Dict[str, Any] = dict(self._convert(k, v) for k,v in layer_parameters.items())

    def _convert(self, key: str, value: str) -> Tuple[str, Any]:
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
        else:
            raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Layer):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        return f"Layer [{self.layer_type}] with id [{self.layer_id}] and params: {self.layer_parameters}"


class Optimiser:

    def __init__(self,
                 optimiser_type: OptimiserType,
                 optimiser_parameters: Dict[str, str]) -> None:
        self.optimiser_type: OptimiserType = optimiser_type
        self.optimiser_parameters: Dict[str, Any] = {
            k: self._convert(k, v) for k,v in optimiser_parameters.items()
        }

    def _convert(self, key: str, value: str) -> Any:
        if key == "nesterov":
            return value.title() == "True"
        elif key in ["lr", "alpha", "weight_decay", "momentum", "beta1", "beta2"]:
            return float(value)
        elif key in ["early_stop", "batch_size", "epochs"]:
            return int(value)
        else:
            raise ValueError(f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Optimiser):
            return self.__dict__ == other.__dict__
        return False

def parse_phenotype(phenotype: str) -> Tuple[List[Layer], Dict[LayerId, List[InputLayerId]], Optimiser]:

    phenotype_as_list: List[List[str]] = \
        list(map(lambda x: x.split(":"), phenotype.split(" ")))

    optimiser: Optimiser
    layers: List[Layer] = []
    layers_connections: Dict[LayerId, List[InputLayerId]] = {}
    layer_id: int = 0
    while phenotype_as_list:
        entity: Entity = Entity(phenotype_as_list[0][0])
        name: str = phenotype_as_list[0][1]
        entity_parameters: Dict[str, str] = {
            kv[0]: kv[1]
            for kv in takewhile(lambda kv: kv[0] not in Entity.enum_values(),
                                phenotype_as_list[1:])
        }
        phenotype_as_list = list(dropwhile(lambda kv: kv[0] not in Entity.enum_values(),
                                           phenotype_as_list[1:]))
        if entity == Entity.LAYER:
            input_info: List[InputLayerId] = \
                list(map(lambda x: InputLayerId(int(x)), entity_parameters.pop("input").split(",")))
            layers.append(Layer(LayerId(layer_id),
                                layer_type=LayerType(name),
                                layer_parameters=entity_parameters))
            layers_connections[LayerId(layer_id)] = input_info
            layer_id += 1
        elif entity == Entity.OPTIMISER:
            optimiser = Optimiser(optimiser_type=OptimiserType(name),
                                  optimiser_parameters=entity_parameters)
    return layers, layers_connections, optimiser
