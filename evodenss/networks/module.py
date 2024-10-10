import random
from typing import cast, TYPE_CHECKING

from evodenss.config.pydantic import ModuleConfig
from evodenss.misc.utils import InputLayerId, LayerId

if TYPE_CHECKING:
    from evodenss.evolution.genotype import Genotype
    from evodenss.evolution.grammar import Grammar


class Module:

    def __init__(self,
                 module_configuration: ModuleConfig,
                 grammar: 'Grammar',
                 reuse_layer: float,
                 **attributes_to_override: list[int]|list[float]) -> None:
        self.module_name: str = module_configuration.name
        self.module_configuration: ModuleConfig = module_configuration
        num_expansions: int = random.randint(module_configuration.network_structure_init.min_expansions,
                                             module_configuration.network_structure_init.max_expansions)
        self.layers: list[Genotype] = \
            self._initialise_layers(num_expansions, grammar, reuse_layer, **attributes_to_override)
        self.connections: dict[LayerId, list[InputLayerId]] = self._initialise_connections(num_expansions)

    def _initialise_layers(self,
                           num_expansions: int,
                           grammar: 'Grammar',
                           reuse: float,
                           **attributes_to_override: list[int]|list[float]) -> list['Genotype']:
        layers: list[Genotype] = []
        #Initialise layers
        for idx in range(num_expansions):
            if idx>0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                layers.append(layers[r_idx])
            else:
                layers.append(grammar.initialise(self.module_name, **attributes_to_override))
        return layers

    def _initialise_connections(self, num_expansions: int) -> dict[LayerId, list[InputLayerId]]:
        levels_back: int
        # Initialise connections: feed-forward and allowing skip-connections
        connections: dict[LayerId, list[InputLayerId]] = {}
        for layer_idx in range(num_expansions):
            if layer_idx == 0:
                # the -1 layer is the input
                connections[LayerId(layer_idx)] = [InputLayerId(-1)]
            else:
                if self.module_configuration.levels_back is None:
                    levels_back = layer_idx + 1
                else:
                    levels_back = self.module_configuration.levels_back
                connection_possibilities: list[InputLayerId] = \
                    cast(list[InputLayerId], list(range(max(-1, layer_idx-levels_back), layer_idx-1)))
                sample_size: int = random.randint(0, len(connection_possibilities))

                connections[LayerId(layer_idx)] = [InputLayerId(layer_idx-1)]
                if sample_size > 0:
                    connections[LayerId(layer_idx)] += random.sample(connection_possibilities, sample_size)
        return connections

    def count_layers(self) -> int:
        return len(self.layers)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Module):
            return self.__dict__ == other.__dict__
        return False
