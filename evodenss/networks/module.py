from dataclasses import dataclass
import random
from typing import Dict, List, TYPE_CHECKING

from evodenss.networks.module_config import ModuleConfig

if TYPE_CHECKING:
    from evodenss.evolution.grammar import Genotype, Grammar


class Module:

    def __init__(self, module_name: str, module_configuration: ModuleConfig) -> None:
        self.module_name: str = module_name
        self.module_configuration: ModuleConfig = module_configuration
        self.layers: List[Genotype] = []
        self.connections: Dict[int, List[int]] = {}

    def initialise(self, grammar: 'Grammar', reuse: float) -> None:
        num_expansions = random.choice(self.module_configuration.initial_network_structure)

        #Initialise layers
        for idx in range(num_expansions):
            if idx>0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.module_name))

        #Initialise connections: feed-forward and allowing skip-connections
        self.connections = {} 

        for layer_idx in range(num_expansions):
            if layer_idx == 0:
                #the -1 layer is the input
                self.connections[layer_idx] = [-1,]
            else:
                connection_possibilities = list(range(max(0, layer_idx-self.module_configuration.levels_back), layer_idx-1))
                if len(connection_possibilities) < self.module_configuration.levels_back-1:
                    connection_possibilities.append(-1)

                sample_size = random.randint(0, len(connection_possibilities))

                self.connections[layer_idx] = [layer_idx-1]
                if sample_size > 0:
                    self.connections[layer_idx] += random.sample(connection_possibilities, sample_size)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Module):
            return self.__dict__ == other.__dict__
        return False