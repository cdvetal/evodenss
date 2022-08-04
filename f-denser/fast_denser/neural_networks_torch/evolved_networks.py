from functools import reduce
import logging
from typing import cast, Dict, List, Optional, Set, Tuple

# from fast_denser.neural_networks_torch import NetworkValidator
from fast_denser.misc.enums import LayerType, ProjectorUsage
from fast_denser.misc.utils import InputLayerId, LayerId

import torch
from torch import nn, Tensor

logger = logging.getLogger(__name__)

class EvolvedNetwork(nn.Module):

    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]]) -> None:

        super(EvolvedNetwork, self).__init__()
        self.cache: Dict[Tuple[InputLayerId, LayerId], Tensor] = {}
        self.output_layer_id: LayerId = self._get_output_layer_id(layers_connections)
        self.layers_connections: Dict[LayerId, List[InputLayerId]] = layers_connections
        self.id_layername_map: Dict[LayerId, str] = {LayerId(i): l[0] for i, l in enumerate(evolved_layers)}

        for (layer_name, layer) in evolved_layers:
            setattr(self, layer_name, layer)


    # It gets the layer id that corresponds to the final/output layer
    def _get_output_layer_id(self, layers_connections: Dict[LayerId, List[InputLayerId]]) -> LayerId:
        keyset: Set[int] = set(layers_connections.keys())
        values_set: Set[int] = set(
            list(reduce(lambda a, b: cast(list, a) + cast(list, b), layers_connections.values()))
        )
        result: Set[int] = keyset.difference(values_set)
        assert len(result) == 1
        return LayerId(list(result)[0])


    def _clear_cache(self) -> None:
        self.cache.clear()


    def _process_forward_pass(self,
                              x: Tensor,
                              layer_id: LayerId,
                              input_ids: List[InputLayerId]) -> Tensor:
        
        assert len(input_ids) > 0
        layer_output: Tensor
        input_tensor: Tensor
        layer_outputs: List[Tensor] = []
        layer_name: str = self.id_layername_map[layer_id]
        for i in input_ids:
            if i == -1:
                input_tensor = x
            else:
                input_tensor = self._process_forward_pass(x, LayerId(i), self.layers_connections[LayerId(i)])
            if (i, layer_id) in self.cache.keys():
                layer_output = self.cache[(i, layer_id)]
            else:
                layer_output = self.__dict__['_modules'][layer_name](input_tensor)
                self.cache[(i, layer_id)] = layer_output

            #print(f"Layer being processed {layer_id}. OUT SHAPE: {layer_output.shape}")
            layer_outputs.append(layer_output)

        self._clear_cache()

        if len(layer_outputs) > 1:
            return torch.cat(tuple(layer_outputs))
        return layer_outputs[0]

    def forward(self, x: Tensor) -> Optional[Tensor]:
        input_layer_ids: List[InputLayerId] = self.layers_connections[self.output_layer_id]
        return self._process_forward_pass(x, self.output_layer_id, input_layer_ids)


class LegacyNetwork(EvolvedNetwork):

    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]]) -> None:
        super().__init__(evolved_layers, layers_connections)

    def forward(self, x: Tensor) -> Optional[Tensor]:
        return super(LegacyNetwork, self).forward(x)


class BarlowTwinsNetwork(EvolvedNetwork):

    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 projector_network_usage: ProjectorUsage):
        super(BarlowTwinsNetwork, self).__init__(evolved_layers, layers_connections)
        self.bn = nn.BatchNorm1d(10, affine=False)
        self.projector: Optional[nn.Sequential]
        if projector_network_usage == ProjectorUsage.EXPLICIT:
            self.projector = nn.Sequential(
                nn.Linear(10, 8192, bias=False),
                nn.BatchNorm1d(8192),
                nn.ReLU(inplace=True),
                nn.Linear(8192, 8192, bias=False)
            )
        elif projector_network_usage == ProjectorUsage.IMPLICIT:
            raise NotImplementedError(f"projector_network_usage cannot be implicit yet")
        else:
            self.projector = None


    def forward(self, x1: Tensor, x2: Tensor=None, batch_size: int=None) -> Optional[Tensor]:
        if x2 is not None:
            assert batch_size is not None
            y1 = super(BarlowTwinsNetwork, self).forward(x1)
            y2 = super(BarlowTwinsNetwork, self).forward(x2)

            if y1 is None or y2 is None:
                return None

            z1 = y1#self.projector(y1)
            z2 = y2#self.projector(y2)

            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(batch_size)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = self._off_diagonal(c).pow_(2).sum()
            loss: Tensor = on_diag + 0.1 * off_diag
            return loss
        else:
            assert batch_size is None
            return super(BarlowTwinsNetwork, self).forward(x1)

        '''
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(8, 8))
        cols, rows = 2, 1
        print("y1: ", y1)
        print("y2: ", y2)
        figure.add_subplot(rows, cols, 1)
        plt.axis("off")
        plt.imshow(x1.squeeze())
        figure.add_subplot(rows, cols, 2)
        plt.axis("off")
        plt.imshow(x2.squeeze())
        plt.show()
        '''

    def _off_diagonal(self, x: Tensor) -> Tensor:
        # return a flattened view of the off-diagonal elements of a square matrix
        n: int
        m: int
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

