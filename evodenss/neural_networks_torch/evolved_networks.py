from functools import reduce
import logging
from sys import float_info
from typing import cast, Dict, List, Optional, Set, Tuple, Union

# from evodenss.neural_networks_torch import NetworkValidator
from evodenss.neural_networks_torch import Dimensions
from evodenss.misc.constants import CHANNEL_INDEX, SEPARATOR_CHAR
from evodenss.misc.enums import Device, LayerType, ProjectorUsage
from evodenss.misc.utils import InputLayerId, LayerId

import torch
from torch import nn, Tensor
import torch.nn.functional as F

logger = logging.getLogger(__name__)


#def find_last_layer_info(layer_names: List[str]) -> Tuple[str, int]:
#    linear_layers: List[str] = list(filter(lambda x: x.startswith(LayerType.FC.value) and "." not in x, layer_names))
#    ids: List[int] = list(map(lambda x: int(x.split("_")[-1].split(".")[0]), linear_layers))
#    return LayerType.FC.value, max(ids)

#def find_last_layer_info(layer_names: List[str]) -> Tuple[LayerType, int]:
#    print(layer_names)
#    #linear_layers: List[str] = list(filter(lambda x: x.startswith(expected_last_layertype) and "." not in x, layer_names))
#    linear_layers: List[str] = list(filter(lambda x: x.startswith(LayerType.FC.value) and "." not in x, layer_names))
#    print(linear_layers)
#    ids: List[int] = list(map(lambda x: int(x.split(SEPARATOR_CHAR)[-1].split(".")[0]), linear_layers))
#    print(ids)
#    return LayerType.FC, max(ids)

def find_last_layer_info(evolved_layers: List[Tuple[str, nn.Module]],
                         last_layer_name: str) -> Tuple[LayerType, int]:
    last_layer_name: str = id_
    print(layer_names)
    #linear_layers: List[str] = list(filter(lambda x: x.startswith(expected_last_layertype) and "." not in x, layer_names))
    linear_layers: List[str] = list(filter(lambda x: x.startswith(LayerType.FC.value) and "." not in x, layer_names))
    print(linear_layers)
    ids: List[int] = list(map(lambda x: int(x.split(SEPARATOR_CHAR)[-1].split(".")[0]), linear_layers))
    print(ids)
    return LayerType.FC, max(ids)

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
        final_input_tensor: Tensor
        input_tensor: Tensor
        output_tensor: Tensor
        # layer_outputs: List[Tensor] = []
        layer_name: str = self.id_layername_map[layer_id]
        layer_inputs = []
        for i in input_ids:
            if i == -1:
                input_tensor = x
                #print("---------- (end) processing layer: ", layer_id, input_tensor.shape)
            else:
                if (i, layer_id) in self.cache.keys():
                    input_tensor = self.cache[(i, layer_id)]
                else:
                    input_tensor = self._process_forward_pass(x, LayerId(i), self.layers_connections[LayerId(i)])
                    self.cache[(i, layer_id)] = input_tensor
                #print("---------- processing layer: ", layer_id, "---", i, "---", input_tensor.shape)
                #if layer_id==5:
                #    print("here:", input_tensor.shape)
                #    print("here:", self.__dict__['_modules'][layer_name])
            layer_inputs.append(input_tensor)        

        self._clear_cache()
        #print("length:", len(layer_inputs), input_ids, layer_id)
        #print([x.shape for x in layer_inputs])
        if len(layer_inputs) > 1:
            # we are using channels first representation, so channels is index 1
            # ADRIANO: another hack to cope with the relu in resnet scenario
            final_input_tensor = torch.stack(layer_inputs, dim=0).sum(dim=0)
            #old way: final_input_tensor = torch.cat(tuple(layer_inputs), dim=CHANNEL_INDEX)
        else:
            final_input_tensor = layer_inputs[0]
        #print("final input tensor: ", layer_id, layer_name, final_input_tensor.shape)
        output_tensor = self.__dict__['_modules'][layer_name](final_input_tensor)
        return output_tensor

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
                 layer_shapes: Dict[InputLayerId, Dimensions],
                 projector_usage: Optional[ProjectorUsage],
                 device: Device):
        
        # TODO: DO IT MORE EFFICIENTLY
        super(BarlowTwinsNetwork, self).__init__(evolved_layers, layers_connections)
        #layer_names: List[str] = list(map(lambda x: x[0], evolved_layers))
        last_layer_name: str = self.id_layername_map[self.output_layer_id]
        last_layer_type: LayerType = LayerType[last_layer_name.split(SEPARATOR_CHAR)[0].upper()]
        #last_layer: nn.Module
        #relevant_index: int
        #last_layer, relevant_index = find_last_layer_info(last_layer_name)
        #last_layer_type: LayerType
        #last_layer_count: int
        #last_layer_type, last_layer_count = find_last_layer_info(layer_names)
        #index: int = 1 if last_layer_count == 1 else 0
        #print(index)
        #print("----")
        #print(list(filter(lambda x: x[0] == f"{last_layer_type.value}{SEPARATOR_CHAR}{last_layer_count}", evolved_layers)))
        #print("----")
        #print(list(filter(lambda x: x[0] == f"{last_layer_type.value}{SEPARATOR_CHAR}{last_layer_count}", evolved_layers))[0])
        #print("----")
        #print(list(filter(lambda x: x[0] == f"{last_layer_type.value}{SEPARATOR_CHAR}{last_layer_count}", evolved_layers))[0][1])
        #last_layer = list(filter(lambda x: x[0] == f"{last_layer_type.value}{SEPARATOR_CHAR}{last_layer_count}", evolved_layers))[0][1]
        
        #self.last_layer_out_features: int
        #if last_layer_type is LayerType.FC:
        #    self.last_layer_out_features = last_layer[relevant_index].out_features
        #else:
        #    self.last_layer_out_features = layer_shapes[self.output_layer_id].flatten()
        self.last_layer_out_features = layer_shapes[InputLayerId(self.output_layer_id)].flatten()
        self.layer_shapes: Dict[InputLayerId, Dimensions] = layer_shapes
        self.projector: Optional[nn.Module]
        if projector_usage is None:
            # TODO: What if there is no projector? This has to be done at some point I think
            raise NotImplementedError()
            #self.projector = None
            #self.bn = nn.BatchNorm1d(self.last_layer_out_features, affine=False, device=device.value)
        if projector_usage == ProjectorUsage.EXPLICIT:
            # projector
            sizes: List[int] = [self.last_layer_out_features] + [512, 128]
            layers: List[nn.Module] = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False, device=device.value))
                layers.append(nn.BatchNorm1d(sizes[i + 1], device=device.value))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False, device=device.value))
            flatten: List[nn.Module] = [nn.Flatten()] if last_layer_type is not LayerType.FC else []
            self.projector = nn.Sequential(*(flatten + layers))
            self.bn = nn.BatchNorm1d(sizes[-1], affine=False, device=device.value)
        else:
            raise NotImplementedError


    def forward(self, x1: Tensor, x2: Tensor=None, batch_size: int=None) -> Optional[Union[Tuple[Tensor, Tensor, Tensor], Tensor]]:
        if x2 is not None:
            assert batch_size is not None
            y1 = super(BarlowTwinsNetwork, self).forward(x1)
            y2 = super(BarlowTwinsNetwork, self).forward(x2)

            #import matplotlib.pyplot as plt
            #figure = plt.figure(figsize=(8, 8))
            #cols, rows = 2, 1
            #print("y1: ", y1)
            #print("y2: ", y2)
            #figure.add_subplot(rows, cols, 1)
            #plt.axis("off")
            #plt.imshow(x1[0,:,:].cpu().squeeze())
            #figure.add_subplot(rows, cols, 2)
            #plt.axis("off")
            #plt.imshow(x2[0,:,:].cpu().squeeze())
            #plt.show()

            #if y1 is None or y2 is None:
            #    return None

            if self.projector is None:
                z1 = y1
                z2 = y2
            else:
                #print(y1.shape, y2.shape)
                z1 = self.projector(y1)
                z2 = self.projector(y2)


            #print(self.last_layer_out_features)
            #z1_max = torch.max(z1, dim=1,keepdim=True).values.repeat(1, self.last_layer_out_features)
            #z2_max = torch.max(z2, dim=1,keepdim=True).values.repeat(1, self.last_layer_out_features)
            #z1_min = torch.min(z1, dim=1,keepdim=True).values.repeat(1, self.last_layer_out_features)
            #z2_min = torch.min(z2, dim=1,keepdim=True).values.repeat(1, self.last_layer_out_features)
            '''
            print(z1)
            print(z2)
            print("---z_max")
            print(z1_max)
            print(z2_max)
            print("---z_min")
            print(z1_min)
            print(z2_min)
            '''
            #z1 = (z1-z1_min)/(z1_max-z1_min) * 2 - 1 
            #z2 = (z2-z2_min)/(z2_max-z2_min) * 2 - 1
            #z1_den = torch.pow(z1, 2).sum(dim=0)
            #z2_den = torch.pow(z2, 2).sum(dim=0)
            #print("den")
            #print(z1_den)
            #print(z2_den)

            #c = (z1.T @ z2)/(z1_den.T @ z2_den)
            #print(c)
            #print("---------------------------------")

            c = self.bn(z1).T @ self.bn(z2)
            c[c.isnan()] = 0.0
            valid_c = c[~c.isinf()]
            limit = 1e+30 if c.dtype == torch.float32 else 1e+4
            try:
                max_value = torch.max(valid_c)
            except RuntimeError:
                max_value = limit
            try:
                min_value = torch.min(valid_c)
            except RuntimeError:
                min_value = -limit
            c[c == float("Inf")] = max_value if max_value != 0.0 else limit
            c[c == float("-Inf")] = min_value if min_value != 0.0 else -limit
            c.div_(batch_size)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = self._off_diagonal(c).pow_(2).sum()

            loss: Tensor = (on_diag + 0.0078125 * off_diag)#.div_(self.last_layer_out_features)

            return on_diag, off_diag, loss
        else:
            # In case we use the network for inference rather than training
            assert batch_size is None
            return super(BarlowTwinsNetwork, self).forward(x1)
            

    def _off_diagonal(self, x: Tensor) -> Tensor:
        # return a flattened view of the off-diagonal elements of a square matrix
        n: int
        m: int
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class EvaluationBarlowTwinsNetwork(nn.Module):

    def __init__(self, barlow_twins_trained_model: nn.Module, n_neurons: int, device: Device) -> None:
        super(EvaluationBarlowTwinsNetwork, self).__init__()
        self.barlow_twins_trained_model: nn.Module = barlow_twins_trained_model

        # used to append the final layers to evolve networks
        output_layer_id: LayerId = barlow_twins_trained_model.output_layer_id
        last_layer_name: str = barlow_twins_trained_model.id_layername_map[output_layer_id]
        last_layer_type: LayerType = LayerType[last_layer_name.split(SEPARATOR_CHAR)[0].upper()]
        last_layer_out_features = barlow_twins_trained_model.layer_shapes[InputLayerId(output_layer_id)].flatten()
        print(last_layer_type, last_layer_out_features)

        #last_layer_out_features = getattr(self.barlow_twins_trained_model, f"{last_layer_type.value}{SEPARATOR_CHAR}{last_layer_count}")[index].out_features
        #flatten: List[nn.Module] = [nn.Flatten()] if last_layer_type is not LayerType.FC else []
        layers = [] if last_layer_type is LayerType.FC else [nn.Flatten()]
        self.final_layer = nn.Sequential(
            *layers,
            nn.Linear(in_features=last_layer_out_features, out_features=n_neurons, bias=True, device=device.value)
        )
        self.relevant_index = len(layers)
        #this was used to inject the original resnet network
        #self.final_layer = nn.Linear(in_features=2048, out_features=10, bias=True, device=device.value) # remove me later and keep the line above
        
        self.softmax = nn.Softmax()
        self.barlow_twins_trained_model.requires_grad_(False)
        # self.barlow_twins_trained_model.fc.requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        embs = self.barlow_twins_trained_model.forward(x)
        assert isinstance(embs, Tensor)
        y: Tensor = self.final_layer(embs)
        return self.softmax(y)