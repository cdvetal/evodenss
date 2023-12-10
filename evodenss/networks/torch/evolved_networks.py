import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
import torch.utils.checkpoint as checkpoint

from evodenss.networks import Dimensions
from evodenss.misc.constants import SEPARATOR_CHAR
from evodenss.misc.enums import Device, LayerType
from evodenss.misc.utils import InputLayerId, LayerId


logger = logging.getLogger(__name__)


class EvolvedNetwork(nn.Module):

    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 output_layer_id: LayerId) -> None:

        super().__init__()
        self.cache: Dict[Tuple[InputLayerId, LayerId], Tensor] = {}
        self.evolved_layers: List[Tuple[str, nn.Module]] = evolved_layers
        self.layers_connections: Dict[LayerId, List[InputLayerId]] = layers_connections
        self.output_layer_id: LayerId = output_layer_id
        self.id_layername_map: Dict[LayerId, str] = {LayerId(i): l[0] for i, l in enumerate(evolved_layers)}

        for (layer_name, layer) in evolved_layers:
            setattr(self, layer_name, layer)


    def _clear_cache(self) -> None:
        self.cache.clear()
        torch.cuda.empty_cache()


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

        del input_tensor
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
        del layer_inputs
        #print("final input tensor: ", layer_id, layer_name, final_input_tensor.shape)
        # checkpoint convolutional layers as they are the ones that seem
        # to bring more memory issues evaluating individuals
        if LayerType.CONV.value in layer_name or LayerType.FC.value in layer_name or \
            LayerType.IDENTITY.value in layer_name or \
            LayerType.POOL_MAX.value in layer_name or \
            LayerType.POOL_AVG.value in layer_name:
            output_tensor = checkpoint.checkpoint(
                self.custom(self.__dict__['_modules'][layer_name]),
                final_input_tensor
            )
        else:
            output_tensor = self.__dict__['_modules'][layer_name](final_input_tensor)
        return output_tensor

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, x: Tensor) -> Optional[Tensor]:
        input_layer_ids: List[InputLayerId]
        input_layer_ids = self.layers_connections[self.output_layer_id]
        return self._process_forward_pass(x, self.output_layer_id, input_layer_ids)


class LegacyNetwork(EvolvedNetwork):

    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 output_layer_id: LayerId) -> None:
        super().__init__(evolved_layers, layers_connections, output_layer_id)

    def forward(self, x: Tensor) -> Optional[Tensor]:
        return super().forward(x)


class BarlowTwinsNetwork(EvolvedNetwork):

    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 output_layer_id: LayerId,
                 layer_shapes: Dict[InputLayerId, Dimensions],
                 projector_layer_shapes: Dict[InputLayerId, Dimensions],
                 projector_output_layer_id: LayerId,
                 projector_model: nn.Module,
                 device: Device,
                 lamb: float):

        super().__init__(evolved_layers, layers_connections, output_layer_id)
        last_layer_name: str = self.id_layername_map[self.output_layer_id]

        # This is needed in the downstream stage, Pylint should ignore it!
        # pylint: disable=unused-variable
        last_layer_type: LayerType = LayerType[last_layer_name.split(SEPARATOR_CHAR)[0].upper()]
        self.lamb: float = lamb
        self.last_layer_out_features = layer_shapes[InputLayerId(self.output_layer_id)].flatten()
        self.layer_shapes: Dict[InputLayerId, Dimensions] = layer_shapes
        self.projector_model: nn.Module = projector_model

        self.bn = nn.BatchNorm1d(projector_layer_shapes[InputLayerId(projector_output_layer_id)].flatten(),
                                 affine=False,
                                 device=device.value)

    def forward_encoder(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        y1 = super().forward(x1)
        y2 = super().forward(x2)
        return y1, y2

    # pylint: disable=arguments-renamed,
    def forward(self,
                x1: Tensor,
                x2: Optional[Tensor]=None,
                batch_size: Optional[int]=None) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        if x2 is not None:
            assert batch_size is not None
            y1 = super().forward(x1)
            y2 = super().forward(x2)

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


            #print(y1.shape, y2.shape)
            #print("projector:")
            z1 = self.projector_model(y1)
            z2 = self.projector_model(y2)

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

            # loss: Tensor = (on_diag + self.lamb * off_diag).div_(self.last_layer_out_features)
            loss: Tensor = on_diag + self.lamb * off_diag

            return on_diag, off_diag, loss
        else:
            # In case we use the network for inference rather than training
            assert batch_size is None
            return super().forward(x1)


    def _off_diagonal(self, x: Tensor) -> Tensor:
        # return a flattened view of the off-diagonal elements of a square matrix
        n: int
        m: int
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class EvaluationBarlowTwinsNetwork(nn.Module):

    def __init__(self, barlow_twins_trained_model: nn.Module, n_neurons: int, device: Device) -> None:
        super().__init__()
        self.barlow_twins_trained_model: nn.Module = barlow_twins_trained_model

        # used to append the final layers to evolve networks
        output_layer_id: LayerId = barlow_twins_trained_model.output_layer_id
        last_layer_name: str = barlow_twins_trained_model.id_layername_map[output_layer_id]
        last_layer_type: LayerType = LayerType[last_layer_name.split(SEPARATOR_CHAR)[0].upper()]
        last_layer_out_features = barlow_twins_trained_model.layer_shapes[InputLayerId(output_layer_id)].flatten()
        print(last_layer_type, last_layer_out_features)

        layers = [] if last_layer_type is LayerType.FC else [nn.Flatten()]
        self.final_layer = nn.Sequential(
            *layers,
            nn.Linear(in_features=last_layer_out_features, out_features=n_neurons, bias=True, device=device.value)
        )
        self.relevant_index = len(layers)

        self.softmax = nn.Softmax()
        self.barlow_twins_trained_model.requires_grad_(False)
        # self.barlow_twins_trained_model.fc.requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        embs = self.barlow_twins_trained_model.forward(x)
        assert isinstance(embs, Tensor)
        y: Tensor = self.final_layer(embs)
        return self.softmax(y)
