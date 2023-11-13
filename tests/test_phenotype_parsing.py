from typing import Any, Dict, List, Optional
from parameterized import parameterized
import unittest

from evodenss.misc.enums import LayerType, OptimiserType, PretextType
from evodenss.misc.utils import LayerId
from evodenss.misc.phenotype_parser import Layer, Optimiser, ParsedNetwork, Pretext, parse_phenotype

optimiser1 = Optimiser(OptimiserType.ADAM, {})

class Test(unittest.TestCase):

    @staticmethod
    def build_string(layers: List[Dict[str, str]],
                     inputs: List[List[str]],
                     optimiser,
                     prefix=''):
        final_string: str = ""
        for l, i in zip(layers, inputs):
            if l['layer'] == "conv":
                final_string += f"{prefix}layer:{l['layer']} act:{l['act']} out_features:{l['out_features']} bias:{l['bias']} input:{','.join(i)}"
            elif l['layer'] == "batch_norm" or l['layer'] == "identity":
                final_string += f"{prefix}layer:{l['layer']} input:{','.join(i)}"
            elif l['layer'] == "dropout":
                final_string += f"{prefix}layer:{l['layer']} rate:{l['rate']} input:{','.join(i)}"
            elif l['layer'] == "fc":
                final_string += f"{prefix}layer:{l['layer']} act:{l['act']} out_features:{l['out_features']} bias:{l['bias']} input:{','.join(i)}"

        if optimiser['opt'] == 'adam':
            final_string += f""
        elif optimiser['opt'] == 'adam':
            final_string += f""
        elif optimiser['opt'] == 'adam':
            final_string += f""

        return final_string

    def setUp(self) -> None:
        phenotype1: str = \
            "layer:fc act:linear out_features:1142 bias:False input:-1 " \
            "layer:fc act:softmax out_features:10 bias:True input:0 " \
            "learning:adam lr:0.07889346277843777 beta1:0.5469297933871174 beta2:0.5141737382610032 " \
            "weight_decay:0.0008359293388159499 early_stop:5 batch_size:406 epochs:10000"
        phenotype2: str = \
            "layer:dropout rate:0.48283254514084895 input:-1 " \
            "layer:batch_norm input:0,-1 " \
            "layer:pool_max kernel_size:2 stride:1 padding:valid input:1,0 " \
            "layer:pool_max kernel_size:5 stride:3 padding:valid input:2,0,1 " \
            "layer:pool_max kernel_size:2 stride:1 padding:valid input:3 " \
            "layer:pool_avg kernel_size:3 stride:2 padding:same input:4,2,3 " \
            "layer:dropout rate:0.2804266331697851 input:5 " \
            "layer:dropout rate:0.49058403604972123 input:6 " \
            "layer:dropout rate:0.6968873473903937 input:7 " \
            "layer:fc act:softmax out_features:10 bias:True input:8 " \
            "learning:gradient_descent lr:0.021251395366932026 momentum:0.7079581891606233 " \
            "weight_decay:0.00025234740040252823 nesterov:False early_stop:18 batch_size:292 epochs:10000"
        phenotype3: str = \
            "layer:conv out_channels:128 kernel_size:2 stride:2 padding:same act:sigmoid bias:False input:-1 " \
            "layer:batch_norm input:0 " \
            "projector_layer:fc act:linear out_features:1024 bias:True input:-1 " \
            "projector_layer:fc act:relu out_features:1024 bias:True input:0 " \
            "projector_layer:identity input:1 " \
            "learning:lars lr_weights:0.10604385506377026 lr_biases:0.001 momentum:0.8158013267718259 " \
            "weight_decay:1e-07 batch_size:256 epochs:50 " \
            "pretext:bt lamb:0.03125"
        self.phenotypes: List[str] = [phenotype1, phenotype2, phenotype3]

        self.optimiser1_params: Dict[str, Any] = {}
        self.optimiser2_params: Dict[str, Any] = {}
        self.optimiser3_params: Dict[str, Any] = {}
        self.optimiser4_params: Dict[str, Any] = {}


    def test_parse_phenotype1(self) -> None:
        expected_optimiser: Optimiser = \
                Optimiser(OptimiserType.ADAM,
                          {"lr": "0.07889346277843777",
                           "beta1": "0.5469297933871174",
                           "beta2": "0.5141737382610032",
                           "weight_decay": "0.0008359293388159499",
                           "early_stop": "5",
                           "batch_size": "406",
                           "epochs": "10000"})
        expected_layers: List[Layer] = [
            Layer(LayerId(0), LayerType.FC, {'act':'linear', 'out_features':'1142', 'bias':'False'}),
            Layer(LayerId(1), LayerType.FC, {'act':'softmax', 'out_features':'10', 'bias':'True'}),
        ]
        expected_layers_connections: Dict[int, List[int]] = {1: [0], 0: [-1]}
            
        optimiser: Optimiser
        parsed_network: ParsedNetwork
        proj_parsed_network: ParsedNetwork
        pretext_task: Optional[Pretext]
        parsed_network, proj_parsed_network, optimiser, pretext_task = \
            parse_phenotype(self.phenotypes[0])

        self.assertEqual(optimiser, expected_optimiser)
        self.assertEqual(parsed_network.layers, expected_layers)
        self.assertEqual(parsed_network.layers_connections, expected_layers_connections)
        self.assertEqual(proj_parsed_network.layers, [])
        self.assertEqual(proj_parsed_network.layers_connections, {})
        self.assertIsNone(pretext_task)

    def test_parse_phenotype2(self) -> None:
        expected_optimiser: Optimiser = \
                Optimiser(OptimiserType.GRADIENT_DESCENT,
                          {"lr": "0.021251395366932026",
                           "momentum": "0.7079581891606233",
                           "weight_decay": "0.00025234740040252823",
                           "nesterov": "False",
                           "early_stop": "18",
                           "batch_size": "292",
                           "epochs": "10000"})
        expected_layers: List[Layer] = [
            Layer(LayerId(0), LayerType.DROPOUT, {'rate': '0.48283254514084895'}),
            Layer(LayerId(1), LayerType.BATCH_NORM, {}),
            Layer(LayerId(2), LayerType.POOL_MAX, {'kernel_size': '2', 'stride': '1', 'padding': 'valid'}),
            Layer(LayerId(3), LayerType.POOL_MAX, {'kernel_size': '5', 'stride': '3', 'padding': 'valid'}),
            Layer(LayerId(4), LayerType.POOL_MAX, {'kernel_size': '2', 'stride': '1', 'padding': 'valid'}),
            Layer(LayerId(5), LayerType.POOL_AVG, {'kernel_size': '3', 'stride': '2', 'padding': 'same'}),
            Layer(LayerId(6), LayerType.DROPOUT, {'rate': '0.2804266331697851'}),
            Layer(LayerId(7), LayerType.DROPOUT, {'rate': '0.49058403604972123'}),
            Layer(LayerId(8), LayerType.DROPOUT, {'rate': '0.6968873473903937'}),
            Layer(LayerId(9), LayerType.FC, {'act':'softmax', 'out_features':'10', 'bias':'True'})
        ]
        expected_layers_connections: Dict[int, List[int]] = \
            {9: [8], 8: [7], 7: [6], 6: [5], 5: [4, 2, 3], 4: [3], 3: [2, 0, 1], 2: [1, 0], 1: [0, -1], 1: [0, -1], 0: [-1]}
            
        optimiser: Optimiser
        parsed_network: ParsedNetwork
        proj_parsed_network: ParsedNetwork
        pretext_task: Optional[Pretext]
        parsed_network, proj_parsed_network, optimiser, pretext_task = \
            parse_phenotype(self.phenotypes[1])

        self.assertEqual(optimiser, expected_optimiser)
        self.assertEqual(parsed_network.layers, expected_layers)
        self.assertEqual(parsed_network.layers_connections, expected_layers_connections)
        self.assertEqual(proj_parsed_network.layers, [])
        self.assertEqual(proj_parsed_network.layers_connections, {})
        self.assertIsNone(pretext_task)

    def test_parse_phenotype3(self) -> None:
        expected_optimiser: Optimiser = \
                Optimiser(OptimiserType.LARS,
                          {"lr_weights": "0.10604385506377026",
                           "lr_biases": "0.001",
                           "momentum": "0.8158013267718259",
                           "weight_decay": "0.0000001",
                           "batch_size": "256",
                           "epochs": "50"})
        expected_layers: List[Layer] = [
            Layer(LayerId(0), LayerType.CONV, {'out_channels':'128', 'kernel_size':'2', 'stride':'2',
                                               'padding':'same', 'act':'sigmoid', 'bias':'False'}),
            Layer(LayerId(1), LayerType.BATCH_NORM, {}),
        ]
        expected_layers_connections: Dict[int, List[int]] = {1: [0], 0: [-1]}
        expected_proj_layers: List[Layer] = [
            Layer(LayerId(0), LayerType.FC, {'act':'linear', 'out_features':'1024', 'bias':'True'}),
            Layer(LayerId(1), LayerType.FC, {'act':'relu', 'out_features':'1024', 'bias':'True'}),
            Layer(LayerId(2), LayerType.IDENTITY, {}),
        ]
        expected_proj_layers_connections: Dict[int, List[int]] = {2: [1], 1: [0], 0: [-1]}
        expected_pretext_task: Pretext = Pretext(PretextType.BT, {"lamb": "0.03125"})

        optimiser: Optimiser
        parsed_network: ParsedNetwork
        proj_parsed_network: ParsedNetwork
        pretext_task: Optional[Pretext]
        parsed_network, proj_parsed_network, optimiser, pretext_task = \
            parse_phenotype(self.phenotypes[2])

        self.assertEqual(optimiser, expected_optimiser)
        self.assertEqual(parsed_network.layers, expected_layers)
        self.assertEqual(parsed_network.layers_connections, expected_layers_connections)
        self.assertEqual(proj_parsed_network.layers, expected_proj_layers)
        self.assertEqual(proj_parsed_network.layers_connections, expected_proj_layers_connections)
        self.assertEqual(pretext_task, expected_pretext_task)

if __name__ == '__main__':
    unittest.main()
