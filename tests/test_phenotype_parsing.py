from typing import Dict, List
import unittest

from fast_denser.misc.enums import LayerType, OptimiserType
from fast_denser.misc.phenotype_parser import Layer, Optimiser, parse_phenotype

class Test(unittest.TestCase):

    def setUp(self):
        phenotype1: str = \
            "layer:fc act:linear out_features:1142 bias:False input:-1 " \
            "layer:fc act:softmax out_features:10 bias:True input:0 " \
            "learning:adam lr:0.07889346277843777 beta1:0.5469297933871174 beta2:0.5141737382610032 " \
            "weight_decay:0.0008359293388159499 early_stop:5 batch_size:406 epochs:10000"
        phenotype2: str = \
            "layer:dropout rate:0.18124172520507434 input:-1 " \
            "layer:fc act:softmax out_features:10 bias:True input:0 " \
            "learning:rmsprop lr:0.09678321949252514 alpha:0.6790246873474941 " \
            "weight_decay:0.0008917689991608618 early_stop:9 batch_size:194 epochs:10000"
        phenotype3: str = \
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
        self.phenotypes: List[str] = [phenotype1, phenotype2, phenotype3]


    def test_parse_phenotype(self):

        expected_optimisers: List[Optimiser] = \
            [
                Optimiser(OptimiserType.ADAM,
                          {"lr": "0.07889346277843777",
                           "beta1": "0.5469297933871174",
                           "beta2": "0.5141737382610032",
                           "weight_decay": "0.0008359293388159499",
                           "early_stop": "5",
                           "batch_size": "406",
                           "epochs": "10000"}),
                Optimiser(OptimiserType.RMSPROP,
                          {"lr": "0.09678321949252514",
                           "alpha": "0.6790246873474941",
                           "weight_decay": "0.0008917689991608618",
                           "early_stop": "9",
                           "batch_size": "194",
                           "epochs": "10000"}),
                Optimiser(OptimiserType.GRADIENT_DESCENT,
                          {"lr": "0.021251395366932026",
                           "momentum": "0.7079581891606233",
                           "weight_decay": "0.00025234740040252823",
                           "nesterov": "False",
                           "early_stop": "18",
                           "batch_size": "292",
                           "epochs": "10000"})
            ]
        expected_layers: List[List[Layer]] = [
            [
                Layer(layer_id=0,
                      layer_type=LayerType.FC,
                      layer_parameters={'act':'linear', 'out_features':'1142', 'bias':'False'}),
                Layer(layer_id=1,
                      layer_type=LayerType.FC,
                      layer_parameters={'act':'softmax', 'out_features':'10', 'bias':'True'}),
            ],
            [
                Layer(layer_id=0,
                      layer_type=LayerType.DROPOUT,
                      layer_parameters={'rate': '0.18124172520507434'}),
                Layer(layer_id=1,
                      layer_type=LayerType.FC,
                      layer_parameters={'act':'softmax', 'out_features':'10', 'bias':'True'})
            ],
            [
                Layer(layer_id=0,
                      layer_type=LayerType.DROPOUT,
                      layer_parameters={'rate': '0.48283254514084895'}),
                Layer(layer_id=1,
                      layer_type=LayerType.BATCH_NORM,
                      layer_parameters={}),
                Layer(layer_id=2,
                      layer_type=LayerType.POOL_MAX,
                      layer_parameters={'kernel_size': '2', 'stride': '1', 'padding': 'valid'}),
                Layer(layer_id=3,
                      layer_type=LayerType.POOL_MAX,
                      layer_parameters={'kernel_size': '5', 'stride': '3', 'padding': 'valid'}),
                Layer(layer_id=4,
                      layer_type=LayerType.POOL_MAX,
                      layer_parameters={'kernel_size': '2', 'stride': '1', 'padding': 'valid'}),
                Layer(layer_id=5,
                      layer_type=LayerType.POOL_AVG,
                      layer_parameters={'kernel_size': '3', 'stride': '2', 'padding': 'same'}),
                Layer(layer_id=6,
                      layer_type=LayerType.DROPOUT,
                      layer_parameters={'rate': '0.2804266331697851'}),
                Layer(layer_id=7,
                      layer_type=LayerType.DROPOUT,
                      layer_parameters={'rate': '0.49058403604972123'}),
                Layer(layer_id=8,
                      layer_type=LayerType.DROPOUT,
                      layer_parameters={'rate': '0.6968873473903937'}),
                Layer(layer_id=9,
                      layer_type=LayerType.FC,
                      layer_parameters={'act':'softmax', 'out_features':'10', 'bias':'True'}),
            ]
        ]
        expected_layers_connections: List[Dict[int, List[int]]] = [
            {1: [0], 0: [-1]},
            {1: [0], 0: [-1]},
            {9: [8], 8: [7], 7: [6], 6: [5], 5: [4, 2, 3], 4: [3], 3: [2, 0, 1], 2: [1, 0], 1: [0, -1], 1: [0, -1], 0: [-1]}
        ]

        layers: List[Layer]
        optimiser: Optimiser
        layers_connections: Dict[int, List[int]]

        for i in range(len(self.phenotypes)):
            layers, layers_connections, optimiser = parse_phenotype(self.phenotypes[i])

            self.assertEqual(optimiser, expected_optimisers[i])
            self.assertEqual(layers, expected_layers[i])
            self.assertEqual(layers_connections, expected_layers_connections[i])


if __name__ == '__main__':
    unittest.main()
