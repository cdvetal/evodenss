import logging
from typing import OrderedDict
import unittest

import torch
from torch import Size, Tensor, nn, optim

import evodenss
from evodenss.misc.enums import Device, LayerType, OptimiserType, PretextType
from evodenss.misc.utils import InputLayerId, LayerId
from evodenss.networks.phenotype_parser import Layer, Optimiser, ParsedNetwork, Pretext
from evodenss.train.learning_parameters import LearningParams
from evodenss.networks.evaluators import BarlowTwinsEvaluator, LegacyEvaluator
from evodenss.networks.model_builder import ModelBuilder


class Test(unittest.TestCase):

    logging.setLogRecordFactory(evodenss.logger_record_factory(0))
    logger: logging.Logger = logging.getLogger(__name__)

    def test_assemble_optimiser_1(self) -> None:
        fake_params: list[Tensor] = [torch.tensor([0.859, -0.0032], requires_grad=True)]
        optimiser: Optimiser = Optimiser(
            OptimiserType.ADAM,
            {"lr": "0.07889346277843777",
            "beta1": "0.5469297933871174",
            "beta2": "0.5141737382610032",
            "weight_decay": "0.0008359293388159499",
            "early_stop": "5",
            "batch_size": "406",
            "epochs": "10000"}
        )
        expected_learning_params: LearningParams = \
            LearningParams(
                early_stop=5,
                batch_size=406,
                epochs=10000,
                torch_optimiser=optim.Adam(
                    params=fake_params,
                    lr=0.07889346277843777,
                    betas=(0.5469297933871174, 0.5141737382610032),
                    weight_decay=0.0008359293388159499
                )
            )
        self.assertEqual(ModelBuilder.assemble_optimiser(fake_params, optimiser),
                         expected_learning_params)


    def test_assemble_optimiser_2(self) -> None:
        fake_params: list[Tensor] = [torch.tensor([0.859, -0.0032], requires_grad=True)]
        optimiser: Optimiser = Optimiser(
            OptimiserType.RMSPROP,
            {"lr": "0.09678321949252514",
             "alpha": "0.6790246873474941",
             "weight_decay": "0.0008917689991608618",
             "early_stop": "9",
             "batch_size": "194",
             "epochs": "10000"}
        )
        expected_learning_params: LearningParams = \
            LearningParams(
                early_stop=9,
                batch_size=194,
                epochs=10000,
                torch_optimiser=optim.RMSprop(
                    params=fake_params,
                    lr=0.09678321949252514,
                    alpha=0.6790246873474941,
                    weight_decay=0.0008917689991608618
                ),
            )
        self.assertEqual(ModelBuilder.assemble_optimiser(fake_params, optimiser),
                         expected_learning_params)


    def test_assemble_optimiser_3(self) -> None:
        fake_params: list[Tensor] = [torch.tensor([0.859, -0.0032], requires_grad=True)]
        optimiser: Optimiser = Optimiser(
            OptimiserType.GRADIENT_DESCENT,
            {"lr": "0.021251395366932026",
            "momentum": "0.7079581891606233",
            "weight_decay": "0.00025234740040252823",
            "nesterov": "False",
            "early_stop": "18",
            "batch_size": "292",
            "epochs": "10000"}
        )
        expected_learning_params: LearningParams = \
            LearningParams(
                early_stop=18,
                batch_size=292,
                epochs=10000,
                torch_optimiser=optim.SGD(
                    params=fake_params,
                    lr=0.021251395366932026,
                    momentum=0.7079581891606233,
                    weight_decay=0.00025234740040252823,
                    nesterov=False
                )
            )
        self.assertEqual(ModelBuilder.assemble_optimiser(fake_params, optimiser),
                         expected_learning_params)


    def test_assemble_network_1(self) -> None:
        layers: list[Layer] = [
            Layer(layer_id=LayerId(0),
                  layer_type=LayerType.CONV,
                  layer_parameters={'out_channels': '6', 'kernel_size': '5', 'stride': '1',
                                    'padding': 'same', 'act': 'relu'}),
            Layer(layer_id=LayerId(1),
                  layer_type=LayerType.BATCH_NORM,
                  layer_parameters={}),
            Layer(layer_id=LayerId(2),
                  layer_type=LayerType.DROPOUT,
                  layer_parameters={'rate': '0.48283254514084895'}),
            Layer(layer_id=LayerId(3),
                  layer_type=LayerType.POOL_MAX,
                  layer_parameters={'kernel_size': '5', 'stride': '3', 'padding': 'valid'}),
            Layer(layer_id=LayerId(4),
                  layer_type=LayerType.FC,
                  layer_parameters={'act':'relu', 'out_features':'100', 'bias':'True'}),
            Layer(layer_id=LayerId(5),
                  layer_type=LayerType.FC,
                  layer_parameters={'act':'softmax', 'out_features':'10', 'bias':'True'})
        ]

        model_builder: ModelBuilder = ModelBuilder(
            parsed_network=ParsedNetwork(
                layers=layers,
                layers_connections={LayerId(6): [InputLayerId(5)],
                                    LayerId(5): [InputLayerId(4)],
                                    LayerId(4): [InputLayerId(3)],
                                    LayerId(3): [InputLayerId(2)],
                                    LayerId(2): [InputLayerId(1)],
                                    LayerId(1): [InputLayerId(0)],
                                    LayerId(0): [InputLayerId(-1)]}),
            parsed_projector_network=None,
            input_shape=Size([1, 28, 28]),
            device=Device.CPU
        )
        model = model_builder.assemble_network(LegacyEvaluator)
        expected_model_structure = OrderedDict([
            ('conv-1', nn.Sequential(nn.Conv2d(in_channels=1,
                                               out_channels=6,
                                               kernel_size=5,
                                               stride=1,
                                               padding='same'),
                                     nn.ReLU())),
            ('batch_norm-1', nn.BatchNorm2d(num_features=6)),
            ('dropout-1', nn.Dropout(p=0.48283254514084895, inplace=False)),
            ('pool_max-1', nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=3, padding=0))),
            ('fc-1', nn.Sequential(nn.Flatten(),
                                   nn.Linear(in_features=8*8*6, out_features=100, bias=True),
                                   nn.ReLU())),
            ('fc-2', nn.Sequential(nn.Linear(in_features=100, out_features=10, bias=True), nn.Softmax(dim=None)))
        ])
        self.assertEqual(repr(model._modules), repr(expected_model_structure))


    def test_assemble_network_2(self) -> None:

        layers: list[Layer] = [
            Layer(layer_id=LayerId(0),
                  layer_type=LayerType.DROPOUT,
                  layer_parameters={'rate': '0.48283254514084895'}),
            Layer(layer_id=LayerId(1),
                  layer_type=LayerType.BATCH_NORM,
                  layer_parameters={}),
            Layer(layer_id=LayerId(2),
                  layer_type=LayerType.POOL_MAX,
                  layer_parameters={'kernel_size': '2', 'stride': '1', 'padding': 'valid'}),
            Layer(layer_id=LayerId(3),
                  layer_type=LayerType.POOL_MAX,
                  layer_parameters={'kernel_size': '5', 'stride': '3', 'padding': 'valid'}),
            Layer(layer_id=LayerId(4),
                  layer_type=LayerType.POOL_AVG,
                  layer_parameters={'kernel_size': '3', 'stride': '2', 'padding': 'same'}),
            Layer(layer_id=LayerId(5),
                  layer_type=LayerType.DROPOUT,
                  layer_parameters={'rate': '0.2804266331697851'}),
            Layer(layer_id=LayerId(6),
                  layer_type=LayerType.FC,
                  layer_parameters={'act':'softmax', 'out_features':'10', 'bias':'True'}),
        ]

        expected_layer_connections: dict[LayerId, list[InputLayerId]] = {
            LayerId(6): [InputLayerId(5)],
            LayerId(5): [InputLayerId(4)],
            LayerId(4): [InputLayerId(1), InputLayerId(2), InputLayerId(3)],
            LayerId(3): [InputLayerId(0), InputLayerId(1), InputLayerId(2)],
            LayerId(2): [InputLayerId(0), InputLayerId(1)],
            LayerId(1): [InputLayerId(-1), InputLayerId(0)],
            LayerId(0): [InputLayerId(-1)]
        }
        model_builder: ModelBuilder = ModelBuilder(
            parsed_network=ParsedNetwork(layers=layers,
                                         layers_connections=expected_layer_connections),
            parsed_projector_network=None,
            input_shape=Size([1, 28, 28]),
            device=Device.CPU
        )
        model = model_builder.assemble_network(LegacyEvaluator)
        expected_model_structure = OrderedDict([
            ('dropout-1', nn.Dropout(p=0.48283254514084895, inplace=False)),
            ('batch_norm-1', nn.BatchNorm2d(num_features=1)),
            ('pool_max-1', nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1, padding=0))),
            ('pool_max-2', nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=3, padding=0))),
            ('pool_avg-1', nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=0), nn.ZeroPad2d((8,7,8,7)))),
            ('dropout-2', nn.Dropout(p=0.2804266331697851, inplace=False)),
            ('fc-1', nn.Sequential(nn.Flatten(),
                                   nn.Linear(in_features=28*28, out_features=10, bias=True),
                                   nn.Softmax(dim=None))),
            ('conv-1', nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))),
            ('conv-2', nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))),
            ('conv-3', nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(10, 10)))
        ])
        self.assertEqual(repr(model._modules), repr(expected_model_structure))


    def test_assemble_network_with_projector(self) -> None:
        layers: list[Layer] = [
            Layer(layer_id=LayerId(0),
                  layer_type=LayerType.BATCH_NORM,
                  layer_parameters={}),
            Layer(layer_id=LayerId(1),
                  layer_type=LayerType.DROPOUT,
                  layer_parameters={'rate': '0.48283254514084895'}),
            Layer(layer_id=LayerId(2),
                  layer_type=LayerType.POOL_MAX,
                  layer_parameters={'kernel_size': '5', 'stride': '3', 'padding': 'valid'})
        ]
        projector_layers: list[Layer] = [
            Layer(layer_id=LayerId(0),
                  layer_type=LayerType.FC,
                  layer_parameters={'act':'linear', 'out_features':'20', 'bias':'True'}),
            Layer(layer_id=LayerId(1),
                  layer_type=LayerType.BATCH_NORM_PROJ,
                  layer_parameters={'act':'relu'}),
            Layer(layer_id=LayerId(2),
                  layer_type=LayerType.FC,
                  layer_parameters={'act':'linear', 'out_features':'10', 'bias':'True'}),
            Layer(layer_id=LayerId(3),
                  layer_type=LayerType.BATCH_NORM_PROJ,
                  layer_parameters={'act':'relu'})
        ]

        model_builder: ModelBuilder = ModelBuilder(
            parsed_network=ParsedNetwork(
                layers=layers,
                layers_connections={LayerId(2): [InputLayerId(1)],
                                    LayerId(1): [InputLayerId(0)],
                                    LayerId(0): [InputLayerId(-1)]}),
            parsed_projector_network=ParsedNetwork(
                layers=projector_layers,
                layers_connections={LayerId(3): [InputLayerId(2)],
                                    LayerId(2): [InputLayerId(1)],
                                    LayerId(1): [InputLayerId(0)],
                                    LayerId(0): [InputLayerId(-1)]}),
            input_shape=Size([1, 28, 28]),
            device=Device.CPU
        )
        model = model_builder.assemble_network(BarlowTwinsEvaluator, Pretext(PretextType.BT, {'lamb': '0.01'}))
        expected_model_structure = OrderedDict([
            ('batch_norm-1', nn.BatchNorm2d(num_features=1)),
            ('dropout-1', nn.Dropout(p=0.48283254514084895, inplace=False)),
            ('pool_max-1', nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=3, padding=0))),
            ('projector_model', OrderedDict([
                ('fc-1', nn.Sequential(nn.Flatten(), nn.Linear(in_features=64, out_features=20, bias=True))),
                ('batch_norm_proj-1', nn.Sequential(nn.Flatten(), nn.BatchNorm1d(20), nn.ReLU())),
                ('fc-2', nn.Sequential(nn.Linear(in_features=20, out_features=10, bias=True))),
                ('batch_norm_proj-2', nn.Sequential(nn.BatchNorm1d(10), nn.ReLU()))
            ]))
        ])
        projector_model = model._modules.pop('projector_model')
        assert projector_model is not None
        expected_projector_model = expected_model_structure.pop('projector_model')
        self.assertEqual(repr(model._modules), repr(expected_model_structure))
        self.assertEqual(repr(projector_model._modules), repr(expected_projector_model))


if __name__ == '__main__':
    unittest.main()
