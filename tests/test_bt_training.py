import logging
from tkinter import N
from typing import Optional, List, Dict
import unittest

from fast_denser.misc.phenotype_parser import Optimiser
from fast_denser.misc.enums import Entity, LayerType, OptimiserType
from fast_denser.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME

import fast_denser
from fast_denser.misc.enums import Device, FitnessMetricName
from fast_denser.misc.phenotype_parser import parse_phenotype
from fast_denser.neural_networks_torch import LearningParams, ModelBuilder
from fast_denser.neural_networks_torch.evaluators import BarlowTwinsEvaluator, EvaluationBarlowTwinsNetwork
from fast_denser.neural_networks_torch.trainers import Trainer
from fast_denser.neural_networks_torch.transformers import LegacyTransformer, BarlowTwinsTransformer
from fast_denser.misc.proportions import ProportionsFloat, ProportionsIndexes
from fast_denser.neural_networks_torch.callbacks import EarlyStoppingCallback, \
    ModelCheckpointCallback, TimedStoppingCallback
from fast_denser.misc.fitness_metrics import *

from fast_denser.neural_networks_torch.dataset_loader import DatasetType, load_dataset

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import os
import random
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

    def __call__(self, x):
        return x


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, batch_size, lambd, projector_str):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.projector_str = projector_str
        self.backbone = torchvision.models.resnet50()
        # Uncomment next line for fashion mnist
        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to("mps")
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        print(self.backbone)

        # projector
        sizes = [2048] + list(map(int, projector_str.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False, device="cuda:0")

    
    def forward(self, y1, y2=None, batch_size=None):
        if y2 is not None:
            z1 = self.projector(self.backbone(y1))
            z2 = self.projector(self.backbone(y2))

            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(batch_size)
            #torch.distributed.all_reduce(c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss = on_diag + self.lambd * off_diag
            return on_diag, off_diag, loss
        else:
            # In case we use the network for inference rather than training
            assert batch_size is None
            return self.backbone(y1)

class Test(unittest.TestCase):

    logging.setLogRecordFactory(fast_denser.logger_record_factory(0))
    logger = logging.getLogger(__name__)

    def _testing_performance(self, model_dir: str) -> float:
        model_filename: str
        weights_filename: str
        model_filename = f"complete_{MODEL_FILENAME}"
        weights_filename = f"complete_{WEIGHTS_FILENAME}"
        
        torch_model: nn.Module = torch.load(os.path.join(model_dir, model_filename))
        torch_model.load_state_dict(torch.load(os.path.join(model_dir, weights_filename)))
        torch_model.eval()

        device = Device.GPU
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)

        test_set = self.dataset[DatasetType.TEST]
        assert test_set is not None
        test_loader: DataLoader = DataLoader(test_set, batch_size=64, shuffle=True)
        metric = AccuracyMetric(batch_size=64)
        return metric.compute_metric(torch_model, test_loader, device)


    def setUp(self) -> None:
        train_transformer = BarlowTwinsTransformer(
            {
                'input_a':
                    {
                        'random_resized_crop': {'size': 32},
                        'random_horizontal_flip': {'probability': 0.5},
                        'color_jitter': {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1, 'probability': 0.8},
                        'random_grayscale': {'probability': 0.2},
                        'normalize': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
                    },
                'input_b':
                    {
                        'random_resized_crop': {'size': 32},
                        'random_horizontal_flip': {'probability': 0.5},
                        'color_jitter': {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1, 'probability': 0.8},
                        'random_grayscale': {'probability': 0.2},
                        'normalize': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
                    }
            }
        )

        supervised_train_transformer = LegacyTransformer({
            'random_resized_crop': {'size': 32},
            'random_horizontal_flip': {'probability': 0.5},
            'color_jitter': {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1, 'probability': 0.8},
            'random_grayscale': {'probability': 0.2},
            'normalize': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
        })
        test_transformer = LegacyTransformer({
                'normalize': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
            }
        )
        self.fitness_metric = FitnessMetricName.ACCURACY
        self.pairwise_dataset: Dict[DatasetType, Subset] = load_dataset(
            "cifar10",
            train_transformer,
            test_transformer,
            enable_stratify=True,
            proportions=ProportionsFloat({
                DatasetType.EVO_TRAIN: 0.8,
                DatasetType.EVO_TEST: 0.2
            }),
            downstream_train_percentage=None
        )
        self.dataset: Dict[DatasetType, Subset] = load_dataset(
            "cifar10",
            supervised_train_transformer,
            test_transformer,
            enable_stratify=True,
            proportions=ProportionsIndexes({
                DatasetType.EVO_TRAIN: list(self.pairwise_dataset[DatasetType.EVO_TRAIN].indices),
                DatasetType.EVO_TEST: list(self.pairwise_dataset[DatasetType.EVO_TEST].indices)
            }),
            downstream_train_percentage=100
        )

    def _get_data_loaders(self, dataset: Dict[DatasetType, Subset], batch_size: int):
        #train_dataset = self.dataset[DatasetType.EVO_TRAIN]
        #validation_dataset =  self.dataset[DatasetType.EVO_VALIDATION]
        #test_dataset = self.dataset[DatasetType.EVO_TEST]

        #assert train_dataset is not None
        #assert validation_dataset is not None
        #assert test_dataset is not None

        # dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=Transform())
        train_loader = DataLoader(dataset[DatasetType.EVO_TRAIN],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True)

        test_loader = DataLoader(dataset[DatasetType.EVO_TEST],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True)
        
        return train_loader, test_loader 

    def test_bt_training(self):
        
        batch_size = 128
        
        phenotype = \
            "layer:dropout rate:0.3207027600698071 input:-1 " \
            "layer:dropout rate:0.3207027600698071 input:0 " \
            "layer:pool_avg kernel_size:2 stride:1 padding:valid input:1 " \
            "layer:fc act:relu out_features:722 bias:True input:2 " \
            f"learning:adam lr:1e-3 weight_decay:1e-6 beta1:0.9 beta2:0.999 batch_size:{batch_size} epochs:50"
            #f"learning:lars lr_weights:0.2 lr_biases:0.005 momentum:0.9 weight_decay:1e-06 batch_size:{batch_size} epochs:1000" 
        '''
        phenotype = \
            "layer:batch_norm input:-1 " \
            "layer:batch_norm input:0,-1 " \
            "layer:dropout rate:0.4542534414679271 input:1,0 " \
            "layer:batch_norm input:2,0,-1 " \
            "layer:batch_norm input:3,1,2,0 " \
            "layer:fc act:relu out_features:1660 bias:False input:4 " \
            "layer:fc act:sigmoid out_features:521 bias:False input:5 " \
            "layer:fc act:relu out_features:1582 bias:False input:6,4 " \
            "layer:fc act:linear out_features:1040 bias:True input:7 " \
            "learning:rmsprop lr:0.02642589919536397 alpha:0.7436328005345161 weight_decay:0.005614487851218655 early_stop:21 batch_size:1076 epochs:1000 early_stop:29 batch_size:1779 epochs:1000"
        '''

        '''
        phenotype = \
            "layer:batch_norm input:-1 " \
            "layer:pool_max kernel_size:5 stride:2 padding:valid input:0,-1 " \
            "layer:conv out_channels:36 kernel_size:5 stride:3 padding:same act:linear bias:True input:1,-1,0 " \
            "layer:pool_avg kernel_size:4 stride:3 padding:valid input:2,-1,1 " \
            "layer:dropout rate:0.6870928621843465 input:3,1,0 " \
            "layer:pool_max kernel_size:5 stride:3 padding:valid input:4,3,2,1 " \
            "layer:dropout rate:0.5105139468268104 input:5 " \
            "layer:dropout rate:0.4366487906467176 input:6,5 " \
            "layer:fc act:sigmoid out_features:1712 bias:True input:7,6 " \
            "layer:fc act:linear out_features:679 bias:True input:8 " \
            f"learning:lars lr_weights:0.2 lr_biases:0.0030885992191835875 momentum:0.9 weight_decay:9.623428748064417e-06 batch_size:{batch_size} epochs:1000"
        '''

        layers, layers_connections, optimiser = parse_phenotype(phenotype)
        
        #model_builder: ModelBuilder = ModelBuilder(
        #    layers=layers,
        #    layers_connections=layers_connections,
        #    input_shape=(3, 32, 32),
        #    device=Device.GPU
        #)
        
        model = BarlowTwins(batch_size=batch_size, lambd=0.0078125, projector_str="512-128").to(Device.GPU.value)
        #model = model_builder.assemble_network(BarlowTwinsEvaluator)
        print(model)
        learning_params: LearningParams = ModelBuilder.assemble_optimiser(model.parameters(), optimiser)

        torch.manual_seed(0)
        train_loader, _ = self._get_data_loaders(self.pairwise_dataset, learning_params.batch_size)

        trainer = Trainer(model=model,
                          optimiser=learning_params.torch_optimiser,
                          loss_function=nn.CrossEntropyLoss(),
                          train_data_loader=train_loader,
                          validation_data_loader=None,
                          n_epochs=learning_params.epochs,
                          initial_epoch=0,
                          device=Device.GPU,
                          callbacks=[TimedStoppingCallback(max_seconds=12000)])
        trainer.barlow_twins_train(learning_params.batch_size)
        
        complete_model: EvaluationBarlowTwinsNetwork = EvaluationBarlowTwinsNetwork(model, 10, Device.GPU)
        complete_model.to(Device.GPU.value, non_blocking=True)
        params_to_tune = [param for name, param in complete_model.named_parameters() if name in {'final_layer.weight', 'final_layer.bias'}]

        final_model_dir = "test_model_dir"
        train_loader, test_loader = self._get_data_loaders(self.dataset, learning_params.batch_size)
        # final_optimiser = torch.optim.SGD(params_to_tune, lr=0.3, momentum=0.9, weight_decay=1e-6)
        final_optimiser = optim.Adam(params_to_tune, lr=1e-3, weight_decay=1e-6)
        final_epochs = 100
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(final_optimiser, final_epochs)
        last_layer_trainer = Trainer(model=complete_model,
                                     optimiser=final_optimiser,
                                     loss_function=nn.CrossEntropyLoss(),
                                     train_data_loader=train_loader,
                                     validation_data_loader=None,
                                     n_epochs=final_epochs,
                                     initial_epoch=0,
                                     device=Device.GPU,
                                     callbacks=[ModelCheckpointCallback(final_model_dir,
                                                                        model_filename=f"complete_{MODEL_FILENAME}",
                                                                        weights_filename=f"complete_{WEIGHTS_FILENAME}")],
                                     scheduler=None)
        last_layer_trainer.train()
        
        fitness_metric = AccuracyMetric(batch_size=batch_size)
        fitness_value = Fitness(
            fitness_metric.compute_metric(complete_model, test_loader, Device.GPU),
            type(fitness_metric)
        )
        print("Fitness: ", fitness_value)
        print("Final accuracy: ", self._testing_performance(final_model_dir))

if __name__ == '__main__':
    unittest.main()
