'''
from __future__ import annotations

from argparse import ArgumentParser
import json
import os
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

import evodenss
from evodenss.config import Config
from evodenss.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device
from evodenss.misc.phenotype_parser import Optimiser
from evodenss.misc.utils import is_valid_file
from evodenss.networks.torch.evolved_networks import BarlowTwinsNetwork, \
    EvaluationBarlowTwinsNetwork, EvolvedNetwork
from evodenss.networks.torch.callbacks import ModelCheckpointCallback
from evodenss.networks.torch.evaluators import BarlowTwinsEvaluator
from evodenss.networks.torch.dataset_loader import load_dataset
from evodenss.networks.torch.model_builder import ModelBuilder
from evodenss.networks.torch.trainers import Trainer
from evodenss.networks.torch.transformers import LegacyTransformer, BarlowTwinsTransformer
from evodenss.misc.phenotype_parser import parse_phenotype

import torch
from torch import Size
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:
    from evodenss.networks.torch import LearningParams

dataset_name = "cifar10"
epochs = 50
model_string = \
    "layer:dropout rate:0.22193279343065397 input:-1 " \
    "layer:conv out_channels:32 kernel_size:2 stride:3 padding:valid act:relu bias:True input:0 " \
    "layer:dropout rate:0.22193279343065397 input:1 " \
    "projector_layer:fc act:sigmoid out_features:2048 bias:False input:-1 " \
    "projector_layer:identity input:0 " \
    "learning:gradient_descent lr:0.058011531573486035 momentum:0.8196745630556581 weight_decay:0.00066058513324
    37667 nesterov:True batch_size:64 epochs:50 " \
    "pretext:bt lamb:0.015625"""

parsed_network, proj_parsed_network, optimiser, pretext_task = \
            parse_phenotype(model_string)

input_size = DATASETS_INFO[dataset_name]["expected_input_dimensions"]
model_builder: ModelBuilder = ModelBuilder(parsed_network, proj_parsed_network, Device.GPU, Size(input_size))


# <class 'evodenss.networks.torch.evaluators.BarlowTwinsEvaluator'>
torch_model = model_builder.assemble_network(BarlowTwinsEvaluator, pretext_task)
learning_params = ModelBuilder.assemble_optimiser(torch_model.parameters(), optimiser)


augmentation_params = {
    'input_a': {
        'random_resized_crop': {
            'size': 32
        },
        'random_horizontal_flip': {
            'probability': 0.5
        },
        'color_jitter': {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.4,
            'hue': 0.1,
            'probability': 0.8
        },
        'random_grayscale': {
            'probability': 0.2
        },
        'normalize': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        }
    },
    'input_b': {
        'random_resized_crop': {
            'size': 32
        },
        'random_horizontal_flip': {
            'probability': 0.5
        },
        'color_jitter': {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.4,
            'hue': 0.1,
            'probability': 0.8
        },
        'random_grayscale': {
            'probability': 0.2
        },
        'normalize': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010]
        }
    }
}

train_transformer = BarlowTwinsTransformer(augmentation_params)
test_transformer = BarlowTwinsTransformer(augmentation_params)

(pretext_train_data, _, _) = load_dataset(dataset_name, train_transformer,test_transformer)
# evo_train_subset = Subset(pretext_train_data, metadata_dict['dataset']['pretext']['train'])
train_data_loader = DataLoader(pretext_train_data,
                               batch_size=learning_params.batch_size,
                               shuffle=False,
                               drop_last=False,
                               pin_memory=True)

pretext_trainer = Trainer(model=torch_model,
                          optimiser=learning_params.torch_optimiser,
                          train_data_loader=train_data_loader,
                          validation_data_loader=None,
                          loss_function=None,
                          n_epochs=learning_params.epochs,
                          initial_epoch=0,
                          device=Device.GPU,
                          callbacks=[])
pretext_trainer.barlow_twins_train(learning_params.batch_size)
'''
