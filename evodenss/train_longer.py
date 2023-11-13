# type: ignore
from __future__ import annotations

from argparse import ArgumentParser
import pickle
import os
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import evodenss
from evodenss.misc.fitness_metrics import AccuracyMetric
from evodenss.config import Config
from evodenss.evolution import Individual
from evodenss.evolution.grammar import Grammar
from evodenss.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device
from evodenss.misc.proportions import ProportionsIndexes
from evodenss.misc.phenotype_parser import parse_phenotype
from evodenss.misc.utils import is_valid_file, is_yaml_file
from evodenss.networks.torch.callbacks import ModelCheckpointCallback
from evodenss.networks.torch.dataset_loader import DatasetType, load_dataset
from evodenss.networks.torch.evaluators import BarlowTwinsEvaluator, EvaluationBarlowTwinsNetwork
from evodenss.networks.torch.model_builder import ModelBuilder
from evodenss.networks.torch.trainers import Trainer
from evodenss.networks.torch.transformers import LegacyTransformer, BarlowTwinsTransformer

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:
    from evodenss.networks import LearningParams


def compute_time_elapsed_human(time_elapsed: int) -> str:
    units: List[str] = ["s", "m", "h", "d"]
    max_units: List[int] = [60, 60, 24]
    divisions: List[int] = [1, 60, 60]
    results: List[int] = []
    x: int = time_elapsed
    for div, max_value in zip(divisions, max_units):
        x = x // div
        results.append(x % max_value)
    results.append(x // 24)
    return ''.join([ f"{value}{unit}" for value, unit in zip(results[::-1], units[::-1]) ])


def get_data_loaders(dataset: Dict[DatasetType, Subset], batch_size: int) -> Tuple[DataLoader, Optional[DataLoader]]:

        train_loader = DataLoader(dataset[DatasetType.EVO_TRAIN],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False,
                                  pin_memory=True)

        validation_loader: Optional[DataLoader]
        if DatasetType.EVO_VALIDATION in dataset.keys():
            validation_loader = DataLoader(dataset[DatasetType.EVO_VALIDATION],
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           drop_last=False,
                                           pin_memory=True)
        else:
            validation_loader = None


        return train_loader, validation_loader


def testing_performance(model_dir: str, dataset: Dict[DatasetType, Subset], device: Device) -> float:
    model_filename: str = f"extended_complete_{MODEL_FILENAME}"
    weights_filename: str = f"extended_complete_{WEIGHTS_FILENAME}"
    
    torch_model: nn.Module = torch.load(os.path.join(model_dir, model_filename))
    torch_model.load_state_dict(torch.load(os.path.join(model_dir, weights_filename)))
    torch_model.eval()

    torch_model.to(device.value, non_blocking=True)

    test_set = dataset[DatasetType.TEST]
    assert test_set is not None
    test_loader: DataLoader = DataLoader(test_set, batch_size=64, shuffle=True)
    metric = AccuracyMetric(batch_size=64)
    return metric.compute_metric(torch_model, test_loader, device)


def main(config: Config,
         grammar: Grammar,
         individual_path: str,
         weights_path: str,
         dataset_name: str,
         pretrained_model_output_dir: str,
         pretrain_epochs: int,
         final_model_output_dir: str,
         downstream_epochs: int,
         is_gpu_run: bool) -> None: #pragma: no cover
    
    device: Device = Device.GPU if is_gpu_run is True else Device.CPU
    with open(individual_path, "rb") as handle_ind:
        individual: Individual = pickle.load(handle_ind)
  
    train_transformer = BarlowTwinsTransformer(config['network']['learning']['augmentation']['train'])
    supervised_train_transformer = LegacyTransformer(config['network']['learning']['augmentation']['last_layer_train'])
    test_transformer = LegacyTransformer(config['network']['learning']['augmentation']['test'])

    phenotype = individual._decode(grammar)
    layers, layers_connections, optimiser = parse_phenotype(phenotype)
    
    input_size = DATASETS_INFO[dataset_name]["expected_input_dimensions"]
    model_builder: ModelBuilder = ModelBuilder(layers, layers_connections, device, input_size)
    torch_model = model_builder.assemble_network(BarlowTwinsEvaluator)
    torch_model.load_state_dict(torch.load(weights_path, map_location=torch.device(device.value)))
    torch_model.to(device.value)
    #print(torch_model.get_device())
    logger.debug(torch_model)

    learning_params: LearningParams = ModelBuilder.assemble_optimiser(
        torch_model.parameters(),
        optimiser
    )
    #print(learning_params.torch_optimiser.get_device())
    pairwise_dataset: Dict[DatasetType, Subset] = load_dataset(
        dataset_name,
        train_transformer,
        test_transformer,
        enable_stratify=True,
        proportions=ProportionsIndexes({
            DatasetType.EVO_TRAIN: list(range(50000)) #TODO: make this flexible
        }),
        downstream_train_percentage=100
    )    
    dataset: Dict[DatasetType, Subset] = load_dataset(
        dataset_name,
        supervised_train_transformer,
        test_transformer,
        enable_stratify=True,
        proportions=ProportionsIndexes({
            DatasetType.EVO_TRAIN: list(pairwise_dataset[DatasetType.EVO_TRAIN].indices)
        }),
        downstream_train_percentage=100
    )

    train_loader, _ = get_data_loaders(pairwise_dataset, learning_params.batch_size)

    trainer = Trainer(model=torch_model,
                      optimiser=learning_params.torch_optimiser,
                      loss_function=nn.CrossEntropyLoss(),
                      train_data_loader=train_loader,
                      validation_data_loader=None,
                      n_epochs=pretrain_epochs,
                      initial_epoch=individual.num_epochs,
                      device=device,
                      callbacks=[ModelCheckpointCallback(pretrained_model_output_dir,
                                                         model_filename="pretrained_model.pt")])
            
    logger.info("start bt train")
    trainer.barlow_twins_train(learning_params.batch_size)

    train_loader, _ = get_data_loaders(dataset, learning_params.batch_size)
    n_classes: int = DATASETS_INFO[dataset_name]["classes"]
    
    complete_model: EvaluationBarlowTwinsNetwork = EvaluationBarlowTwinsNetwork(torch_model, n_classes, device)

    complete_model.to(device.value, non_blocking=True)
    relevant_index: int = complete_model.relevant_index
    params_to_tune = [param for name, param in complete_model.named_parameters()
                     if name in {f'final_layer.{relevant_index}.weight', f'final_layer.{relevant_index}.bias'}]
    last_layer_trainer = Trainer(model=complete_model,
                                 optimiser=torch.optim.Adam(params_to_tune, lr=1e-3, weight_decay=1e-6),
                                 loss_function=nn.CrossEntropyLoss(),
                                 train_data_loader=train_loader,
                                 validation_data_loader=None,
                                 n_epochs=downstream_epochs,
                                 initial_epoch=0,
                                 device=device,
                                 callbacks=[ModelCheckpointCallback(
                                                final_model_output_dir,
                                                model_filename=f"extended_complete_{MODEL_FILENAME}",
                                                weights_filename=f"extended_complete_{WEIGHTS_FILENAME}"
                                            )])
    last_layer_trainer.train()

    # compute testing performance of the fittest network
    best_test_acc: float = testing_performance(final_model_output_dir, dataset, device)
    logger.info(f"Best test accuracy: {best_test_acc}")


if __name__ == '__main__': #pragma: no cover
    parser: ArgumentParser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-path", '-c', required=True, help="Path to the config file to be used",
                        type=lambda x: is_yaml_file(parser, x))
    parser.add_argument("--grammar-path", '-g', required=True, help="Path to the grammar to be used",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--individual-path", '-i', required=True, help="Path to the individual",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--weights-path", '-w', required=True, help="Path to the weight file",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--dataset-name", '-d', required=True, help="Name of the dataset to be used",
                        type=str, choices=list(DATASETS_INFO.keys()))
    parser.add_argument("--output-pretrained-model-path", 
                        required=False,
                        help="Path to the pretrained model",
                        default=".")
    parser.add_argument("--epochs", required=True, help="Number of epochs to train", type=int)
    parser.add_argument("--output-final-model-path",
                        required=False,
                        help="Path to final model",
                        default=".")
    parser.add_argument("--downstream-epochs", required=True, help="Number of epochs to train", type=int)
    parser.add_argument("--cuda-enabled", required=False, help="Runs the experiment in the GPU",
                        action='store_true')

    
    
    args: Any = parser.parse_args()

    logging.setLogRecordFactory(evodenss.logger_record_factory(0))
    logger = logging.getLogger(__name__)

    start = time.time()
    torch.backends.cudnn.benchmark = True
    main(config=Config(args.config_path),
         grammar=Grammar(args.grammar_path, None),
         individual_path=args.individual_path,
         weights_path=args.weights_path,
         dataset_name=args.dataset_name,
         pretrained_model_output_dir=args.output_pretrained_model_path,
         pretrain_epochs=args.epochs,
         final_model_output_dir=args.output_final_model_path,
         downstream_epochs=args.downstream_epochs,
         is_gpu_run=args.cuda_enabled)

    end = time.time()
    time_elapsed = int(end - start)
    secs_elapsed = time_elapsed % 60
    mins_elapsed = time_elapsed//60 % 60
    hours_elapsed = time_elapsed//3600 % 60
    logger.info(f"Time taken to perform run: {compute_time_elapsed_human(time_elapsed)}")
    logging.shutdown()
