from __future__ import annotations

import json
import logging
import os
import time
from argparse import ArgumentParser
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator, Optional, cast

import dill
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

import evodenss
from evodenss.config.pydantic import AugmentationConfig, ConfigBuilder, get_config
from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetProcessor, DatasetType
from evodenss.evolution.individual import Individual
from evodenss.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device, DownstreamMode, OptimiserType
from evodenss.misc.metadata_info import MetadataInfo, PretextTrainingInfo, TrainingInfo
from evodenss.misc.utils import is_valid_file
from evodenss.networks.evolved_networks import BarlowTwinsNetwork, EvaluationBarlowTwinsNetwork, EvolvedNetwork
from evodenss.networks.model_builder import ModelBuilder
from evodenss.networks.phenotype_parser import Optimiser
from evodenss.networks.transformers import BarlowTwinsTransformer, LegacyTransformer
from evodenss.train.callbacks import AccuracyTrackerCallback, Callback, ModelCheckpointCallback
from evodenss.train.learning_parameters import LearningParams
from evodenss.train.losses import BarlowTwinsLoss
from evodenss.train.trainers import Trainer

if TYPE_CHECKING:
    from evodenss.train.learning_parameters import LearningParams


def compute_time_elapsed_human(time_elapsed: int) -> str:
    units: list[str] = ["s", "m", "h", "d"]
    max_units: list[int] = [60, 60, 24]
    divisions: list[int] = [1, 60, 60]
    results: list[int] = []
    x: int = time_elapsed
    for div, max_value in zip(divisions, max_units):
        x = x // div
        results.append(x % max_value)
    results.append(x // 24)
    return ''.join([ f"{value}{unit}" for value, unit in zip(results[::-1], units[::-1]) ])



def compute_metric(model: nn.Module, data_loader: DataLoader[ConcreteDataset], device: Device) -> float:
    model.eval()
    correct_guesses: float = 0
    size: int = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device.value, non_blocking=True), \
                data[1].to(device.value, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_guesses += (predicted == labels).float().sum().item()
            size += len(labels)
    return correct_guesses/size


def recreate_transformers(augmentation_params: AugmentationConfig) -> tuple[BarlowTwinsTransformer,
                                                                        LegacyTransformer,
                                                                        LegacyTransformer]:
    train_transformer = BarlowTwinsTransformer(augmentation_params.pretext)
    supervised_train_transformer = LegacyTransformer(augmentation_params.downstream)
    supervised_test_transformer = LegacyTransformer(augmentation_params.test)
    return train_transformer, supervised_train_transformer, supervised_test_transformer


def recreate_dataset_partitioning(
        metadata_info: MetadataInfo,
        complete_metadata_info: MetadataInfo,
        ssl_transformer: BarlowTwinsTransformer,
        train_transformer: LegacyTransformer,
        test_transformer: LegacyTransformer) -> dict[DatasetType, Subset[ConcreteDataset]]:
    dataset_processor: DatasetProcessor = DatasetProcessor(ssl_transformer, train_transformer, test_transformer)
    dataset_partitioning: dict[DatasetType, Subset[ConcreteDataset]] = {}
    dataset_name: str

    if metadata_info.pretext_training_info is not None:
        # TODO: Not proud of this if statement below
        if metadata_info.downstream_training_info is not None:
            assert metadata_info.pretext_training_info.dataset_name == \
                  metadata_info.downstream_training_info.dataset_name
        dataset_name = metadata_info.pretext_training_info.dataset_name
    else:
        assert metadata_info.downstream_training_info is not None
        dataset_name = metadata_info.downstream_training_info.dataset_name

    (unlabelled_data, train_labelled_data, evaluation_labelled_data, test_data) = \
            dataset_processor._load_dataset(dataset_name)
    
    if metadata_info.pretext_training_info is not None:
       dataset_partitioning[DatasetType.PRETEXT_TRAIN] = \
        Subset(unlabelled_data,
               list(metadata_info.pretext_training_info.train_indices))
       dataset_partitioning[DatasetType.EVO_TEST] = Subset(train_labelled_data,
                                                           metadata_info.pretext_training_info.test_indices)
    if complete_metadata_info.downstream_training_info is not None:
        dataset_partitioning[DatasetType.DOWNSTREAM_TRAIN] = Subset(
            train_labelled_data,
            list(complete_metadata_info.downstream_training_info.train_indices) + \
                list(complete_metadata_info.downstream_training_info.test_indices)
        )
        dataset_partitioning[DatasetType.VALIDATION] = Subset(train_labelled_data,
                                                              complete_metadata_info.downstream_training_info.validation_indices)

    dataset_partitioning[DatasetType.TEST] = Subset(test_data, list(range(len(test_data.targets))))
    
    return dataset_partitioning


def extend_supervised_train(model: EvaluationBarlowTwinsNetwork | EvolvedNetwork,
                            dataset: dict[DatasetType, Subset[ConcreteDataset]],
                            metadata_info: TrainingInfo,
                            model_output_dir: str,
                            downstream_epochs: int,
                            downstream_mode: DownstreamMode,
                            device: Device,
                            pretext_epochs: Optional[int]=None) -> None:

    print(metadata_info.batch_size)
    train_data_loader: DataLoader[ConcreteDataset] = \
        DataLoader(dataset[DatasetType.DOWNSTREAM_TRAIN],
                   batch_size=metadata_info.batch_size,
                   shuffle=False,
                   num_workers=4,
                   drop_last=False,
                   pin_memory=True)
    validation_data_loader: DataLoader[ConcreteDataset] = \
        DataLoader(dataset[DatasetType.VALIDATION],
                   batch_size=metadata_info.batch_size,
                   shuffle=False,
                   num_workers=4,
                   drop_last=False,
                   pin_memory=True)

    params_to_tune: Iterator[nn.Parameter]
    if isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        if downstream_mode == DownstreamMode.freeze:
            params_to_tune = iter([param for name, param in model.named_parameters() if 'final_layer' in name])
            model.barlow_twins_trained_model.requires_grad_(False)
            model.final_layer.requires_grad_(True)
        else:
            params_to_tune = iter([param for _, param in model.named_parameters()])
            model.barlow_twins_trained_model.requires_grad_(True)
            model.final_layer.requires_grad_(True)
    else:
        params_to_tune = model.parameters()
    
    downstream_optimiser = optim.Adam(params_to_tune, lr=0.001, weight_decay=0.000001, betas=(0.9, 0.999))
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(downstream_optimiser, 10)



    final_test_data_loader = DataLoader(dataset[DatasetType.TEST],
                                        batch_size=metadata_info.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        drop_last=False,
                                        pin_memory=True)
    
    downstream_trainer = Trainer(model=model,
                                 optimiser=downstream_optimiser,
                                 train_data_loader=train_data_loader,
                                 validation_data_loader=validation_data_loader,
                                 loss_function=nn.CrossEntropyLoss(),
                                 n_epochs=downstream_epochs,
                                 initial_epoch=0,
                                 device=device,
                                 callbacks=list[Callback](
                                     [ModelCheckpointCallback(
                                        model_output_dir,
                                        model_filename=f"extended_complete_{MODEL_FILENAME}",
                                        weights_filename=f"extended_complete_{WEIGHTS_FILENAME}",
                                        metadata_info=MetadataInfo.new_instance(
                                            metadata_info.dataset_name,
                                            dataset,
                                            Optimiser(OptimiserType.ADAM, {'lr': '0.001', 'weight_decay': '0.000001',
                                                                           'beta1': '0.9', 'beta2': '0.999'}),
                                            LearningParams(None, 2048, downstream_epochs, downstream_optimiser),
                                            None)),
                                      AccuracyTrackerCallback(
                                        final_test_data_loader,
                                        device,
                                        epochs_delta=100,
                                        filename=os.path.join(model_output_dir,
                                                              f"accuracy_tracker_pretext_epochs={pretext_epochs}.csv"))]),
                                scheduler=None)
    logger.info("extending downstream training")
    downstream_trainer.train()

    test_accuracy: float = compute_metric(model, final_test_data_loader, device)
    logger.info(f"Accuracy of extended model on final test set: {test_accuracy}")




def extend_barlow_twins_train(model: BarlowTwinsNetwork | EvaluationBarlowTwinsNetwork,
                              dataset: dict[DatasetType, Subset[ConcreteDataset]],
                              metadata_info: MetadataInfo,
                              complete_metadata_info: MetadataInfo,
                              model_output_dir: str,
                              pretext_epochs: int,
                              downstream_epochs: int,
                              downstream_mode: DownstreamMode,
                              device: Device) -> None:
    delta: int = 100 # hardcoded. apologies for that
    if isinstance(model, BarlowTwinsNetwork) is True:
        assert metadata_info.pretext_training_info is not None
        assert complete_metadata_info.downstream_training_info is not None
        model_copy: Optional[BarlowTwinsNetwork] = None
        for epochs_setup in range(1, 6):
            pretext_info: PretextTrainingInfo = metadata_info.pretext_training_info
            downstream_info: TrainingInfo = complete_metadata_info.downstream_training_info
            actual_initial_epoch: int
            actual_pretext_epochs: int
            if epochs_setup == 1:
                actual_initial_epoch = pretext_info.trained_epochs
                actual_pretext_epochs = pretext_epochs
            else:
                actual_initial_epoch = actual_pretext_epochs
                actual_pretext_epochs = delta * epochs_setup

            train_data_loader = DataLoader(dataset[DatasetType.PRETEXT_TRAIN],
                                           batch_size=pretext_info.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           drop_last=False,
                                           pin_memory=True)
            # load previous state if model has been trained for downstream before
            if model_copy is not None:
                model = model_copy
                print(model.state_dict().keys())
            betas: list[float]
            if "betas" in pretext_info.optimiser_parameters:
                betas = pretext_info.optimiser_parameters.pop("betas")
                pretext_info.optimiser_parameters['beta1'] = betas[0]
                pretext_info.optimiser_parameters['beta2'] = betas[1]
            pretext_info.optimiser_parameters['batch_size'] = pretext_info.batch_size
            pretext_info.optimiser_parameters['epochs'] = actual_pretext_epochs
            learning_params: LearningParams = ModelBuilder.assemble_optimiser(
                list(model.parameters()),
                Optimiser(OptimiserType(pretext_info.optimiser_name), pretext_info.optimiser_parameters)
            )
            print(actual_pretext_epochs, actual_initial_epoch)
            pretext_trainer = Trainer(model=model,
                                      optimiser=learning_params.torch_optimiser,
                                      loss_function=BarlowTwinsLoss(pretext_info.pretext_algorithm_params['lamb']),
                                      train_data_loader=train_data_loader,
                                      validation_data_loader=None,
                                      n_epochs=actual_pretext_epochs,
                                      initial_epoch=actual_initial_epoch,
                                      device=device,
                                      callbacks=[])
            logger.info("extending pretext training")
            pretext_trainer.barlow_twins_train()

            # save the model state for the next training
            model_copy = cast(BarlowTwinsNetwork, deepcopy(model))
            n_neurons: int = DATASETS_INFO[pretext_info.dataset_name]['classes']

            if downstream_info is not None:
                ## there is a bug in evolutionary code so we have to do this
                downstream_info.batch_size = pretext_info.batch_size
                #downstream_info.batch_size = 2048
                extend_supervised_train(EvaluationBarlowTwinsNetwork(model, n_neurons, device),
                                        dataset=dataset,
                                        metadata_info=downstream_info,
                                        model_output_dir=model_output_dir,
                                        downstream_epochs=downstream_epochs,
                                        downstream_mode=downstream_mode,
                                        device=device,
                                        pretext_epochs=actual_pretext_epochs)

    elif isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        assert metadata_info.downstream_training_info is not None
        extend_supervised_train(model,
                                dataset=dataset,
                                metadata_info=metadata_info.downstream_training_info,
                                model_output_dir=model_output_dir,
                                downstream_epochs=downstream_epochs,
                                downstream_mode=downstream_mode,
                                device=device)


def main(model_path: str,
         weights_path: str,
         metadata_path: str,
         complete_metadata_path: str,
         augmentation_params: AugmentationConfig,
         model_output_dir: str,
         pretext_epochs: int,
         downstream_epochs: int,
         downstream_mode: DownstreamMode,
         is_gpu_run: bool,
         trained_pretext_epochs: int) -> None: #pragma: no cover
    
    model = torch.load(model_path)
    model.load_state_dict(torch.load(weights_path))
    device: Device = Device.GPU if is_gpu_run is True else Device.CPU
    model.to(device.value)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_info: MetadataInfo = MetadataInfo(**json.load(f))

    with open(complete_metadata_path, 'r', encoding='utf-8') as f:
        complete_metadata_info: MetadataInfo = MetadataInfo(**json.load(f))

    (ssl_transformer, train_transformer, test_transformer) = recreate_transformers(augmentation_params)
    dataset: dict[DatasetType, Subset[ConcreteDataset]] = \
        recreate_dataset_partitioning(metadata_info,
                                      complete_metadata_info,
                                      ssl_transformer,
                                      train_transformer,
                                      test_transformer)
    logger.info("Dataset partition sizes:")
    for partition, subset in dataset.items():
        logger.info(f"{partition} size -- {len(subset.indices)}")
    if metadata_info.pretext_training_info is not None:
        # this is a stupid hack because the metadata_info is not saving epochs correctly
        metadata_info.pretext_training_info.trained_epochs = trained_pretext_epochs
        extend_barlow_twins_train(model,
                                  dataset,
                                  metadata_info,
                                  complete_metadata_info,
                                  model_output_dir,
                                  pretext_epochs,
                                  downstream_epochs,
                                  downstream_mode,
                                  device)
    elif complete_metadata_info.downstream_training_info is not None:
        extend_supervised_train(model,
                                dataset,
                                complete_metadata_info.downstream_training_info,
                                model_output_dir,
                                downstream_epochs,
                                downstream_mode,
                                device)



if __name__ == '__main__': #pragma: no cover
    parser: ArgumentParser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-path",
                        required=True,
                        help="Path to the config file used to perform the neuroevolutionary run")
    parser.add_argument("--model-path", required=True, help="Path to the model",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--weights-path", '-w', required=True, help="Path to the weights file",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--metadata-path", required=True, help="Path to the metadata file",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--complete-metadata-path", required=True,
                        help="Path to the complete metadata file (needed when you want to extend downstream training)",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--individual-path", required=False, help="Path to the individual file",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--output-model-path",
                        required=False,
                        help="Path to final model",
                        default=".")
    parser.add_argument("--pretext-epochs", required=False, help="Number of pretext epochs to train", type=int)
    parser.add_argument("--downstream-epochs", required=True, help="Number of downstream epochs to train", type=int)
    parser.add_argument("--downstream-mode", required=False, choices=[dm.value for dm in DownstreamMode],
                         default=DownstreamMode.freeze, help="Mode of downstream training")
    parser.add_argument("--gpu-enabled", required=False, help="Runs the experiment in the GPU",
                        action='store_true')
    parser.add_argument("--run", "-r", required=True, help="Identifies the run id and seed to be used",
                        type=int)

    args: Any = parser.parse_args()
    
    
    torch.manual_seed(args.run)
    print(os.path.dirname(args.output_model_path))
    file_path = f"{os.path.dirname(args.output_model_path)}/file.log"
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    logging.setLogRecordFactory(evodenss.logger_record_factory(args.run))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG,
                        style="{",
                        format="{asctime} :: {levelname} :: {name} :: [{run}] -- {message}",
                        handlers=[stream_handler, logging.FileHandler(file_path)], force=True)
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    start = time.time()
    torch.backends.cudnn.benchmark = True

    _= ConfigBuilder(config_path=args.config_path, args_to_override=[])
    augmentation_params: AugmentationConfig = get_config().network.learning.augmentation

    with open(args.individual_path, 'rb') as handle_individual:
        individual: Individual = dill.load(handle_individual)
    assert individual.metrics is not None
    print(individual.metrics)

    main(model_path=args.model_path,
         weights_path=args.weights_path,
         metadata_path=args.metadata_path,
         complete_metadata_path=args.complete_metadata_path,
         augmentation_params=augmentation_params,
         model_output_dir=args.output_model_path,
         pretext_epochs=args.pretext_epochs,
         downstream_epochs=args.downstream_epochs,
         downstream_mode=DownstreamMode(args.downstream_mode),
         is_gpu_run=args.gpu_enabled,
         trained_pretext_epochs=individual.metrics.total_epochs_trained)
    end = time.time()
    time_elapsed = int(end - start)
    secs_elapsed = time_elapsed % 60
    mins_elapsed = time_elapsed//60 % 60
    hours_elapsed = time_elapsed//3600 % 60
    logger.info(f"Time taken to perform run: {compute_time_elapsed_human(time_elapsed)}")

    #for lr_weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    #    for lr_bias in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    #        for m in [0.2, 0.4, 0.6, 0.8, 0.9]:
    #            for decay in [1e-6]:
    #                torch.manual_seed(0)
    #                logger.info(f"LR w:{lr_weight} b:{lr_bias}, mom:{m}, decay:{decay}")
    #                main(model_path=args.model_path,
    #                     weights_path=args.weights_path,
    #                     metadata_path=args.metadata_path,
    #                     augmentation_params=augmentation_params,
    #                     model_output_dir=args.output_model_path,
    #                     pretext_epochs=args.pretext_epochs,
    #                     downstream_epochs=args.downstream_epochs,
    #                     is_gpu_run=args.gpu_enabled,
    #                     lr_weight=lr_weight,
    #                     lr_bias=lr_bias,
    #                     m=m,
    #                     decay=decay)
    #                end = time.time()
    #                time_elapsed = int(end - start)
    #                secs_elapsed = time_elapsed % 60
    #                mins_elapsed = time_elapsed//60 % 60
    #                hours_elapsed = time_elapsed//3600 % 60
    #                logger.info(f"Time taken to perform run: {compute_time_elapsed_human(time_elapsed)}")

    logging.shutdown()
