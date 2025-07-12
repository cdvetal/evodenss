from __future__ import annotations

import json
import logging
import os
import time
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Any, Iterator

import dill
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import evodenss
from evodenss.config.pydantic import AugmentationConfig, ConfigBuilder, get_config
from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetProcessor, DatasetType
from evodenss.evolution.individual import Individual
from evodenss.misc.constants import MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device, OptimiserType
from evodenss.misc.metadata_info import MetadataInfo, TrainingInfo
from evodenss.misc.utils import is_valid_file
from evodenss.networks.evolved_networks import EvaluationBarlowTwinsNetwork, EvolvedNetwork
from evodenss.networks.model_builder import ModelBuilder
from evodenss.networks.phenotype_parser import Optimiser
from evodenss.networks.transformers import BarlowTwinsTransformer, LegacyTransformer
from evodenss.train.callbacks import AccuracyTrackerCallback, Callback, ModelCheckpointCallback
from evodenss.train.learning_parameters import LearningParams
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
    
    if metadata_info.downstream_training_info is not None:
        dataset_partitioning[DatasetType.DOWNSTREAM_TRAIN] = Subset(
            train_labelled_data,
            list(metadata_info.downstream_training_info.train_indices) + \
                list(metadata_info.downstream_training_info.test_indices)
        )
        dataset_partitioning[DatasetType.VALIDATION] = Subset(train_labelled_data,
                                                              metadata_info.downstream_training_info.validation_indices)
        dataset_partitioning[DatasetType.EVO_TEST] = Subset(train_labelled_data,
                                                            metadata_info.downstream_training_info.test_indices)
    dataset_partitioning[DatasetType.TEST] = Subset(evaluation_labelled_data, list(range(len(test_data.targets))))
    return dataset_partitioning


def extend_supervised_train(model: EvaluationBarlowTwinsNetwork | EvolvedNetwork,
                            dataset: dict[DatasetType, Subset[ConcreteDataset]],
                            metadata_info: TrainingInfo,
                            model_output_dir: str,
                            downstream_epochs: int,
                            device: Device) -> None:

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

    params_to_tune: Iterator[nn.Parameter] = model.parameters()
    betas: list[float]
    if "betas" in metadata_info.optimiser_parameters:
        betas = metadata_info.optimiser_parameters.pop("betas")
        metadata_info.optimiser_parameters['beta1'] = betas[0]
        metadata_info.optimiser_parameters['beta2'] = betas[1]
    metadata_info.optimiser_parameters['batch_size'] = metadata_info.batch_size
    metadata_info.optimiser_parameters['epochs'] = metadata_info.trained_epochs
    optimiser = Optimiser(OptimiserType(metadata_info.optimiser_name), metadata_info.optimiser_parameters)
    learning_params: LearningParams = ModelBuilder.assemble_optimiser(
        list(params_to_tune),
        optimiser
    )
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(downstream_optimiser, 10)



    final_test_data_loader = DataLoader(dataset[DatasetType.TEST],
                                        batch_size=metadata_info.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        drop_last=False,
                                        pin_memory=True)
    print("Epochs trained: ", metadata_info.trained_epochs)
    print("Epochs to train: ", downstream_epochs)
    downstream_trainer = Trainer(model=model,
                                 optimiser=learning_params.torch_optimiser,
                                 train_data_loader=train_data_loader,
                                 validation_data_loader=validation_data_loader,
                                 loss_function=nn.CrossEntropyLoss(),
                                 n_epochs=downstream_epochs,
                                 initial_epoch=metadata_info.trained_epochs,
                                 device=device,
                                 callbacks=list[Callback](
                                     [ModelCheckpointCallback(
                                        model_output_dir,
                                        model_filename=f"extended_complete_{MODEL_FILENAME}",
                                        weights_filename=f"extended_complete_{WEIGHTS_FILENAME}",
                                        metadata_info=MetadataInfo.new_instance(
                                            metadata_info.dataset_name,
                                            dataset,
                                            optimiser,
                                            learning_params,
                                            None)),
                                      AccuracyTrackerCallback(
                                        final_test_data_loader,
                                        device,
                                        epochs_delta=100,
                                        filename=os.path.join(model_output_dir,
                                                              "accuracy_tracker.csv"))]),
                                scheduler=None)
    logger.info("extending downstream training")
    downstream_trainer.train()

    test_accuracy: float = compute_metric(model, final_test_data_loader, device)
    logger.info(f"Accuracy of extended model on final test set: {test_accuracy}")


def main(model_path: str,
         weights_path: str,
         metadata_path: str,
         augmentation_params: AugmentationConfig,
         model_output_dir: str,
         downstream_epochs: int,
         is_gpu_run: bool) -> None: #pragma: no cover
    
    model = torch.load(model_path)
    model.load_state_dict(torch.load(weights_path))
    device: Device = Device.GPU if is_gpu_run is True else Device.CPU
    model.to(device.value)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_info: MetadataInfo = MetadataInfo(**json.load(f))


    print(os.path.dirname(model_output_dir))
    os.makedirs(os.path.dirname(model_output_dir), exist_ok=True)

    (ssl_transformer, train_transformer, test_transformer) = recreate_transformers(augmentation_params)
    dataset: dict[DatasetType, Subset[ConcreteDataset]] = \
        recreate_dataset_partitioning(metadata_info,
                                      ssl_transformer,
                                      train_transformer,
                                      test_transformer)
    
    assert metadata_info.downstream_training_info is not None
    extend_supervised_train(model,
                            dataset,
                            metadata_info.downstream_training_info,
                            model_output_dir,
                            downstream_epochs,
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
    parser.add_argument("--individual-path", required=False, help="Path to the individual file",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--output-model-path",
                        required=False,
                        help="Path to final model",
                        default=".")
    parser.add_argument("--downstream-epochs", required=True, help="Number of downstream epochs to train", type=int)
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
         augmentation_params=augmentation_params,
         model_output_dir=args.output_model_path,
         downstream_epochs=args.downstream_epochs,
         is_gpu_run=args.gpu_enabled)
    end = time.time()
    time_elapsed = int(end - start)
    secs_elapsed = time_elapsed % 60
    mins_elapsed = time_elapsed//60 % 60
    hours_elapsed = time_elapsed//3600 % 60
    logger.info(f"Time taken to perform run: {compute_time_elapsed_human(time_elapsed)}")

    logging.shutdown()
