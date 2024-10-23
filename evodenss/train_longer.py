from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Any

import dill
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

import evodenss
from evodenss.config.pydantic import AugmentationConfig, ConfigBuilder, get_config
from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetProcessor, DatasetType
from evodenss.evolution.individual import Individual
from evodenss.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device, OptimiserType
from evodenss.misc.metadata_info import MetadataInfo, PretextTrainingInfo, TrainingInfo
from evodenss.misc.utils import is_valid_file
from evodenss.networks.evolved_networks import BarlowTwinsNetwork, EvaluationBarlowTwinsNetwork, EvolvedNetwork
from evodenss.networks.model_builder import ModelBuilder
from evodenss.networks.phenotype_parser import Optimiser
from evodenss.networks.transformers import BarlowTwinsTransformer, LegacyTransformer
from evodenss.train.callbacks import ModelCheckpointCallback
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
    
    if metadata_info.pretext_training_info is not None:
       dataset_partitioning[DatasetType.PRETEXT_TRAIN] = Subset(unlabelled_data,
                                                                metadata_info.pretext_training_info.train_indices)
       dataset_partitioning[DatasetType.EVO_TEST] = Subset(train_labelled_data,
                                                           metadata_info.pretext_training_info.test_indices)
    if metadata_info.downstream_training_info is not None:
        dataset_partitioning[DatasetType.DOWNSTREAM_TRAIN] = Subset(train_labelled_data,
                                                                    metadata_info.downstream_training_info.train_indices)
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

    train_data_loader = DataLoader(dataset[DatasetType.DOWNSTREAM_TRAIN],
                                   batch_size=metadata_info.batch_size,
                                   shuffle=False,
                                   num_workers=4,
                                   drop_last=False,
                                   pin_memory=True)
    validation_data_loader = DataLoader(dataset[DatasetType.VALIDATION],
                                        batch_size=metadata_info.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        drop_last=False,
                                        pin_memory=True)
    test_data_loader = DataLoader(dataset[DatasetType.EVO_TEST],
                                  batch_size=metadata_info.batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False,
                                  pin_memory=True)

    if isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        #params_to_tune = iter([param for name, param in model.named_parameters() if 'final_layer' in name])
        params_to_tune = iter([param for _, param in model.named_parameters()])
    else:
        params_to_tune = model.parameters()
    #learning_params = ModelBuilder.assemble_optimiser(
    #    params_to_tune,
    #    Optimiser(metadata_dict['learning']['downstream']['optimiser']['name'],
    #                metadata_dict['learning']['downstream']['optimiser']['params'])
    #)
    downstream_optimiser = optim.Adam(params_to_tune, lr=0.001, weight_decay=0.000005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(downstream_optimiser, 10)
    downstream_trainer = Trainer(model=model,
                                 optimiser=downstream_optimiser,
                                 train_data_loader=train_data_loader,
                                 validation_data_loader=validation_data_loader,
                                 loss_function=nn.CrossEntropyLoss(),
                                 n_epochs=downstream_epochs,
                                 initial_epoch=metadata_info.trained_epochs,
                                 device=device,
                                 callbacks=[ModelCheckpointCallback(
                                                model_output_dir,
                                                model_filename=f"extended_complete_{MODEL_FILENAME}",
                                                weights_filename=f"extended_complete_{WEIGHTS_FILENAME}",
                                                metadata_info=MetadataInfo(pretext_training_info=None,
                                                                           downstream_training_info=None)
                                            )],
                                scheduler=scheduler)
    logger.info("extending downstream training")
    if isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        model.requires_grad_(True)
        model.final_layer.requires_grad_(True)
    downstream_trainer.train()

    final_test_data_loader = DataLoader(dataset[DatasetType.TEST],
                                        batch_size=metadata_info.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        drop_last=False,
                                        pin_memory=True)
    evo_test_accuracy: float = compute_metric(model, test_data_loader, device)
    test_accuracy: float = compute_metric(model, final_test_data_loader, device)
    logger.info(f"Accuracy of extended model on evo test set: {evo_test_accuracy}")
    logger.info(f"Accuracy of extended model on final test set: {test_accuracy}")




def extend_barlow_twins_train(model: BarlowTwinsNetwork | EvaluationBarlowTwinsNetwork,
                              dataset: dict[DatasetType, Subset[ConcreteDataset]],
                              metadata_info: MetadataInfo,
                              model_output_dir: str,
                              pretext_epochs: int,
                              downstream_epochs: int,
                              device: Device) -> None:
    if isinstance(model, BarlowTwinsNetwork) is True:
        assert metadata_info.pretext_training_info is not None
        pretext_info: PretextTrainingInfo = metadata_info.pretext_training_info
        train_data_loader = DataLoader(dataset[DatasetType.PRETEXT_TRAIN],
                                       batch_size=pretext_info.batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       drop_last=False,
                                       pin_memory=True)
        learning_params: LearningParams = ModelBuilder.assemble_optimiser(
            model.parameters(),
            Optimiser(OptimiserType(pretext_info.optimiser_name), pretext_info.optimiser_parameters)
        )
        pretext_trainer = Trainer(model=model,
                                  optimiser=learning_params.torch_optimiser,
                                  train_data_loader=train_data_loader,
                                  validation_data_loader=None,
                                  loss_function=None,
                                  n_epochs=pretext_epochs,
                                  initial_epoch=pretext_info.trained_epochs,
                                  device=device,
                                  callbacks=[])
        logger.info("extending pretext training")
        pretext_trainer.barlow_twins_train()

        n_neurons: int = DATASETS_INFO[pretext_info.dataset_name]['classes']

        if metadata_info.downstream_training_info is not None:
            extend_supervised_train(EvaluationBarlowTwinsNetwork(model, n_neurons, device),
                                    dataset=dataset,
                                    metadata_info=metadata_info.downstream_training_info,
                                    model_output_dir=model_output_dir,
                                    downstream_epochs=downstream_epochs,
                                    device=device)

    elif isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        assert metadata_info.downstream_training_info is not None
        extend_supervised_train(model,
                                dataset=dataset,
                                metadata_info=metadata_info.downstream_training_info,
                                model_output_dir=model_output_dir,
                                downstream_epochs=downstream_epochs,
                                device=device)


def main(model_path: str,
         weights_path: str,
         metadata_path: str,
         augmentation_params: AugmentationConfig,
         model_output_dir: str,
         pretext_epochs: int,
         downstream_epochs: int,
         is_gpu_run: bool) -> None: #pragma: no cover
    
    model = torch.load(model_path)
    model.load_state_dict(torch.load(weights_path))
    device: Device = Device.GPU if is_gpu_run is True else Device.CPU
    model.to(device.value)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_info: MetadataInfo = MetadataInfo(**json.load(f))


    (ssl_transformer, train_transformer, test_transformer) = recreate_transformers(augmentation_params)
    dataset: dict[DatasetType, Subset[ConcreteDataset]] = \
        recreate_dataset_partitioning(metadata_info,
                                      ssl_transformer,
                                      train_transformer,
                                      test_transformer)
    if metadata_info.pretext_training_info is not None:
        extend_barlow_twins_train(model,
                                  dataset,
                                  metadata_info,
                                  model_output_dir,
                                  pretext_epochs,
                                  downstream_epochs,
                                  device)
    if metadata_info.downstream_training_info is not None:
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
    parser.add_argument("--pretext-epochs", required=False, help="Number of pretext epochs to train", type=int)
    parser.add_argument("--downstream-epochs", required=True, help="Number of downstream epochs to train", type=int)
    parser.add_argument("--gpu-enabled", required=False, help="Runs the experiment in the GPU",
                        action='store_true')

    args: Any = parser.parse_args()

    logging.setLogRecordFactory(evodenss.logger_record_factory(-1))
    logger = logging.getLogger(__name__)

    start = time.time()
    torch.backends.cudnn.benchmark = True

    _= ConfigBuilder(config_path=args.config_path)
    augmentation_params: AugmentationConfig = get_config().network.learning.augmentation

    with open(args.individual_path, 'rb') as handle_individual:
        individual: Individual = dill.load(handle_individual)
    print(individual.metrics)

    main(model_path=args.model_path,
         weights_path=args.weights_path,
         metadata_path=args.metadata_path,
         augmentation_params=augmentation_params,
         model_output_dir=args.output_model_path,
         pretext_epochs=args.pretext_epochs,
         downstream_epochs=args.downstream_epochs,
         is_gpu_run=args.gpu_enabled)
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
