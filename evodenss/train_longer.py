from __future__ import annotations

from argparse import ArgumentParser
import json
import os
import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

import evodenss
from evodenss.config import Config
from evodenss.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME
from evodenss.misc.enums import Device
from evodenss.misc.phenotype_parser import Optimiser
from evodenss.misc.utils import is_valid_file
from evodenss.networks.torch.evolved_networks import BarlowTwinsNetwork, \
    EvaluationBarlowTwinsNetwork, EvolvedNetwork
from evodenss.networks.torch.callbacks import ModelCheckpointCallback
from evodenss.networks.torch.dataset_loader import load_dataset
from evodenss.networks.torch.model_builder import ModelBuilder
from evodenss.networks.torch.trainers import Trainer
from evodenss.networks.torch.transformers import LegacyTransformer, BarlowTwinsTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:
    from evodenss.networks.torch import LearningParams


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



def compute_metric(model: nn.Module, data_loader: DataLoader, device: Device) -> float:
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


def extend_supervised_train(model: EvaluationBarlowTwinsNetwork | EvolvedNetwork,
                            augmentation_params: Dict[str, Any],
                            metadata_dict: Dict[str, Any],
                            model_output_dir: str,
                            downstream_epochs: int,
                            device: Device) -> None:
    supervised_train_transformer = LegacyTransformer(augmentation_params['downstream'])
    supervised_test_transformer = LegacyTransformer(augmentation_params['test'])
    (downstream_train_data, downstream_evo_test_data, final_test_data) = \
        load_dataset(metadata_dict['dataset']['name'],
                        supervised_train_transformer,
                        supervised_test_transformer)
    
    evo_train_subset = Subset(downstream_train_data, metadata_dict['dataset']['downstream']['train'])
    evo_validation_subset: Optional[Subset]
    if metadata_dict['dataset']['downstream']['validation'] is None:
        evo_validation_subset = None
    else:
        evo_validation_subset = Subset(downstream_train_data, metadata_dict['dataset']['downstream']['validation'])
    evo_test_subset = Subset(downstream_evo_test_data, metadata_dict['dataset']['downstream']['test'])

    downstream_batch_size: int = metadata_dict['learning']['downstream']['batch_size']
    train_data_loader = DataLoader(evo_train_subset,
                                batch_size=downstream_batch_size,
                                shuffle=False,
                                num_workers=4,
                                drop_last=False,
                                pin_memory=True)
    validation_data_loader: Optional[DataLoader]
    if evo_validation_subset is None:
        validation_data_loader = None
    else:
        validation_data_loader = DataLoader(evo_test_subset,
                                            batch_size=downstream_batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            drop_last=False,
                                            pin_memory=True)
    test_data_loader = DataLoader(evo_test_subset,
                                batch_size=downstream_batch_size,
                                shuffle=False,
                                num_workers=4,
                                drop_last=False,
                                pin_memory=True)

    if isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        params_to_tune = iter([param for name, param in model.named_parameters() if 'final_layer' in name])
    else:
        params_to_tune = model.parameters()
    learning_params = ModelBuilder.assemble_optimiser(
        params_to_tune,
        Optimiser(metadata_dict['learning']['downstream']['optimiser']['name'],
                    metadata_dict['learning']['downstream']['optimiser']['params'])
    )
    
    downstream_trainer = Trainer(model=model,
                                optimiser=learning_params.torch_optimiser,
                                train_data_loader=train_data_loader,
                                validation_data_loader=validation_data_loader,
                                loss_function=nn.CrossEntropyLoss(),
                                n_epochs=downstream_epochs,
                                initial_epoch=metadata_dict['trained_downstream_epochs'],
                                device=device,
                                callbacks=[ModelCheckpointCallback(
                                                model_output_dir,
                                                model_filename=f"extended_complete_{MODEL_FILENAME}",
                                                weights_filename=f"extended_complete_{WEIGHTS_FILENAME}",
                                                metadata_info={'dataset_name': metadata_dict['dataset']['name']}
                                            )])
    logger.info("extending downstream training")
    if isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        model.requires_grad_(False)
        model.final_layer.requires_grad_(True)
    downstream_trainer.train()

    final_test_data_loader = DataLoader(final_test_data,
                                        batch_size=downstream_batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        drop_last=False,
                                        pin_memory=True)
    evo_test_accuracy: float = compute_metric(model, test_data_loader, device)
    test_accuracy: float = compute_metric(model, final_test_data_loader, device)
    logger.info(f"Accuracy of extended model on evo test set: {evo_test_accuracy}")
    logger.info(f"Accuracy of extended model on final test set: {test_accuracy}")




def extend_barlow_twins_train(model: BarlowTwinsNetwork | EvaluationBarlowTwinsNetwork,
                              augmentation_params: Dict[str, Any],
                              metadata_dict: Dict[str, Any],
                              model_output_dir: str,
                              pretext_epochs: int,
                              downstream_epochs: int,
                              device: Device) -> None:
    if isinstance(model, BarlowTwinsNetwork) is True:
        pretext_batch_size: int = metadata_dict['learning']['pretext']['batch_size']
        train_transformer = BarlowTwinsTransformer(augmentation_params['pretext'])
        test_transformer = BarlowTwinsTransformer(augmentation_params['pretext'])
        
        (pretext_train_data, _, _) = load_dataset(metadata_dict['dataset']['name'],
                                                    train_transformer,
                                                    test_transformer)
        evo_train_subset = Subset(pretext_train_data, metadata_dict['dataset']['pretext']['train'])
        train_data_loader = DataLoader(evo_train_subset,
                                    batch_size=pretext_batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    drop_last=False,
                                    pin_memory=True)
        learning_params: LearningParams = ModelBuilder.assemble_optimiser(
            model.parameters(),
            Optimiser(metadata_dict['learning']['pretext']['optimiser']['name'],
                    metadata_dict['learning']['pretext']['optimiser']['params'])
        )
        pretext_trainer = Trainer(model=model,
                                    optimiser=learning_params.torch_optimiser,
                                    train_data_loader=train_data_loader,
                                    validation_data_loader=None,
                                    loss_function=None,
                                    n_epochs=pretext_epochs,
                                    initial_epoch=metadata_dict['trained_pretext_epochs'],
                                    device=device,
                                    callbacks=[])
        logger.info("extending pretext training")
        pretext_trainer.barlow_twins_train(pretext_batch_size)

        n_neurons: int = DATASETS_INFO[metadata_dict['dataset']['name']]['classes']

        extend_supervised_train(EvaluationBarlowTwinsNetwork(model, n_neurons, device),
                                metadata_dict=metadata_dict,
                                augmentation_params=augmentation_params,
                                model_output_dir=model_output_dir,
                                downstream_epochs=downstream_epochs,
                                device=device)

    elif isinstance(model, EvaluationBarlowTwinsNetwork) is True:
        extend_supervised_train(model,
                                metadata_dict=metadata_dict,
                                augmentation_params=augmentation_params,
                                model_output_dir=model_output_dir,
                                downstream_epochs=downstream_epochs,
                                device=device)


def main(model_path: str,
         weights_path: str,
         metadata_path: str,
         augmentation_params: Dict[str, Any],
         model_output_dir: str,
         pretext_epochs: int,
         downstream_epochs: int,
         is_gpu_run: bool) -> None: #pragma: no cover
    
    model = torch.load(model_path)
    model.load_state_dict(torch.load(weights_path))
    device: Device = Device.GPU if is_gpu_run is True else Device.CPU
    model.to(device.value)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dict: Dict[str, Any] = json.load(f)

    # just a gimmick to prepare the dict to create the optimiser
    pretext_params: Dict[str, Any] = metadata_dict['learning']['pretext']['optimiser']['params']
    if pretext_params is not None:
        metadata_dict['learning']['pretext']['optimiser']['params']['batch_size'] = \
            metadata_dict['learning']['pretext']['batch_size']
        metadata_dict['learning']['pretext']['optimiser']['params']['epochs'] = \
            metadata_dict['trained_pretext_epochs']
        if 'betas' in pretext_params.keys():
            metadata_dict['learning']['pretext']['optimiser']['params']['beta1'] = \
                metadata_dict['learning']['pretext']['optimiser']['params']['betas'][0]
            metadata_dict['learning']['pretext']['optimiser']['params']['beta2'] = \
                metadata_dict['learning']['pretext']['optimiser']['params']['betas'][1]
            metadata_dict['learning']['pretext']['optimiser']['params'].pop("betas")
        
    downstream_params: Dict[str, Any] = metadata_dict['learning']['downstream']['optimiser']['params']
    if downstream_params is not None:
        metadata_dict['learning']['downstream']['optimiser']['params']['batch_size'] = \
            metadata_dict['learning']['downstream']['batch_size']
        metadata_dict['learning']['downstream']['optimiser']['params']['epochs'] = \
            metadata_dict['trained_downstream_epochs']
        if 'betas' in downstream_params.keys():
            metadata_dict['learning']['downstream']['optimiser']['params']['beta1'] = \
                metadata_dict['learning']['downstream']['optimiser']['params']['betas'][0]
            metadata_dict['learning']['downstream']['optimiser']['params']['beta2'] = \
                metadata_dict['learning']['downstream']['optimiser']['params']['betas'][1]
            metadata_dict['learning']['downstream']['optimiser']['params'].pop("betas")

    pretext_algorithm_name: str = metadata_dict['learning']['pretext']['algorithm']['name']
    if pretext_algorithm_name is not None:
        extend_barlow_twins_train(model,
                                  augmentation_params,
                                  metadata_dict,
                                  model_output_dir,
                                  pretext_epochs,
                                  downstream_epochs,
                                  device)
    else:
        extend_supervised_train(model,
                                augmentation_params,
                                metadata_dict,
                                model_output_dir,
                                downstream_epochs,
                                device)



if __name__ == '__main__': #pragma: no cover
    parser: ArgumentParser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-path", required=True, help="Path to the config file used to perform the neuroevolutionary run")
    parser.add_argument("--model-path", required=True, help="Path to the model",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--weights-path", '-w', required=True, help="Path to the weights file",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--metadata-path", required=True, help="Path to the metadata file",
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

    config: Config = Config(args.config_path)
    augmentation_params: Dict[str, Any] = config['network']['learning']['augmentation']

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
    logging.shutdown()
