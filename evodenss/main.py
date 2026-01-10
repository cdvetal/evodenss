from __future__ import annotations

import logging
import os
import random
import time
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

import evodenss
from evodenss.config.pydantic import Config, ConfigBuilder, DataSplits, get_config, get_fitness_extra_params
from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetProcessor, create_dataset_processor
from evodenss.evolution import engine
from evodenss.evolution.grammar import Grammar
from evodenss.misc.checkpoint import Checkpoint
from evodenss.misc.constants import DATASETS_INFO, DEFAULT_SEED, START_FROM_SCRATCH, STATS_FOLDER_NAME
from evodenss.misc.enums import DownstreamMode, FitnessMetricName, OptimiserType
from evodenss.misc.persistence import RestoreCheckpoint, build_overall_best_path
from evodenss.misc.utils import ConfigPairAction, is_valid_file, is_yaml_file
from evodenss.networks.evaluators import BaseEvaluator

if TYPE_CHECKING:
    from torch.utils.data import Subset

    from evodenss.dataset.dataset_loader import DatasetType

logger: logging.Logger

def setup_logger(file_path: str, run: int) -> logging.Logger:
    file_path = f"{file_path}/run_{run}/file.log"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logging.setLogRecordFactory(evodenss.logger_record_factory(run))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG,
                        style="{",
                        format="{asctime} :: {levelname} :: {name} :: [{run}] -- {message}",
                        handlers=[stream_handler,
                                  logging.FileHandler(file_path)], force=True)
    return logging.getLogger(__name__)


def create_initial_checkpoint(dataset_name: str, run: int, is_gpu_run: bool) -> Checkpoint:
    evaluator = BaseEvaluator.create_evaluator(dataset_name, is_gpu_run)

    os.makedirs(os.path.join(get_config().checkpoints_path, f"run_{run}"), exist_ok=True)
    os.makedirs(os.path.join(get_config().checkpoints_path, f"run_{run}", STATS_FOLDER_NAME), exist_ok=True)

    return Checkpoint(
        run=run,
        random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        torch_random_state=torch.get_rng_state(),
        last_processed_generation=START_FROM_SCRATCH,
        total_epochs=0,
        best_fitness=None,
        evaluator=evaluator,
        best_gen_ind_test_accuracy=0.0
    )

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


@RestoreCheckpoint
def main(run: int,
         dataset_name: str,
         grammar: Grammar,
         config: Config,
         is_gpu_run: bool,
         possible_checkpoint: Optional[Checkpoint] = None) -> Checkpoint:

    if not logging.getLogger(__name__).hasHandlers():
        global logger
        logger = setup_logger(config.checkpoints_path, run)

    checkpoint: Checkpoint
    if possible_checkpoint is None:
        logger.info("Starting fresh run")
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        checkpoint = create_initial_checkpoint(dataset_name, run, is_gpu_run)
    else:
        logger.info("Loading previous checkpoint")
        checkpoint = possible_checkpoint
        random.setstate(checkpoint.random_state)
        np.random.set_state(checkpoint.numpy_random_state)
        torch.set_rng_state(checkpoint.torch_random_state)

    dataset_processor: DatasetProcessor = create_dataset_processor(config.network.learning.augmentation)

    total_generations: int = config.evolutionary.generations
    max_epochs: int = config.evolutionary.max_epochs
    proportions: DataSplits = config.network.learning.data_splits
    dataset: dict[DatasetType, Subset[ConcreteDataset]] = \
        dataset_processor.load_partitioned_dataset(dataset_name, proportions, DEFAULT_SEED)

    logger.info("Dataset partition sizes:")
    for partition, subset in dataset.items():
        logger.info(f"{partition} size -- {len(subset.indices)}")


    for gen in range(checkpoint.last_processed_generation + 1, total_generations):
        # check the total number of epochs (stop criteria)
        if checkpoint.total_epochs >= max_epochs:
            break
        checkpoint = engine.evolve(run, gen, dataset, grammar, checkpoint)

    # compute testing performance of the fittest network
    best_network_path: str = build_overall_best_path(config.checkpoints_path, run)
    fitness_metric_name: FitnessMetricName = get_config().evolutionary.fitness.metric_name
    if fitness_metric_name == FitnessMetricName.KNN_ACCURACY:
        best_test_acc_knn: float = \
            checkpoint.evaluator.testing_performance(
                dataset=dataset,
                model_dir=best_network_path,
                fitness_metric_name=FitnessMetricName.KNN_ACCURACY,
                **get_fitness_extra_params()
            )
        best_test_acc_linear: float = \
            checkpoint.evaluator.testing_performance(
                dataset=dataset,
                model_dir=best_network_path,
                fitness_metric_name=FitnessMetricName.DOWNSTREAM_ACCURACY,
                dataset_name=checkpoint.evaluator.dataset_name,
                batch_size=2048,
                downstream_mode=DownstreamMode.finetune,
                downstream_epochs=50,
                optimiser_type=OptimiserType.ADAM,
                optimiser_parameters={'lr': 0.001, 'weight_decay': 0.000001, 'beta1': 0.9, 'beta2': 0.999},
                **get_fitness_extra_params()
                )
        logger.info(f"Best test accuracy (KNN): {best_test_acc_knn}")
        logger.info(f"Best test accuracy (Linear): {best_test_acc_linear}")
    else:
        best_test_acc: float = checkpoint.evaluator.testing_performance(
            dataset=dataset,
            model_dir=best_network_path,
            fitness_metric_name=fitness_metric_name,
            dataset_name=dataset_name,
            model_saving_dir=None,
            **get_fitness_extra_params()
        )
        logger.info(f"Best test accuracy: {best_test_acc}")

    return checkpoint




if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-path", '-c', required=True, help="Path to the config file to be used",
                        type=lambda x: is_yaml_file(parser, x))
    parser.add_argument("--dataset-name", '-d', required=True, help="Name of the dataset to be used",
                        type=str, choices=list(DATASETS_INFO.keys()))
    parser.add_argument("--grammar-path", '-g', required=True, help="Path to the grammar to be used",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--run", "-r", required=False, help="Identifies the run id and seed to be used",
                        type=int, default=0)
    parser.add_argument("--override", required=False, help="Sets or overrides values in the config file",
                        action=ConfigPairAction, nargs=2, metavar=('config_key','value'), default=[])
    parser.add_argument("--gpu-enabled", required=False, help="Runs the experiment in the GPU",
                        action='store_true')
    args: Any = parser.parse_args()

    start = time.time()
    torch.backends.cudnn.benchmark = True
    # loads config. it is a singleton
    # from now onwards you can access the config anywhere by calling
    # `get_config()` located in evodenss.config
    config: Config = ConfigBuilder(config_path=args.config_path,
                                   args_to_override=args.override)
    logger = setup_logger(config.checkpoints_path, args.run)
    os.makedirs(config.checkpoints_path, exist_ok=True)
    main(run=args.run,
         dataset_name=args.dataset_name,
         grammar=Grammar(args.grammar_path, backup_path=get_config().checkpoints_path),
         config=config,
         is_gpu_run=args.gpu_enabled)

    end = time.time()
    time_elapsed = int(end - start)
    secs_elapsed = time_elapsed % 60
    mins_elapsed = time_elapsed//60 % 60
    hours_elapsed = time_elapsed//3600 % 60
    logger.info(f"Time taken to perform run: {compute_time_elapsed_human(time_elapsed)}")
    logging.shutdown()
