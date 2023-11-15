from __future__ import annotations

from argparse import ArgumentParser
import os
import logging
import random
import time
from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import torch

import evodenss
from evodenss.config import Config
from evodenss.evolution import engine
from evodenss.evolution.grammar import Grammar
from evodenss.misc import Checkpoint
from evodenss.misc.constants import DATASETS_INFO, STATS_FOLDER_NAME
from evodenss.misc.enums import FitnessMetricName
from evodenss.misc.persistence import RestoreCheckpoint, build_overall_best_path
from evodenss.misc.utils import is_valid_file, is_yaml_file
from evodenss.networks.torch.evaluators import create_evaluator


if TYPE_CHECKING:
    from evodenss.networks.torch.evaluators import BaseEvaluator


# pylint: disable=redefined-outer-name
def create_initial_checkpoint(dataset_name: str, config: Config, run: int, is_gpu_run: bool) -> Checkpoint:

    fitness_metric_name: FitnessMetricName = FitnessMetricName(config['evolutionary']['fitness_metric'])

    evaluator: BaseEvaluator = create_evaluator(dataset_name,
                                                fitness_metric_name,
                                                run,
                                                config['network']['learning'],
                                                is_gpu_run,)

    os.makedirs(os.path.join(config['checkpoints_path'], f"run_{run}"), exist_ok=True)
    os.makedirs(os.path.join(config['checkpoints_path'], f"run_{run}", STATS_FOLDER_NAME), exist_ok=True)

    return Checkpoint(
        run=run,
        random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        torch_random_state=torch.get_rng_state(),
        last_processed_generation=-1,
        total_epochs=0,
        best_fitness=None,
        evaluator=evaluator,
        best_gen_ind_test_accuracy=0.0
    )

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


@RestoreCheckpoint
def main(run: int,
         dataset_name: str,
         config: Config,
         grammar: Grammar,
         is_gpu_run: bool,
         possible_checkpoint: Optional[Checkpoint] = None) -> Checkpoint: #pragma: no cover

    checkpoint: Checkpoint
    if possible_checkpoint is None:
        logger.info("Starting fresh run")
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        checkpoint = create_initial_checkpoint(dataset_name, config, run, is_gpu_run)
    else:
        logger.info("Loading previous checkpoint")
        checkpoint = possible_checkpoint
        random.setstate(checkpoint.random_state)
        np.random.set_state(checkpoint.numpy_random_state)
        torch.set_rng_state(checkpoint.torch_random_state)

    total_generations: int = config['evolutionary']['generations']
    max_epochs: int = config['evolutionary']['max_epochs']
    for gen in range(checkpoint.last_processed_generation + 1, total_generations):
        # check the total number of epochs (stop criteria)
        if checkpoint.total_epochs is not None and checkpoint.total_epochs >= max_epochs:
            break
        checkpoint = engine.evolve(run, grammar, gen, checkpoint, config)

    # compute testing performance of the fittest network
    best_network_path: str = build_overall_best_path(config['checkpoints_path'], run)
    best_test_acc: float = checkpoint.evaluator.testing_performance(best_network_path)
    logger.info(f"Best test accuracy: {best_test_acc}")

    return checkpoint




if __name__ == '__main__': #pragma: no cover
    parser: ArgumentParser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-path", '-c', required=True, help="Path to the config file to be used",
                        type=lambda x: is_yaml_file(parser, x))
    parser.add_argument("--dataset-name", '-d', required=True, help="Name of the dataset to be used",
                        type=str, choices=list(DATASETS_INFO.keys()))
    parser.add_argument("--grammar-path", '-g', required=True, help="Path to the grammar to be used",
                        type=lambda x: is_valid_file(parser, x))
    parser.add_argument("--run", "-r", required=False, help="Identifies the run id and seed to be used",
                        type=int, default=0)
    parser.add_argument("--gpu-enabled", required=False, help="Runs the experiment in the GPU",
                        action='store_true')
    args: Any = parser.parse_args()


    logging.setLogRecordFactory(evodenss.logger_record_factory(args.run))
    logger = logging.getLogger(__name__)


    start = time.time()
    torch.backends.cudnn.benchmark = True
    config: Config = Config(args.config_path) 
    main(run=args.run,
         dataset_name=args.dataset_name,
         config=config,
         grammar=Grammar(args.grammar_path, backup_path=config['checkpoints_path']),
         is_gpu_run=args.gpu_enabled)

    end = time.time()
    time_elapsed = int(end - start)
    secs_elapsed = time_elapsed % 60
    mins_elapsed = time_elapsed//60 % 60
    hours_elapsed = time_elapsed//3600 % 60
    logger.info(f"Time taken to perform run: {compute_time_elapsed_human(time_elapsed)}")
    logging.shutdown()
