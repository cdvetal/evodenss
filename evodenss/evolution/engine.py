from copy import deepcopy
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from evodenss.config.pydantic import get_config, get_fitness_extra_params
from evodenss.evolution.operators import mutation, selection
from evodenss.evolution.grammar import Grammar
from evodenss.evolution.individual import Individual
from evodenss.misc import persistence
from evodenss.misc.checkpoint import Checkpoint
from evodenss.misc.enums import DownstreamMode, FitnessMetricName, OptimiserType

if TYPE_CHECKING:
    from evodenss.metrics.fitness_metrics import Fitness
    from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetType
    from torch.utils.data import Subset

logger = logging.getLogger(__name__)

@persistence.SaveCheckpoint
def evolve(run: int,
           generation: int,
           dataset: dict['DatasetType', 'Subset[ConcreteDataset]'],
           grammar: Grammar,
           checkpoint: Checkpoint) -> Checkpoint:

    logger.info(f"Performing generation: {generation}")
    population: list[Individual]
    population_fits: list[Fitness] = []

    lambda_: int = get_config().evolutionary.lambda_

    if generation == 0:
        logger.info("Creating the initial population")

        #create initial population
        population = [
            Individual(grammar,
                       get_config().network.architecture,
                       _id_,
                       True)
            for _id_ in range(lambda_)
        ]

        #set initial population variables and evaluate population
        for idx, ind in enumerate(population):
            ind.id = idx
            ind.total_allocated_train_time = get_config().network.learning.default_train_time
            ind.reset_keys('current_time', 'num_epochs', 'total_training_time_spent')
            population_fits.append(
                ind.evaluate(
                    grammar,
                    dataset,
                    checkpoint.evaluator,
                    get_config().network.architecture.projector,
                    persistence.build_individual_path(get_config().checkpoints_path, run, generation, idx)
                )
            )
    else:
        assert checkpoint.parent is not None

        logger.info("Applying mutation operators")

        # generate offspring (by mutation)
        offspring_before_mutation: list[Individual] = [deepcopy(checkpoint.parent) for _ in range(lambda_)]
        for idx in range(len(offspring_before_mutation)):
            offspring_before_mutation[idx].reset_keys('total_training_time_spent')
            offspring_before_mutation[idx].id = idx + 1
        offspring: list[Individual] = \
            [mutation.mutate(ind,
                             grammar,
                             generation,
                             get_config().evolutionary.mutation,
                             get_config().network.learning.default_train_time)
             for ind in offspring_before_mutation]

        assert checkpoint.parent is not None
        population = [deepcopy(checkpoint.parent)] + offspring

        # set elite variables to re-evaluation
        population[0].reset_keys('id', 'current_time', 'num_epochs', 'metrics')

        # evaluate population
        for idx, ind in enumerate(population):
            population_fits.append(
                ind.evaluate(
                    grammar,
                    dataset,
                    checkpoint.evaluator,
                    get_config().network.architecture.projector,
                    persistence.build_individual_path(get_config().checkpoints_path, run, generation, idx),
                    persistence.build_individual_path(get_config().checkpoints_path,
                                                      run,
                                                      generation-1,
                                                      checkpoint.parent.id)
                )
            )
        

    logger.info("Selecting the fittest individual")
    selection_method: str = 'fittest'
    #selection_method_params: Optional[dict] = None
    # select parent
    parent: Individual = selection.select_fittest(
        selection_method,
        population,
        grammar,
        dataset,
        checkpoint.evaluator,
        get_config().network.architecture.projector,
        run,
        generation,
        get_config().checkpoints_path,
        get_config().network.learning.default_train_time
    )
    assert parent.fitness is not None

    logger.info(f"Fitnesses: {population_fits}")

    # update best individual
    best_individual_path: str = persistence.build_individual_path(get_config().checkpoints_path,
                                                                  run,
                                                                  generation,
                                                                  parent.id)
    if checkpoint.best_fitness is None or parent.fitness > checkpoint.best_fitness:
        checkpoint.best_fitness = parent.fitness
        persistence.save_overall_best_individual(best_individual_path, parent)
    fitness_metric_name: FitnessMetricName = get_config().evolutionary.fitness.metric_name
    best_test_acc: float
    if fitness_metric_name == FitnessMetricName.KNN_ACCURACY:
        best_test_acc = \
            checkpoint.evaluator.testing_performance(
                dataset=dataset,
                model_dir=best_individual_path,
                fitness_metric_name=FitnessMetricName.KNN_ACCURACY,
                dataset_name=checkpoint.evaluator.dataset_name,
                **get_fitness_extra_params())
        best_test_acc_linear: float = \
            checkpoint.evaluator.testing_performance(
                dataset=dataset,
                model_dir=best_individual_path,
                fitness_metric_name=FitnessMetricName.DOWNSTREAM_ACCURACY,
                dataset_name=checkpoint.evaluator.dataset_name,
                batch_size=2048,
                downstream_mode=DownstreamMode.finetune,
                downstream_epochs=50,
                optimiser_type=OptimiserType.ADAM,
                optimiser_parameters={'lr': 0.001, 'weight_decay': 0.000001, 'beta1': 0.9, 'beta2': 0.999},
                **get_fitness_extra_params()
                )
        logger.info(f"Generation best test accuracy (KNN): {best_test_acc}")
        logger.info(f"Generation best test accuracy (Linear): {best_test_acc_linear}")
    else:
        best_test_acc = \
            checkpoint.evaluator.testing_performance(
                dataset,
                best_individual_path,
                fitness_metric_name,
                model_saving_dir=persistence.build_individual_path(
                    get_config().checkpoints_path, run, generation, idx
                ),
                **get_fitness_extra_params())

        logger.info(f"Generation best test accuracy: {best_test_acc}")

    logger.info(f"Best fitness of generation {generation}: {max(population_fits)}")
    logger.info(f"Best overall fitness: {checkpoint.best_fitness}\n\n\n")

    return Checkpoint(
        run=run,
        random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        torch_random_state=torch.get_rng_state(),
        last_processed_generation=generation,
        total_epochs=checkpoint.total_epochs + sum([ind.num_epochs for ind in population]),
        best_fitness=checkpoint.best_fitness,
        evaluator=checkpoint.evaluator,
        population=population,
        parent=parent,
        best_gen_ind_test_accuracy=best_test_acc
    )
