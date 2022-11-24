from copy import deepcopy
import logging
import os
import random
from typing import List

from fast_denser.config import Config
from fast_denser.evolution import Grammar, Individual, operators
from fast_denser.misc import Checkpoint, persistence
from fast_denser.misc.fitness_metrics import Fitness

import numpy as np
import torch

logger = logging.getLogger(__name__)

@persistence.SaveCheckpoint
def evolve(run: int,
           grammar: Grammar,
           generation: int,
           checkpoint: Checkpoint,
           config: Config) -> Checkpoint:

    logger.info(f"Performing generation: {generation}")

    population: List[Individual]
    population_fits: List[Fitness]
    if generation == 0:
        logger.info("Creating the initial population")

        #population = initialise_population(config) if checkpoint.population is None else checkpoint.population

        #create initial population
        population = [
            Individual(config['network']['architecture'], _id_, seed=run) \
                .initialise(grammar,
                            config['network']['architecture']['levels_back'],
                            config['network']['architecture']['reuse_layer'],
                            config['network']['architecture']['network_structure_init'])
            for _id_ in range(config['evolutionary']['lambda'])
        ]

        #set initial population variables and evaluate population
        population_fits = []
        for idx, ind in enumerate(population):
            ind.current_time = 0
            ind.num_epochs = 0
            ind.total_training_time_spent = 0.0 # TODO: This might not be right
            ind.total_allocated_train_time = config['network']['learning']['default_train_time']
            ind.id = idx
            population_fits.append(
                ind.evaluate(grammar,
                             checkpoint.evaluator,
                             persistence.build_individual_path(config['checkpoints_path'], run, generation, idx))
            )

    else:
        assert checkpoint.parent is not None

        logger.info("Applying mutation operators")
        
        lambd: int = config['evolutionary']['lambda']
        # generate offspring (by mutation)
        offspring_before_mutation: List[Individual] = [deepcopy(checkpoint.parent) for _ in range(lambd)]
        for idx in range(len(offspring_before_mutation)):
            offspring_before_mutation[idx].total_training_time_spent = 0.0 # TODO: This might not be right
            offspring_before_mutation[idx].id = idx + 1
        offspring: List[Individual] = \
            [operators.mutation(ind,
                                grammar,
                                config['evolutionary']['mutation'],
                                config['network']['learning']['default_train_time'])
             for ind in offspring_before_mutation]

        assert checkpoint.parent is not None
        population = [deepcopy(checkpoint.parent)] + offspring

        # set elite variables to re-evaluation
        population[0].current_time = 0
        population[0].num_epochs = 0
        population[0].id = 0
        population[0].metrics = None

        # evaluate population
        population_fits = []
        for idx, ind in enumerate(population):
            population_fits.append(
                ind.evaluate(
                    grammar,
                    checkpoint.evaluator,
                    persistence.build_individual_path(config['checkpoints_path'], run, generation, idx),
                    persistence.build_individual_path(config['checkpoints_path'], run, generation-1, checkpoint.parent.id)
                )
            )

    logger.info("Selecting the fittest individual")
    # select parent
    parent = operators.select_fittest(
                population,
                population_fits,
                grammar,
                checkpoint.evaluator,
                run,
                generation,
                config['checkpoints_path'],
                config['network']['learning']["default_train_time"])
    assert parent.fitness is not None

    logger.info(f"Fitnesses: {population_fits}")

    # update best individual
    best_individual_path: str = persistence.build_individual_path(config['checkpoints_path'], run, generation, parent.id)
    if checkpoint.best_fitness is None or parent.fitness > checkpoint.best_fitness:
        checkpoint.best_fitness = parent.fitness
        persistence.save_overall_best_individual(best_individual_path, parent)
    best_test_acc: float = checkpoint.evaluator.testing_performance(best_individual_path)

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
        parent=parent
    )
