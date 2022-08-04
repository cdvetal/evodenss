from copy import deepcopy
import logging
import os
import random
from typing import List

from fast_denser.config import Config
from fast_denser.evolution import Grammar, Individual, operators
from fast_denser.misc import Checkpoint, persistence

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
    population_fits: List[float]
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
            ind.train_time = config['network']['learning']['default_train_time']
            ind.id = idx
            population_fits.append(
                ind.evaluate(grammar,
                             checkpoint.evaluator,
                             persistence.build_individual_path(config['checkpoints_path'], run, generation, idx))
            )

    else:
        assert checkpoint.parent is not None

        # generate offspring (by mutation)
        offspring: List[Individual] = \
            [operators.mutation(checkpoint.parent,
                                grammar,
                                config['evolutionary']['mutation'],
                                config['network']['learning']['default_train_time'])
             for _ in range(config['evolutionary']['lambda'])]

        assert checkpoint.parent is not None
        population = [deepcopy(checkpoint.parent)] + offspring

        # set elite variables to re-evaluation
        population[0].current_time = 0
        population[0].num_epochs = 0

        # evaluate population
        population_fits = []
        for idx, ind in enumerate(population):
            ind.total_training_time_spent = 0.0 # TODO: This might not be right
            ind.id = idx
            population_fits.append(
                ind.evaluate(
                    grammar,
                    checkpoint.evaluator,
                    persistence.build_individual_path(config['checkpoints_path'], run, generation, idx),
                    persistence.build_individual_path(config['checkpoints_path'], run, generation-1, checkpoint.parent.id)
                )
            )

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

    print(population_fits)

    # update best individual
    if checkpoint.best_fitness is None or parent.fitness > checkpoint.best_fitness:
        checkpoint.best_fitness = parent.fitness
        best_individual_path: str = persistence.build_individual_path(config['checkpoints_path'], run, generation, parent.id)
        persistence.save_overall_best_individual(best_individual_path, parent)

    logger.info(f"Best fitness of generation {generation}: {max(population_fits)}")
    logger.info(f"Best overall fitness: {checkpoint.best_fitness}")

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
