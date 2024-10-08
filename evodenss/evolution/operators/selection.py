import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np

from evodenss.evolution.individual import Individual
from evodenss.evolution.grammar import Grammar
from evodenss.misc import persistence

if TYPE_CHECKING:
    from evodenss.networks.evaluators import BaseEvaluator
    from evodenss.dataset.dataset_loader import DatasetType
    from torch.utils.data import Subset

logger = logging.getLogger(__name__)

def select_fittest(method: str,
                   population: list[Individual],
                   grammar: Grammar,
                   dataset: dict['DatasetType', 'Subset'],
                   cnn_eval: 'BaseEvaluator',
                   static_projector_config: Optional[list[int]],
                   run: int,
                   generation: int,
                   checkpoint_base_path: str,
                   default_train_time: int) -> Individual:

    elite: Individual
    parent: Individual
    parent_dt: Individual  # dt stands for default time

    idx_max: int
    if method == 'fittest':
        # Get best individual just according to fitness
        idx_max = np.argmax([ind.fitness for ind in population]) # type: ignore
        parent = population[idx_max]
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    assert parent.fitness is not None
    logger.info(f"Parent: idx: {idx_max}, id: {parent.id}")
    logger.info(f"Training times: {[ind.current_time for ind in population]}")
    logger.info(f"ids: {[ind.id for ind in population]}")

    # however if the parent is not the elite, and the parent is trained for longer, the elite
    # is granted the same evaluation time.
    if parent.total_allocated_train_time > default_train_time:
        retrain_elite = False
        if idx_max != 0 and population[0].total_allocated_train_time > default_train_time and \
            population[0].total_allocated_train_time < parent.total_allocated_train_time:
            logger.info("Elite train was extended, since parent was trained for longer")
            retrain_elite = True
            elite = population[0]
            assert elite.fitness is not None
            elite.total_allocated_train_time = parent.total_allocated_train_time
            elite.evaluate(grammar,
                           dataset,
                           cnn_eval,
                           static_projector_config,
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id),
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id))

        min_train_time = min([ind.current_time for ind in population])

        #also retrain the best individual that is trained just for the default time
        retrain_dt = False
        if min_train_time < parent.total_allocated_train_time:
            ids_dt = [ind.current_time == min_train_time for ind in population]
            logger.info(f"Individuals trained for the minimum time: {ids_dt}")
            if sum(ids_dt) > 0:
                retrain_dt = True
                indvs_dt = np.array(population)[ids_dt]
                max_fitness_dt = max([ind.fitness for ind in indvs_dt])
                idx_max_dt = np.argmax([ind.fitness for ind in indvs_dt])
                parent_dt = indvs_dt[idx_max_dt]
                assert parent_dt.fitness is not None

                parent_dt.total_allocated_train_time = parent.total_allocated_train_time
                logger.info(f"Min train time parent: idx: {idx_max_dt}, id: {parent_dt.id},"
                            f" max fitness detected: {max_fitness_dt}")
                logger.info(f"Fitnesses from min train individuals before selecting best individual:"
                            f" {[ind.fitness for ind in indvs_dt]}")
                logger.info(f"Individual {parent_dt.id} has its train extended."
                            f" Current fitness {parent_dt.fitness}")
                path: str = persistence.build_individual_path(checkpoint_base_path, run, generation, parent_dt.id)
                print(f"Reusing individual from path: {path}")
                print(f"Static projector config: {static_projector_config}")
                parent_dt.evaluate(grammar,
                                   dataset,
                                   cnn_eval,
                                   static_projector_config,
                                   path,
                                   path)

        #select the fittest among all retrains and the initial parent
        assert parent_dt.fitness is not None
        if retrain_elite:
            assert elite.fitness is not None
            if retrain_dt:
                if parent_dt.fitness > elite.fitness and parent_dt.fitness > parent.fitness:
                    return deepcopy(parent_dt)
                elif elite.fitness > parent_dt.fitness and elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
            else:
                if elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
        elif retrain_dt:
            if parent_dt.fitness > parent.fitness:
                return deepcopy(parent_dt)
            else:
                return deepcopy(parent)
        else:
            return deepcopy(parent)

    return deepcopy(parent)
