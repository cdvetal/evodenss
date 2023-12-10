from copy import deepcopy
import logging
import random
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple

import numpy as np

from evodenss.evolution import Individual
from evodenss.evolution.grammar import Derivation, Grammar, NonTerminal, Terminal
from evodenss.misc import persistence

if TYPE_CHECKING:
    from evodenss.networks.torch import BaseEvaluator
    from evodenss.misc.fitness_metrics import Fitness
    from evodenss.evolution.grammar import Genotype, Symbol

logger = logging.getLogger(__name__)


def mutation_dsge(layer: 'Genotype', grammar: Grammar) -> None:
    nt_keys: List[NonTerminal] = sorted(list(layer.expansions.keys()))
    random_nt: NonTerminal = random.choice(nt_keys)
    nt_derivation_idx: int = random.randint(0, len(layer.expansions[random_nt])-1)
    nt_derivation: Derivation = layer.expansions[random_nt][nt_derivation_idx]

    sge_possibilities: List[List[Symbol]] = []
    node_type_possibilites: List[type[Symbol]] = []
    if len(grammar.grammar[random_nt]) > 1:
        all_possibilities: List[Tuple[Symbol, ...]] = \
            [tuple(derivation) for derivation in grammar.grammar[random_nt]]
        # exclude current derivation to avoid neutral mutation
        sge_possibilities = [list(d) for d in set(all_possibilities) - set([tuple(nt_derivation)])]
        node_type_possibilites.append(NonTerminal)

    terminal_symbols_with_attributes: List[Symbol] = \
        list(filter(lambda x: isinstance(x, Terminal) and x.attribute is not None, nt_derivation))

    if terminal_symbols_with_attributes:
        node_type_possibilites.extend([Terminal, Terminal])

    if node_type_possibilites:
        random_mt_type: type[Symbol] = random.choice(node_type_possibilites)
        if random_mt_type is Terminal:
            symbol_to_mutate: Symbol = random.choice(terminal_symbols_with_attributes)
            assert isinstance(symbol_to_mutate, Terminal) and \
                symbol_to_mutate.attribute is not None and \
                symbol_to_mutate.attribute.values is not None
            is_neutral_mutation: bool = True
            while is_neutral_mutation is True:
                current_values = tuple(symbol_to_mutate.attribute.values)
                symbol_to_mutate.attribute.generate()
                new_values = tuple(symbol_to_mutate.attribute.values)
                if current_values != new_values:
                    is_neutral_mutation = False
        elif random_mt_type is NonTerminal:
            # assignment with side-effect.
            # layer variable will also be affected
            new_derivation: Derivation = deepcopy(Derivation(random.choice(sge_possibilities)))
            # this line is here because otherwise the index function
            # will not be able to find the derivation after we generate values
            layer.codons[random_nt][nt_derivation_idx] = grammar.grammar[random_nt].index(new_derivation)
            for symbol in new_derivation:
                if isinstance(symbol, Terminal) and symbol.attribute is not None:
                    assert symbol.attribute.values is None
                    symbol.attribute.generate()
            layer.expansions[random_nt][nt_derivation_idx] = new_derivation
        else:
            raise AttributeError(f"Invalid value from random_mt_type: [{random_mt_type}]")


def mutation(individual: Individual,
             grammar: Grammar,
             mutation_config: Dict[str, float],
             default_train_time: int) -> Individual:
    """
        Network mutations: add and remove layer, add and remove connections, macro structure


        Parameters
        ----------
        individual : Individual
            individual to be mutated

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping

        default_train_time : int
            default training time

        Returns
        -------
        ind : Individual
            mutated individual
    """

    add_layer_prob: float = mutation_config['add_layer']
    reuse_layer_prob: float = mutation_config['reuse_layer']
    remove_layer_prob: float = mutation_config['remove_layer']
    add_connection_prob: float = mutation_config['add_connection']
    remove_connection_prob: float = mutation_config['remove_connection']
    dsge_layer_prob: float = mutation_config['dsge_layer']
    macro_layer_prob: float = mutation_config['macro_layer']
    train_longer_prob: float = mutation_config['train_longer']
    individual_copy: Individual = deepcopy(individual)

    #Train individual for longer - no other mutation is applied
    if random.random() <= train_longer_prob:
        individual_copy.total_allocated_train_time += default_train_time
        logger.info(f"Individual {individual_copy.id} total train time is going "
                    f"to be extended to {individual_copy.total_allocated_train_time}")
        return individual_copy


    #in case the individual is mutated in any of the structural parameters
    #the training time is reset
    individual_copy.current_time = 0
    individual_copy.num_epochs = 0
    individual_copy.total_allocated_train_time = default_train_time
    individual_copy.metrics = None

    for m_idx, module in enumerate(individual_copy.modules):

        #add-layer (duplicate or new)
        for _ in range(random.randint(1,2)):
            if len(module.layers) < module.module_configuration.max_expansions and random.random() <= add_layer_prob:
                if random.random() <= reuse_layer_prob and len(module.layers) > 0:
                    new_layer = random.choice(module.layers)
                else:
                    new_layer = grammar.initialise(module.module_name)

                insert_pos: int = random.randint(0, len(module.layers))
                #fix connections
                for _key_ in sorted(module.connections, reverse=True):
                    if _key_ >= insert_pos:
                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= insert_pos-1:
                                module.connections[_key_][value_idx] += 1

                        module.connections[_key_+1] = module.connections.pop(_key_)


                module.layers.insert(insert_pos, new_layer)

                #make connections of the new layer
                if insert_pos == 0:
                    module.connections[insert_pos] = [-1]
                else:
                    connection_possibilities = list(range(max(0, insert_pos-module.module_configuration.levels_back),
                                                          insert_pos-1))
                    if len(connection_possibilities) < module.module_configuration.levels_back-1:
                        connection_possibilities.append(-1)

                    sample_size = random.randint(0, len(connection_possibilities))

                    module.connections[insert_pos] = [insert_pos-1]
                    if sample_size > 0:
                        module.connections[insert_pos] += random.sample(connection_possibilities, sample_size)

                logger.info(f"Individual {individual_copy.id} is going to have an extra layer at "
                            f"Module {m_idx}: {module.module_name}; position {insert_pos}")
                #print("**** new layer: ", new_layer)
        #remove-layer
        for _ in range(random.randint(1,2)):
            if len(module.layers) > module.module_configuration.min_expansions and random.random() <= remove_layer_prob:
                remove_idx = random.randint(0, len(module.layers)-1)

                #print("**** removed layer: ", module.layers[remove_idx])
                del module.layers[remove_idx]

                #fix connections
                if remove_idx == max(module.connections.keys()):
                    module.connections.pop(remove_idx)
                else:
                    for _key_ in sorted(module.connections.keys()):
                        if _key_ > remove_idx:
                            if _key_ > remove_idx+1 and remove_idx in module.connections[_key_]:
                                module.connections[_key_].remove(remove_idx)

                            for value_idx, value in enumerate(module.connections[_key_]):
                                if value >= remove_idx:
                                    module.connections[_key_][value_idx] -= 1
                            module.connections[_key_-1] = list(set(module.connections.pop(_key_)))
                    if remove_idx == 0:
                        module.connections[0] = [-1]
                logger.info(f"Individual {individual_copy.id} is going to have a layer removed from "
                            f"Module {m_idx}: {module.module_name}; position {remove_idx}")

        for layer_idx, layer in enumerate(module.layers):
            #dsge mutation
            if random.random() <= dsge_layer_prob:
                mutation_dsge(layer, grammar)
                logger.info(f"Individual {individual_copy.id} is going to have a DSGE mutation on "
                            f"Module {m_idx}: {module.module_name}; position {layer_idx}")

            #add connection
            if layer_idx != 0 and random.random() <= add_connection_prob:
                connection_possibilities = list(range(max(0, layer_idx-module.module_configuration.levels_back),
                                                      layer_idx-1))
                connection_possibilities = list(set(connection_possibilities) - set(module.connections[layer_idx]))
                if len(connection_possibilities) > 0:
                    new_input: int = random.choice(connection_possibilities)
                    module.connections[layer_idx].append(new_input)
                logger.info(f"Individual {individual_copy.id} is going to have a new connection "
                            f"Module {m_idx}: {module.module_name}; layer {layer_idx}")
            #remove connection
            r_value = random.random()
            if layer_idx != 0 and r_value <= remove_connection_prob:
                connection_possibilities = list(set(module.connections[layer_idx]) - set([layer_idx-1]))
                if len(connection_possibilities) > 0:
                    r_connection = random.choice(connection_possibilities)
                    module.connections[layer_idx].remove(r_connection)
                    logger.info(f"Individual {individual_copy.id} is going to have a connection removed from "
                                f"Module {m_idx}: {module.module_name}; layer {layer_idx}")
    #macro level mutation
    for macro in individual_copy.macro:
        if random.random() <= macro_layer_prob:
            mutation_dsge(macro, grammar)
            logger.info(f"Individual {individual_copy.id} is going to have a macro mutation")

    #print("****************************")
    #print("****************************")
    #print("MUTATED INDIVIDUAL:")
    #for i, module in enumerate(individual_copy.modules):
    #    for layer_idx, layer_genotype in enumerate(module.layers):
    #        print("module: ", i, "layer_idx: ", layer_idx, layer_genotype)
    #        print("--")
    #print("****************************")
    #print("****************************")
    return individual_copy

def select_fittest(population: List[Individual],
                   population_fits: List['Fitness'],
                   grammar: Grammar,
                   cnn_eval: 'BaseEvaluator',
                   static_projector_config: Optional[List[int]],
                   run: int,
                   generation: int,
                   checkpoint_base_path: str,
                   default_train_time: int) -> Individual: #pragma: no cover

    #Get best individual just according to fitness
    elite: Individual
    parent_10min: Individual
    idx_max: int = np.argmax(population_fits) #  type: ignore
    parent: Individual = population[idx_max]
    assert parent.fitness is not None
    logger.info(f"Parent: idx: {idx_max}, id: {parent.id}")
    logger.info(f"Training times: {[ind.current_time for ind in population]}")
    logger.info(f"ids: {[ind.id for ind in population]}")

    #however if the parent is not the elite, and the parent is trained for longer, the elite
    #is granted the same evaluation time.
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
                           cnn_eval,
                           static_projector_config,
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id),
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id))
            population_fits[0] = elite.fitness

        min_train_time = min([ind.current_time for ind in population])

        #also retrain the best individual that is trained just for the default time
        retrain_10min = False
        if min_train_time < parent.total_allocated_train_time:
            ids_10min = [ind.current_time == min_train_time for ind in population]
            logger.info(f"Individuals trained for the minimum time: {ids_10min}")
            if sum(ids_10min) > 0:
                retrain_10min = True
                indvs_10min = np.array(population)[ids_10min]
                max_fitness_10min = max([ind.fitness for ind in indvs_10min])
                idx_max_10min = np.argmax([ind.fitness for ind in indvs_10min])
                parent_10min = indvs_10min[idx_max_10min]
                assert parent_10min.fitness is not None

                parent_10min.total_allocated_train_time = parent.total_allocated_train_time
                logger.info(f"Min train time parent: idx: {idx_max_10min}, id: {parent_10min.id},"
                            f" max fitness detected: {max_fitness_10min}")
                logger.info(f"Fitnesses from min train individuals before selecting best individual:"
                            f" {[ind.fitness for ind in indvs_10min]}")
                logger.info(f"Individual {parent_10min.id} has its train extended."
                            f" Current fitness {parent_10min.fitness}")
                path: str = persistence.build_individual_path(checkpoint_base_path, run, generation, parent_10min.id)
                print(f"Reusing individual from path: {path}")
                print(f"Static projector config: {static_projector_config}")
                print("Macro genotype from parent 10 min: ", parent_10min.macro)
                parent_10min.evaluate(grammar,
                                      cnn_eval,
                                      static_projector_config,
                                      path,
                                      path)

                population_fits[population.index(parent_10min)] = parent_10min.fitness


        #select the fittest among all retrains and the initial parent
        if retrain_elite:
            assert elite.fitness is not None
            if retrain_10min:
                assert parent_10min.fitness is not None
                if parent_10min.fitness > elite.fitness and parent_10min.fitness > parent.fitness:
                    return deepcopy(parent_10min)
                elif elite.fitness > parent_10min.fitness and elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
            else:
                if elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
        elif retrain_10min:
            assert parent_10min.fitness is not None
            if parent_10min.fitness > parent.fitness:
                return deepcopy(parent_10min)
            else:
                return deepcopy(parent)
        else:
            return deepcopy(parent)

    return deepcopy(parent)
