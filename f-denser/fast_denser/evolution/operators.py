import random
from copy import deepcopy
from typing import Dict

import numpy as np

from fast_denser.evolution import Grammar, Individual
from fast_denser.misc import persistence

def mutation_dsge(layer, grammar: Grammar):
    """
        DSGE mutations (check DSGE for futher details)


        Parameters
        ----------
        layer : dict
            layer to be mutated (DSGE genotype)

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping
    """

    nt_keys = sorted(list(layer.keys()))
    nt_key = random.choice(nt_keys)
    nt_idx = random.randint(0, len(layer[nt_key])-1)

    sge_possibilities = []
    random_possibilities = []
    if len(grammar.grammar[nt_key]) > 1:
        sge_possibilities = list(set(range(len(grammar.grammar[nt_key]))) -\
                                 set([layer[nt_key][nt_idx]['ge']]))
        random_possibilities.append('ge')

    if layer[nt_key][nt_idx]['ga']:
        random_possibilities.extend(['ga', 'ga'])

    if random_possibilities:
        mt_type = random.choice(random_possibilities)

        if mt_type == 'ga':
            var_name = random.choice(sorted(list(layer[nt_key][nt_idx]['ga'].keys())))
            var_type, min_val, max_val, values = layer[nt_key][nt_idx]['ga'][var_name]
            value_idx = random.randint(0, len(values)-1)

            if var_type == 'int':
                new_val = random.randint(min_val, max_val)
            elif var_type == 'float':
                new_val = values[value_idx]+random.gauss(0, 0.15)
                new_val = np.clip(new_val, min_val, max_val)

            layer[nt_key][nt_idx]['ga'][var_name][-1][value_idx] = new_val

        elif mt_type == 'ge':
            layer[nt_key][nt_idx]['ge'] = random.choice(sge_possibilities)

        else:
            return NotImplementedError



def mutation(individual: Individual,
             grammar: Grammar,
             mutation_config: Dict[str, float],
             default_train_time: int):
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

    # copy so that elite is preserved
    ind: Individual = deepcopy(individual)

    #Train individual for longer - no other mutation is applied
    if random.random() <= train_longer_prob:
        ind.train_time += default_train_time
        return ind


    #in case the individual is mutated in any of the structural parameters
    #the training time is reset
    ind.current_time = 0
    ind.num_epochs = 0
    ind.train_time = default_train_time

    for module in ind.modules:

        #add-layer (duplicate or new)
        for _ in range(random.randint(1,2)):
            if len(module.layers) < module.max_expansions and random.random() <= add_layer_prob:
                if random.random() <= reuse_layer_prob:
                    new_layer = random.choice(module.layers)
                else:
                    new_layer = grammar.initialise(module.module)

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
                    connection_possibilities = list(range(max(0, insert_pos-module.levels_back), insert_pos-1))
                    if len(connection_possibilities) < module.levels_back-1:
                        connection_possibilities.append(-1)

                    sample_size = random.randint(0, len(connection_possibilities))

                    module.connections[insert_pos] = [insert_pos-1]
                    if sample_size > 0:
                        module.connections[insert_pos] += random.sample(connection_possibilities, sample_size)


        #remove-layer
        for _ in range(random.randint(1,2)):
            if len(module.layers) > module.min_expansions and random.random() <= remove_layer_prob:
                remove_idx = random.randint(0, len(module.layers)-1)
                del module.layers[remove_idx]

                #fix connections
                for _key_ in sorted(module.connections):
                    if _key_ > remove_idx:
                        if _key_ > remove_idx+1 and remove_idx in module.connections[_key_]:
                            module.connections[_key_].remove(remove_idx)

                        for value_idx, value in enumerate(module.connections[_key_]):
                            if value >= remove_idx:
                                module.connections[_key_][value_idx] -= 1
                        module.connections[_key_-1] = list(set(module.connections.pop(_key_)))

                if remove_idx == 0:
                    module.connections[0] = [-1]


        for layer_idx, layer in enumerate(module.layers):
            #dsge mutation
            if random.random() <= dsge_layer_prob:
                mutation_dsge(layer, grammar)

            #add connection
            if layer_idx != 0 and random.random() <= add_connection_prob:
                connection_possibilities = list(range(max(0, layer_idx-module.levels_back), layer_idx-1))
                connection_possibilities = list(set(connection_possibilities) - set(module.connections[layer_idx]))
                if len(connection_possibilities) > 0:
                    module.connections[layer_idx].append(random.choice(connection_possibilities))
            #remove connection
            r_value = random.random()
            if layer_idx != 0 and r_value <= remove_connection_prob:
                connection_possibilities = list(set(module.connections[layer_idx]) - set([layer_idx-1]))
                if len(connection_possibilities) > 0:
                    r_connection = random.choice(connection_possibilities)
                    module.connections[layer_idx].remove(r_connection)
    #macro level mutation
    for macro in ind.macro:
        if random.random() <= macro_layer_prob:
            mutation_dsge(macro, grammar)

    return ind

def select_fittest(population,
                   population_fits,
                   grammar,
                   cnn_eval,
                   run: int,
                   generation: int,
                   checkpoint_base_path: str,
                   default_train_time): #pragma: no cover

    #Get best individual just according to fitness
    idx_max = np.argmax(population_fits)
    parent = population[idx_max]

    #however if the parent is not the elite, and the parent is trained for longer, the elite
    #is granted the same evaluation time.
    if parent.train_time > default_train_time:
        retrain_elite = False
        #print(f"parent {idx_max}!=0, {parent.train_time}>{default_train_time}, {population[0].train_time}<{parent.train_time}")
        if idx_max != 0 and population[0].train_time > default_train_time and \
            population[0].train_time < parent.train_time:
            retrain_elite = True
            elite = population[0]
            elite.train_time = parent.train_time
            elite.evaluate(grammar,
                           cnn_eval,
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id),
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id))
            population_fits[0] = elite.fitness

        min_train_time = min([ind.current_time for ind in population])

        #also retrain the best individual that is trained just for the default time
        retrain_10min = False
        if min_train_time < parent.train_time:
            ids_10min = [ind.current_time == min_train_time for ind in population]
            #print(f"parent {[ind.current_time for ind in population]}, {min_train_time}<{parent.train_time}, {parent.train_time}>{default_train_time}")
            if sum(ids_10min) > 0:
                retrain_10min = True
                indvs_10min = np.array(population)[ids_10min]
                max_fitness_10min = max([ind.fitness for ind in indvs_10min])
                idx_max_10min = np.argmax(max_fitness_10min)
                parent_10min = indvs_10min[idx_max_10min]

                parent_10min.train_time = parent.train_time

                parent_10min.evaluate(grammar,
                                      cnn_eval,
                                      persistence.build_individual_path(checkpoint_base_path, run, generation, parent_10min.id),
                                      persistence.build_individual_path(checkpoint_base_path, run, generation, parent_10min.id))

                population_fits[population.index(parent_10min)] = parent_10min.fitness


        #select the fittest amont all retrains and the initial parent
        if retrain_elite:
            if retrain_10min:
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
            if parent_10min.fitness > parent.fitness:
                return deepcopy(parent_10min)
            else:
                return deepcopy(parent)
        else:
            return deepcopy(parent)

    return deepcopy(parent)
