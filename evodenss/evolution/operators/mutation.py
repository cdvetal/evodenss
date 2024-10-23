import logging
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, cast

from evodenss.config.pydantic import MutationConfig
from evodenss.evolution.genotype import Genotype
from evodenss.evolution.grammar import Attribute, Derivation, Grammar, NonTerminal, Terminal
from evodenss.evolution.individual import Individual
from evodenss.evolution.operators.mutation_tracker import enable_tracking
from evodenss.misc.enums import AttributeType, MutationType
from evodenss.misc.utils import InputLayerId, LayerId
from evodenss.networks.module import Module

if TYPE_CHECKING:
    from evodenss.evolution.genotype import IndividualGenotype
    from evodenss.evolution.grammar import Symbol
    from evodenss.networks.module import Module


__all__ = ['mutate']

logger = logging.getLogger(__name__)


@enable_tracking(MutationType.ADD_LAYER)
def mutation_add_layer(individual: Individual,
                       module_idx: int,
                       grammar: Grammar,
                       reuse_layer_prob: float,
                       generation: int,
                       **attributes_to_override: list[int]|list[float]) -> Optional[dict[str, Any]]:
    module: Module = list(individual.individual_genotype.modules_dict.values())[module_idx]

    if len(module.layers) >= module.module_configuration.network_structure.max_expansions:
        return None
    
    is_reused: bool = False
    if random.random() <= reuse_layer_prob and len(module.layers) > 0:
        new_layer = random.choice(module.layers)
        is_reused = True
    else:
        new_layer = grammar.initialise(module.module_name, **attributes_to_override)
    insert_pos: LayerId = LayerId(random.randint(0, len(module.layers)))

    #fix connections (old)
    #for _key_ in sorted(module.connections, reverse=True):
    #    if _key_ >= insert_pos:
    #        for value_idx, value in enumerate(module.connections[_key_]):
    #            if value >= insert_pos-1:
    #                module.connections[_key_][value_idx] = InputLayerId(module.connections[_key_][value_idx] + 1)
    #        module.connections[LayerId(_key_+1)] = module.connections.pop(_key_)
    #module.layers.insert(insert_pos, new_layer)

    # fix connections
    for _key_ in sorted(module.connections, reverse=True):
        if _key_ >= insert_pos:
            module.connections[LayerId(_key_)] = [InputLayerId(value + 1) for value in module.connections[_key_]]
            module.connections[LayerId(_key_+1)] = module.connections.pop(_key_)
    module.layers.insert(insert_pos, new_layer)
        
    #make connections of the new layer
    if insert_pos == 0:
        module.connections[insert_pos] = [InputLayerId(-1)]
    else:
        levels_back: int
        if module.module_configuration.levels_back is None:
            levels_back = insert_pos + 1
        else:
            levels_back = module.module_configuration.levels_back
        connection_possibilities: list[InputLayerId] = \
            cast(list[InputLayerId], list(range(max(-1, insert_pos-levels_back), insert_pos-1)))
        sample_size = random.randint(0, len(connection_possibilities))
        module.connections[insert_pos] = [InputLayerId(insert_pos-1)]
        if sample_size > 0:
            module.connections[insert_pos] += random.sample(connection_possibilities, sample_size)

    return {
        "individual": individual,
        "generation": generation,
        "module_idx": module_idx,
        "module_name": module.module_name,
        "layer": grammar.decode(module.module_name, new_layer),
        "insert_pos": insert_pos,
        "is_reused": is_reused
    }


@enable_tracking(mutation_type=MutationType.REMOVE_LAYER)
def mutation_remove_layer(individual: Individual,
                          module_idx: int,
                          grammar: Grammar,
                          generation: int) -> Optional[dict[str, Any]]:
    module: Module = list(individual.individual_genotype.modules_dict.values())[module_idx]

    if len(module.layers) <= module.module_configuration.network_structure.min_expansions:
        return None
    
    remove_idx = LayerId(random.randint(0, len(module.layers)-1))
    track_mutation_data: dict[str, Any] = {
        "individual": individual,
        "generation": generation,
        "module_idx": module_idx,
        "module_name": module.module_name,
        "layer": grammar.decode(module.module_name, module.layers[remove_idx]),
        "remove_idx": remove_idx
    }
    del module.layers[remove_idx]

    #fix connections
    if remove_idx == max(module.connections.keys()):
        module.connections.pop(remove_idx)
    else:
        for _key_ in sorted(module.connections.keys()):
            if _key_ > remove_idx:
                if _key_ > remove_idx+1 and InputLayerId(remove_idx) in module.connections[_key_]:
                    module.connections[_key_].remove(InputLayerId(remove_idx))

                for value_idx, value in enumerate(module.connections[_key_]):
                    if value >= remove_idx:
                        module.connections[_key_][value_idx] = \
                            InputLayerId(module.connections[_key_][value_idx] - 1)
                module.connections[LayerId(_key_-1)] = list(set(module.connections.pop(_key_)))
        if remove_idx == 0:
            module.connections[LayerId(0)] = [InputLayerId(-1)]
    
    return track_mutation_data


def _mutation_dsge(layer: 'Genotype',
                   grammar: Grammar,
                   **attributes_to_override: list[int]|list[float]) -> None:

    nt_keys: list[NonTerminal] = sorted(list(layer.expansions.keys()))
    random_nt: NonTerminal = random.choice(nt_keys)
    nt_derivation_idx: int = random.randint(0, len(layer.expansions[random_nt])-1)
    nt_derivation: Derivation = layer.expansions[random_nt][nt_derivation_idx]

    sge_possibilities: list[list[Symbol]] = []
    node_type_possibilites: list[type[Symbol]] = []
    if len(grammar.grammar[random_nt]) > 1:
        all_possibilities: list[tuple[Symbol, ...]] = \
            [tuple(derivation) for derivation in grammar.grammar[random_nt]]
        # exclude current derivation to avoid neutral mutation
        sge_possibilities = [list(d) for d in set(all_possibilities) - set([tuple(nt_derivation)])]
        node_type_possibilites.append(NonTerminal)

    # stride mutations happen separately
    terminal_symbols_with_attributes: list[Symbol] = \
        list(filter(lambda x: isinstance(x, Terminal) and x.attribute is not None and x.name != "stride",
                    nt_derivation))

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
                    if symbol.name in attributes_to_override.keys():
                        if AttributeType(symbol.attribute.var_type) == AttributeType.FLOAT:
                            cast_attribute_f= cast(Attribute[float], symbol.attribute)
                            cast_attribute_f.override_value(cast(list[float], attributes_to_override[symbol.name]))
                        else:
                            cast_attribute_i = cast(Attribute[int], symbol.attribute)
                            cast_attribute_i.override_value(cast(list[int], attributes_to_override[symbol.name]))
                    else:
                        # this method has side-effects. The Derivation object is altered because of this
                        symbol.attribute.generate()
            layer.expansions[random_nt][nt_derivation_idx] = new_derivation
        else:
            raise AttributeError(f"Invalid value from random_mt_type: [{random_mt_type}]")


@enable_tracking(MutationType.ADD_CONNECTION)
def mutation_add_connection(individual: Individual,
                            module_idx: int,
                            layer_idx: LayerId,
                            generation: int) -> Optional[dict[str, Any]]:
    module: Module = list(individual.individual_genotype.modules_dict.values())[module_idx]

    levels_back: int
    if module.module_configuration.levels_back is None:
        levels_back = layer_idx + 1
    else:
        levels_back = module.module_configuration.levels_back
    connection_possibilities = list(range(max(-1, layer_idx-levels_back), layer_idx-1))
    connection_possibilities = list(set(connection_possibilities) - set(module.connections[LayerId(layer_idx)]))
    
    if len(connection_possibilities) == 0:
        return None

    new_input: InputLayerId = InputLayerId(random.choice(connection_possibilities))
    module.connections[layer_idx].append(new_input)

    return {
        "individual": individual,
        "generation": generation,
        "module_idx": module_idx,
        "module_name": module.module_name,
        "layer_idx": layer_idx,
        "new_input": new_input
    }


@enable_tracking(MutationType.REMOVE_CONNECTION)
def mutation_remove_connection(individual: Individual,
                               module_idx: int,
                               layer_idx: LayerId,
                               generation: int) -> Optional[dict[str, Any]]:
    module: Module = list(individual.individual_genotype.modules_dict.values())[module_idx]

    connection_possibilities = list(set(module.connections[layer_idx]) - set([InputLayerId(layer_idx-1)]))
    if len(connection_possibilities) == 0:
        return None
    
    r_connection: InputLayerId = random.choice(connection_possibilities)
    module.connections[layer_idx].remove(r_connection)

    return {
        "individual": individual,
        "generation": generation,
        "module_idx": module_idx,
        "module_name": module.module_name,
        "layer_idx": layer_idx,
        "removed_input": r_connection
    }


@enable_tracking(mutation_type=MutationType.TRAIN_LONGER)
def mutation_increase_training_time(individual: Individual,
                                    default_train_time: int,
                                    max_train_time: Optional[int],
                                    generation: int) -> dict[str, Any]:

    # double-checks if the mutation should be applied
    if max_train_time is None or individual.total_allocated_train_time < max_train_time:
        individual.total_allocated_train_time += default_train_time
        return {
            "individual": individual,
            "generation": generation,
            "from": individual.total_allocated_train_time - default_train_time,
            "to": individual.total_allocated_train_time
        }
    else:
        return {
            "individual": individual,
            "generation": generation
        }

@enable_tracking(mutation_type=MutationType.DSGE_TOPOLOGICAL)
def mutation_dsge_topological(individual: Individual,
                              module_idx: int,
                              layer_idx: LayerId,
                              grammar: Grammar,
                              generation: int) -> dict[str, Any]:
    module: Module = list(individual.individual_genotype.modules_dict.values())[module_idx]
    old_phenotype: str = grammar.decode(module.module_name, module.layers[layer_idx])
    _mutation_dsge(module.layers[layer_idx], grammar)
    new_phenotype: str = grammar.decode(module.module_name, module.layers[layer_idx])
    track_mutation_data: dict[str, Any] = {
        "individual": individual,
        "generation": generation,
        "module_idx": module_idx,
        "layer_idx": layer_idx,
        "from": old_phenotype,
        "to": new_phenotype
    }
    return track_mutation_data


@enable_tracking(mutation_type=MutationType.DSGE_NON_TOPOLOGICAL)
def mutation_dsge_non_topological(individual: Individual,
                                  symbol: str,
                                  genotype: Genotype,
                                  grammar: Grammar,
                                  generation: int) -> dict[str, Any]:
    old_phenotype: str = grammar.decode(symbol, genotype)
    _mutation_dsge(genotype, grammar)
    new_phenotype: str = grammar.decode(symbol, genotype)

    track_mutation_data: dict[str, Any] = {
        "individual": individual,
        "generation": generation,
        "from": old_phenotype,
        "to": new_phenotype,
        "symbol": symbol
    }
    return track_mutation_data


def mutate(individual: Individual,
           grammar: Grammar,
           generation: int,
           mutation_rates: MutationConfig,
           default_train_time: int,
           max_train_time: Optional[int] = None) -> Individual:

    individual_copy: Individual = deepcopy(individual)

    def should_mutate(mutation_rate: float) -> bool:
        return random.random() <= mutation_rate

    # Train individual for longer - no other mutation is applied EXCEPT if max_train_time has been reached
    if should_mutate(mutation_rates.train_longer):
        # check if max_train_time is defined and has not been reached yet
        if max_train_time is None or individual_copy.total_allocated_train_time < max_train_time:
            mutation_increase_training_time(individual_copy, default_train_time, max_train_time, generation)
            return individual_copy

    # in case the individual is mutated in any of the structural parameters the training time is reset
    individual_copy.total_allocated_train_time = default_train_time
    individual_copy.reset_keys('current_time', 'num_epochs', 'metrics')

    for m_idx, module in enumerate(individual_copy.individual_genotype.modules_dict.values()):
        # TODO: this outer cycle is very dodgy and needs to be reviewed in the future
        for _ in range(random.randint(1, 2)):
            if should_mutate(mutation_rates.remove_layer) is True:
                mutation_remove_layer(individual_copy, m_idx, grammar, generation)
            if should_mutate(mutation_rates.add_layer) is True:
                mutation_add_layer(individual_copy, m_idx, grammar, mutation_rates.reuse_layer, generation)
        for layer_idx in range(len(module.layers)):
            if should_mutate(mutation_rates.dsge_topological) is True:
                mutation_dsge_topological(individual_copy, m_idx, LayerId(layer_idx), grammar, generation)
            if should_mutate(mutation_rates.add_connection) is True:
                mutation_add_connection(individual_copy, m_idx, LayerId(layer_idx), generation)
            if should_mutate(mutation_rates.remove_connection) is True:
                mutation_remove_connection(individual_copy, m_idx, LayerId(layer_idx), generation)

    ind_genotype: IndividualGenotype = individual_copy.individual_genotype
    for symbol_name, genotype in zip(ind_genotype.extra_genotype_start_symbol_names,
                                     ind_genotype.extra_genotype):
        if should_mutate(mutation_rates.dsge_non_topological) is True:
            mutation_dsge_non_topological(individual_copy, symbol_name, genotype, grammar, generation)

    return individual_copy
