# type: ignore
from copy import deepcopy
import random
from typing import Any, Dict, List
import unittest
import warnings

from evodenss.evolution import Individual, operators
from evodenss.evolution.grammar import Genotype, Grammar, NonTerminal
from evodenss.networks import ModuleConfig
from tests.resources.genotype_examples import mutation_added_layer_genotype, simple_sample1

class Test(unittest.TestCase):

    def setUp(self):
        self.mutation_config = {
            'add_connection': 0.0,
            'remove_connection': 0.0,
            'add_layer': 0.0,
            'reuse_layer': 0.0,
            'remove_layer': 0.0,
            'dsge_layer': 0.0,
            'macro_layer': 0.0,
            'train_longer': 0.0
        }
        warnings.simplefilter('ignore', category=DeprecationWarning)


    def assert_layers_mutation(self,
                               obtained_layers: List[Genotype],
                               original_layers: List[Genotype],
                               added_layers_idx: List[int],
                               added_layers: List[Genotype],
                               removed_layers_idx: List[int]):
        expected_layers = deepcopy(original_layers)
        for idx, l in zip(added_layers_idx, added_layers):
            expected_layers = expected_layers[:idx] + [l] + expected_layers[idx:]
        for idx in removed_layers_idx:
            expected_layers = expected_layers[:idx] + expected_layers[idx+1:]
        self.assertEqual(obtained_layers, expected_layers)


    def count_unique_layers(self, modules):
        unique_layers = []
        for module in modules:
            for layer in module.layers:
                unique_layers.append(id(layer))

        return len(set(unique_layers))


    def count_layers(self, modules):
        return sum([len(module.layers) for module in modules])


    def create_individual(self):
        seed: int = 0
        grammar = Grammar('tests/resources/example_full.grammar')
        network_config: Dict[str, Any] = {
            'reuse_layer': 0.0,
            'macro_structure': ['learning'],
            'output': 'softmax',
            'modules': {
                'features': ModuleConfig(min_expansions=1,
                                         max_expansions=3,
                                         initial_network_structure=[2],
                                         levels_back=1)
            }
        }
        ind: Individual = Individual(
            network_architecture_config=network_config,
            ind_id=seed,
            seed=seed
        )
        ind.initialise(grammar, network_config['reuse_layer'])
        return ind, grammar



    def test_add_layer_random(self):
        random.seed(0)
        self.mutation_config['add_layer'] = 1.0
        ind, grammar = self.create_individual()
        new_ind = operators.mutation(deepcopy(ind), grammar, self.mutation_config, 60)
        self.mutation_config['add_layer'] = 0.0
        obtained_layers = new_ind.modules[0].layers
        original_layers = ind.modules[0].layers
        connections = ind.modules[0].connections
        obtained_connections = new_ind.modules[0].connections
        self.assert_layers_mutation(obtained_layers,
                                    original_layers,
                                    [1],
                                    [mutation_added_layer_genotype],
                                    [])
        self.assertEqual(obtained_connections, {**connections, **{2: [1]}})
        self.assertEqual(self.count_layers(ind.modules)+1,
                         self.count_layers(new_ind.modules),
                         "Error: add layer wrong size")


    def test_add_layer_replicate(self):
        random.seed(0)
        self.mutation_config['add_layer'] = 1.0
        self.mutation_config['reuse_layer'] = 1.0

        ind, grammar = self.create_individual()
        new_ind = operators.mutation(ind, grammar, self.mutation_config, 60)

        self.mutation_config['add_layer'] = 0.0
        self.mutation_config['reuse_layer'] = 0.0
        obtained_layers = new_ind.modules[0].layers
        original_layers = ind.modules[0].layers
        connections = ind.modules[0].connections
        obtained_connections = new_ind.modules[0].connections
        self.assert_layers_mutation(obtained_layers,
                                    original_layers,
                                    [1],
                                    [original_layers[0]],
                                    [])
        self.assertEqual(obtained_connections, {**connections, **{2: [1]}})
        self.assertEqual(self.count_unique_layers(ind.modules),
                         self.count_unique_layers(new_ind.modules),
                         "Error: duplicate layer wrong size")
        self.assertEqual(self.count_layers(ind.modules)+1,
                         self.count_layers(new_ind.modules),
                         "Error: duplicate layer wrong size")


    def test_remove_layer(self):
        random.seed(0)
        self.mutation_config['remove_layer'] = 1.0
        ind, grammar = self.create_individual()
        connections = ind.modules[0].connections
        new_ind = operators.mutation(deepcopy(ind), grammar, self.mutation_config, 60)
        self.mutation_config['remove_layer'] = 0.0
        obtained_layers = new_ind.modules[0].layers
        original_layers = ind.modules[0].layers

        obtained_connections = new_ind.modules[0].connections
        connections.pop(1)

        self.assert_layers_mutation(obtained_layers,
                                    original_layers,
                                    [],
                                    [],
                                    [1])
        self.assertEqual(obtained_connections, connections)
        self.assertEqual(self.count_layers(ind.modules)-1,
                         self.count_layers(new_ind.modules),
                         "Error: remove layer wrong size")


    def test_dsge_mutation(self):
        random.seed(0)
        grammar = Grammar("tests/resources/simple_grammar.grammar")
        sample_to_mutate = deepcopy(simple_sample1)
        expected_sample = deepcopy(simple_sample1)
        expected_sample.expansions[NonTerminal(name='value')][1] = [NonTerminal(name='var')]
        expected_sample.codons[NonTerminal(name='value')][1] = 1
        operators.mutation_dsge(sample_to_mutate, grammar)

        self.assertEqual(sample_to_mutate, expected_sample)

if __name__ == '__main__':
    unittest.main()
