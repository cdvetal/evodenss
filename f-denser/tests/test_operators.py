# type: ignore
import random
import unittest
import warnings

from fast_denser.evolution import Individual, operators
from fast_denser.evolution.grammar import Grammar

class Test(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=DeprecationWarning)


    def count_unique_layers(self, modules):
        unique_layers = []
        for module in modules:
            for layer in module.layers:
                unique_layers.append(id(layer))

        return len(set(unique_layers))


    def count_layers(self, modules):
        return sum([len(module.layers) for module in modules])


    def create_individual(self):
        network_structure = [["features", 1, 3]]
        grammar = Grammar('tests/utils/example_full.grammar')
        levels_back = {"features": 1, "classification": 1}
        network_structure_init = {"features":[2]}
        ind = Individual({'network_structure': network_structure,
                          'output': "softmax",
                          'macro_structure': ["learning"]},
                         0,
                         0).initialise(grammar, levels_back, 0, network_structure_init)
        print(ind._decode(grammar))
        return ind, grammar



    def test_add_layer_random(self):
        random.seed(0)
        mutation_config = {
            'add_connection': 0.0,
            'remove_connection': 0.0,
            'add_layer': 1.0,
            'reuse_layer': 0.0,
            'remove_layer': 0.0,
            'dsge_layer': 0.0,
            'macro_layer': 0.0,
            'train_longer': 0.0
        }
        ind, grammar = self.create_individual()
        new_ind = operators.mutation(ind, grammar, mutation_config, 60)
        self.assertEqual(self.count_unique_layers(ind.modules)+1, self.count_unique_layers(new_ind.modules), "Error: add layer wrong size")
        self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: add layer wrong size")


    def test_add_layer_replicate(self):
        random.seed(0)
        mutation_config = {
            'add_connection': 0.0,
            'remove_connection': 0.0,
            'add_layer': 1.0,
            'reuse_layer': 1.0,
            'remove_layer': 0.0,
            'dsge_layer': 0.0,
            'macro_layer': 0.0,
            'train_longer': 0.0
        }
        ind, grammar = self.create_individual()
        new_ind = operators.mutation(ind, grammar, mutation_config, 60)

        self.assertEqual(self.count_unique_layers(ind.modules), self.count_unique_layers(new_ind.modules), "Error: duplicate layer wrong size")
        self.assertEqual(self.count_layers(ind.modules)+1, self.count_layers(new_ind.modules), "Error: duplicate layer wrong size")


    def test_remove_layer(self):
        random.seed(0)
        mutation_config = {
            'add_connection': 0.0,
            'remove_connection': 0.0,
            'add_layer': 0.0,
            'reuse_layer': 0.0,
            'remove_layer': 1.0,
            'dsge_layer': 0.0,
            'macro_layer': 0.0,
            'train_longer': 0.0
        }
        ind, grammar = self.create_individual()
        new_ind = operators.mutation(ind, grammar, mutation_config, 60)

        self.assertEqual(self.count_layers(ind.modules)-1, self.count_layers(new_ind.modules), "Error: remove layer wrong size")


    def test_mutate_ge(self):
        random.seed(0)
        mutation_config = {
            'add_connection': 0.0,
            'remove_connection': 0.0,
            'add_layer': 0.0,
            'reuse_layer': 0.0,
            'remove_layer': 0.0,
            'dsge_layer': 1.0,
            'macro_layer': 0.0,
            'train_longer': 0.0
        }
        ind, grammar = self.create_individual()
        new_ind = operators.mutation(ind, grammar, mutation_config, 60)

        self.assertEqual(self.count_layers(ind.modules), self.count_layers(new_ind.modules), "Error: change ge parameter")

        count_ref = list()
        count_differences = 0
        total_dif = 0
        for module_idx in range(len(ind.modules)):
            for layer_idx in range(len(ind.modules[module_idx].layers)):
                total_dif += 1
                if ind.modules[module_idx].layers[layer_idx] != new_ind.modules[module_idx].layers[layer_idx]:
                    if id(ind.modules[module_idx].layers[layer_idx]) not in count_ref:
                        count_ref.append(id(ind.modules[module_idx].layers[layer_idx]))
                        count_differences += 1

        self.assertEqual(total_dif, count_differences, "Error: change ge parameter")


if __name__ == '__main__':
    unittest.main()
