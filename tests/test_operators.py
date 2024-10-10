from copy import deepcopy
import random
import unittest
import warnings

from evodenss.config.pydantic import ArchitectureConfig, ModuleConfig, MutationConfig, NetworkStructure
from evodenss.evolution.genotype import Genotype
from evodenss.evolution.grammar import Derivation
from evodenss.evolution.operators import mutation
from evodenss.evolution.individual import Individual
from evodenss.evolution.grammar import Grammar, NonTerminal
from evodenss.misc.utils import LayerId
from evodenss.networks.module import Module
from tests.resources.genotype_examples import mutation_added_layer_genotype, simple_sample1

class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.mutation_config = MutationConfig(
            add_connection=0.0,
            remove_connection=0.0,
            add_layer=0.0,
            reuse_layer=0.0,
            remove_layer=0.0,
            dsge_topological=0.0,
            dsge_non_topological=0.0,
            train_longer=0.0
        )
        warnings.simplefilter('ignore', category=DeprecationWarning)


    def assert_layers_mutation(self,
                               obtained_layers: list[Genotype],
                               original_layers: list[Genotype],
                               added_layers_idx: list[int],
                               added_layers: list[Genotype],
                               removed_layers_idx: list[int]) -> None:
        expected_layers: list[Genotype] = deepcopy(original_layers)
        for idx, l in zip(added_layers_idx, added_layers):
            expected_layers = expected_layers[:idx] + [l] + expected_layers[idx:]
        for idx in removed_layers_idx:
            expected_layers = expected_layers[:idx] + expected_layers[idx+1:]
        self.assertEqual(obtained_layers, expected_layers)


    def count_unique_layers(self, modules: dict[str, Module]) -> int:
        unique_layers = []
        for module in modules.values():
            for layer in module.layers:
                unique_layers.append(id(layer))

        return len(set(unique_layers))


    def count_layers(self, modules: dict[str, Module]) -> int:
        return sum([len(module.layers) for module in modules.values()])


    def create_individual(self) -> tuple[Individual, Grammar]:
        seed: int = 0
        grammar = Grammar('tests/resources/example_full.grammar')
        network_config = ArchitectureConfig(
            reuse_layer=0.0,
            extra_components=['learning'],
            output="softmax",
            projector=None,
            modules=[ModuleConfig(name="features",
                                  network_structure_init=NetworkStructure(min_expansions=2, max_expansions=2),
                                  network_structure=NetworkStructure(min_expansions=1, max_expansions=3),
                                  levels_back=1)]
        )
        ind: Individual = Individual(
            grammar=grammar,
            network_architecture_config=network_config,
            ind_id=seed,
            track_mutations=True
        )
        return ind, grammar


    def test_add_layer_random(self):
        random.seed(0)
        self.mutation_config.add_layer = 1.0
        ind, grammar = self.create_individual()
        new_ind = mutation.mutate(deepcopy(ind), grammar, 0, self.mutation_config, 60)
        self.mutation_config.add_layer = 0.0
        obtained_layers = new_ind.individual_genotype.modules_dict['features'].layers
        original_layers = ind.individual_genotype.modules_dict['features'].layers
        connections = ind.individual_genotype.modules_dict['features'].connections
        obtained_connections = new_ind.individual_genotype.modules_dict['features'].connections
        self.assert_layers_mutation(obtained_layers,
                                    original_layers,
                                    [2],
                                    [mutation_added_layer_genotype],
                                    [])
        self.assertEqual(obtained_connections, {**connections, **{2: [1]}})
        self.assertEqual(self.count_layers(ind.individual_genotype.modules_dict)+1,
                         self.count_layers(new_ind.individual_genotype.modules_dict),
                         "Error: add layer wrong size")


    def test_add_layer_replicate(self):
        random.seed(0)
        self.mutation_config.add_layer = 1.0
        self.mutation_config.reuse_layer = 1.0

        ind, grammar = self.create_individual()
        new_ind = mutation.mutate(ind, grammar, 0, self.mutation_config, 60)

        self.mutation_config.add_layer = 0.0
        self.mutation_config.reuse_layer = 0.0
        obtained_layers = new_ind.individual_genotype.modules_dict['features'].layers
        original_layers = ind.individual_genotype.modules_dict['features'].layers
        connections = ind.individual_genotype.modules_dict['features'].connections
        obtained_connections = new_ind.individual_genotype.modules_dict['features'].connections
        self.assert_layers_mutation(obtained_layers,
                                    original_layers,
                                    [2],
                                    [original_layers[0]],
                                    [])
        self.assertEqual(obtained_connections, {**connections, **{2: [1]}})
        self.assertEqual(self.count_unique_layers(ind.individual_genotype.modules_dict),
                         self.count_unique_layers(new_ind.individual_genotype.modules_dict),
                         "Error: duplicate layer wrong size")
        self.assertEqual(self.count_layers(ind.individual_genotype.modules_dict)+1,
                         self.count_layers(new_ind.individual_genotype.modules_dict),
                         "Error: duplicate layer wrong size")


    def test_remove_layer(self):
        random.seed(0)
        self.mutation_config.remove_layer = 1.0
        ind, grammar = self.create_individual()
        connections = ind.individual_genotype.modules_dict['features'].connections
        new_ind = mutation.mutate(deepcopy(ind), grammar, 0, self.mutation_config, 60)
        self.mutation_config.remove_layer = 0.0
        obtained_layers = new_ind.individual_genotype.modules_dict['features'].layers
        original_layers = ind.individual_genotype.modules_dict['features'].layers

        obtained_connections = new_ind.individual_genotype.modules_dict['features'].connections
        connections.pop(LayerId(1))

        self.assert_layers_mutation(obtained_layers,
                                    original_layers,
                                    [],
                                    [],
                                    [0])
        self.assertEqual(obtained_connections, connections)
        self.assertEqual(self.count_layers(ind.individual_genotype.modules_dict)-1,
                         self.count_layers(new_ind.individual_genotype.modules_dict),
                         "Error: remove layer wrong size")


    def test_dsge_mutation(self):
        random.seed(0)
        grammar = Grammar("tests/resources/simple_grammar.grammar")
        sample_to_mutate = deepcopy(simple_sample1)
        expected_sample = deepcopy(simple_sample1)
        expected_sample.expansions[NonTerminal(name='value')][1] = Derivation([NonTerminal(name='var')])
        expected_sample.codons[NonTerminal(name='value')][1] = 1
        mutation._mutation_dsge(sample_to_mutate, grammar)

        self.assertEqual(sample_to_mutate, expected_sample)

if __name__ == '__main__':
    unittest.main()
