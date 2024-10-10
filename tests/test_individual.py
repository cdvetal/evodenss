import random
import unittest
from typing import Optional

from parameterized import parameterized

from evodenss.config.pydantic import ArchitectureConfig, ModuleConfig, NetworkStructure
from evodenss.evolution.grammar import Grammar
from evodenss.evolution.genotype import Genotype
from evodenss.evolution.individual import Individual

from tests.resources.genotype_examples import ind_test_output1, ind_test_output2, \
    ind_test_learning1, ind_test_learning2
from tests.resources.phenotype_examples import ind_phenotype1, ind_phenotype2, ind_phenotype3, ind_phenotype4


class Test(unittest.TestCase):

    def _set_network_config(self, output: str) -> ArchitectureConfig:
        return ArchitectureConfig(
            reuse_layer=0.2,
            extra_components=['learning'],
            output=output,
            projector=None,
            modules=[ModuleConfig(name="features",
                                network_structure_init=NetworkStructure(min_expansions=3, max_expansions=6),
                                network_structure=NetworkStructure(min_expansions=3, max_expansions=30),
                                levels_back=1)]
        )

    def setUp(self) -> None:
        grammar_path: str = "tests/resources/example_full.grammar"
        self.grammar: Grammar = Grammar(grammar_path)

    
    @parameterized.expand([
        (0, ind_test_output1, ind_test_learning1),
        (1, ind_test_output2, ind_test_learning2)
    ])
    def test_initialise(self, seed: int, expected_output: Genotype, expected_learning: Genotype) -> None:
        random.seed(seed)
        individual: Individual = Individual(
            grammar=self.grammar,
            network_architecture_config=self._set_network_config("softmax"),
            ind_id=seed,
            track_mutations=True
        )
        self.assertEqual(individual.individual_genotype.output_layer, expected_output)
        self.assertEqual(individual.individual_genotype.extra_genotype, [expected_learning])


    @parameterized.expand([
        (0, ind_phenotype1, None),
        (1, ind_phenotype2, None),
        (1, ind_phenotype3, [512, 32, 10]),
        (1, ind_phenotype4, [5])
    ])
    def test_decode(self, seed: int, expected_phenotype: str, static_projector_config: Optional[list[int]]) -> None:
        random.seed(seed)
        network_config: ArchitectureConfig
        if static_projector_config is None:
            network_config = self._set_network_config("softmax")
        else:
            network_config = self._set_network_config("identity")
        individual: Individual = Individual(
            grammar=self.grammar,
            network_architecture_config=network_config,
            ind_id=seed,
            track_mutations=True
        )
        phenotype: str = individual._decode(self.grammar, static_projector_config)
        self.assertEqual(phenotype, expected_phenotype)



if __name__ == '__main__':
    unittest.main()
