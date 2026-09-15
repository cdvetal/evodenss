import os
import random
import shutil
import unittest

import dill

from evodenss.config.pydantic import ArchitectureConfig, ModuleConfig, NetworkStructure, init_context
from evodenss.evolution.grammar import Grammar
from evodenss.evolution.individual import Individual
from evodenss.misc.checkpoint import Checkpoint
from evodenss.misc.constants import OVERALL_BEST_FOLDER, STATS_FOLDER_NAME
from evodenss.misc.enums import LearningType
from evodenss.misc.persistence import RestoreCheckpoint, SaveCheckpoint, build_individual_path, build_overall_best_path


class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.save_checkpoint: SaveCheckpoint = SaveCheckpoint(None) # type: ignore
        self.restore_checkpoint: RestoreCheckpoint = RestoreCheckpoint(None) # type: ignore
        with init_context({'learning_type': LearningType.supervised}):
            self.network_config: ArchitectureConfig = ArchitectureConfig(
                reuse_layer=0.2,
                extra_components=['learning'],
                output="identity",
                projector=None,
                modules=[ModuleConfig(name="features",
                                      network_structure_init=NetworkStructure(min_expansions=3, max_expansions=6),
                                      network_structure=NetworkStructure(min_expansions=3, max_expansions=30),
                                      levels_back=1)]
            )
        grammar_path: str = "tests/resources/example_full.grammar"
        self.grammar: Grammar = Grammar(grammar_path)

    def test_save_checkpoint(self) -> None:
        folder_name: str = "test_dir"
        run: int = 0
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(os.path.join(folder_name, f"run_{run}"), exist_ok=True)
        os.makedirs(os.path.join(folder_name, f"run_{run}", STATS_FOLDER_NAME), exist_ok=True)
        random.seed(0)
        fake_parent: Individual = Individual(
            grammar=self.grammar,
            network_architecture_config=self.network_config,
            ind_id=0,
            track_mutations=True
        )
        test_checkpoint: Checkpoint = \
            Checkpoint(run, None, None, None, 0, 0, None, None, 0, [], fake_parent) # type: ignore
        self.save_checkpoint._save_checkpoint(test_checkpoint, save_path=folder_name, max_generations=10)

        expected_path: str = os.path.join(folder_name, f"run_{run}", "checkpoint.pkl")
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path, 'rb') as handle_checkpoint:
            loaded_checkpoint: Checkpoint = dill.load(handle_checkpoint)
        self.assertEqual(test_checkpoint, loaded_checkpoint)
        
        shutil.rmtree(folder_name)

    def test_restore_checkpoint(self) -> None:
        random.seed(0)
        fake_parent: Individual = Individual(
            grammar=self.grammar,
            network_architecture_config=self.network_config,
            ind_id=0,
            track_mutations=True
        )
        run: int = 0
        expected_checkpoint: Checkpoint = \
            Checkpoint(run, None, None, None, 0, 0, None, None, 0, [], fake_parent) # type: ignore
        self.assertEqual(expected_checkpoint, self.restore_checkpoint.restore_checkpoint("tests/resources", run))

    def test_build_individual_path(self) -> None:
        folder_name: str = "test_dir"
        run: int = 0
        generation: int = 0
        ind_id: int = 0
        path: str = build_individual_path(folder_name, run, generation, ind_id)
        self.assertEqual(path, os.path.join(folder_name,
                                            f"run_{run}",
                                            f"ind={ind_id}_generation={generation}"))

    def test_build_overall_best_path(self) -> None:
        folder_name: str = "test_dir"
        run: int = 0
        path: str = build_overall_best_path(folder_name, run)
        self.assertEqual(path, os.path.join(folder_name, f"run_{run}", OVERALL_BEST_FOLDER))

if __name__ == '__main__':
    unittest.main()
