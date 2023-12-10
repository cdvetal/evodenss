# type: ignore
import os
import shutil
from typing import Dict
import unittest

import dill

from evodenss.evolution import Individual
from evodenss.misc import Checkpoint
from evodenss.misc.constants import OVERALL_BEST_FOLDER, STATS_FOLDER_NAME
from evodenss.misc.persistence import * # pylint: disable=unused-wildcard-import,wildcard-import
from evodenss.networks import ModuleConfig

class Test(unittest.TestCase):

    def setUp(self):
        self.save_checkpoint: SaveCheckpoint = SaveCheckpoint(None)
        self.restore_checkpoint: RestoreCheckpoint = RestoreCheckpoint(None)
        self.network_config: Dict[str, Any] = {
            'reuse_layer': 0.2,
            'macro_structure': {},
            'output': 'identity',
            'modules': {
                'features': ModuleConfig(min_expansions=3,
                                         max_expansions=30,
                                         initial_network_structure=[3,4,5,6],
                                         levels_back=1)
            }
        }

    def test_save_checkpoint(self):
        folder_name: str = "test_dir"
        run: int = 0
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(os.path.join(folder_name, f"run_{run}"), exist_ok=True)
        os.makedirs(os.path.join(folder_name, f"run_{run}", STATS_FOLDER_NAME), exist_ok=True)
        fake_parent: Individual = Individual(
            network_architecture_config=self.network_config,
            ind_id=0,
            seed=run
        )
        test_checkpoint: Checkpoint = Checkpoint(run, None, None, None, 0, 0, None, None, 0, [], fake_parent)
        self.save_checkpoint._save_checkpoint(test_checkpoint, save_path=folder_name, max_generations=10)

        expected_path: str = os.path.join(folder_name, f"run_{run}", "checkpoint.pkl")
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path, 'rb') as handle_checkpoint:
            loaded_checkpoint: Checkpoint = dill.load(handle_checkpoint)
        self.assertEqual(test_checkpoint, loaded_checkpoint)
        shutil.rmtree(folder_name)


    def test_restore_checkpoint(self):
        fake_parent: Individual = Individual(
            network_architecture_config=self.network_config,
            ind_id=0,
            seed=0
        )
        run: int = 0
        expected_checkpoint: Checkpoint = Checkpoint(run, None, None, None, 0, 0, None, None, 0, [], fake_parent)
        self.assertEqual(expected_checkpoint, self.restore_checkpoint.restore_checkpoint("tests/resources", run))


    def test_build_individual_path(self):
        folder_name: str = "test_dir"
        run: int = 0
        generation: int = 0
        ind_id: int = 0
        path: str = build_individual_path(folder_name, run, generation, ind_id)
        self.assertEqual(path, os.path.join(folder_name,
                                            f"run_{run}",
                                            f"ind={ind_id}_generation={generation}"))


    def test_build_overall_best_path(self):
        folder_name: str = "test_dir"
        run: int = 0
        path: str = build_overall_best_path(folder_name, run)
        self.assertEqual(path, os.path.join(folder_name, f"run_{run}", OVERALL_BEST_FOLDER))

if __name__ == '__main__':
    unittest.main()
