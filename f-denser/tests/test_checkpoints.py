# type: ignore
import pickle
import os
import shutil
import unittest

from fast_denser.evolution import Individual
from fast_denser.misc import Checkpoint
from fast_denser.misc.constants import OVERALL_BEST_FOLDER, STATS_FOLDER_NAME
from fast_denser.misc.persistence import *

class Test(unittest.TestCase):
	
    def setUp(self):
        self.save_checkpoint: SaveCheckpoint = SaveCheckpoint(None)
        self.restore_checkpoint: RestoreCheckpoint = RestoreCheckpoint(None)

    def test_save_checkpoint(self):
        folder_name = "test_dir"
        run = 0
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(os.path.join(folder_name, "run_0"), exist_ok=True)
        os.makedirs(os.path.join(folder_name, "run_0", STATS_FOLDER_NAME), exist_ok=True)
        fake_parent = Individual({'network_structure': [["features", 1, 3]],
                                  'output': "softmax",
                                  'macro_structure': ["learning"]},
                                  0,
                                  0)
        test_checkpoint: Checkpoint = Checkpoint(run, None, None, None, 0, 0, 0, None, [], fake_parent)
        
        self.save_checkpoint._save_checkpoint(test_checkpoint, save_path=folder_name, max_generations=10)

        expected_path: str = os.path.join(folder_name, f"run_{run}", "checkpoint.pkl")
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path, 'rb') as handle_checkpoint:
            loaded_checkpoint: Checkpoint = pickle.load(handle_checkpoint)
        self.assertEqual(test_checkpoint, loaded_checkpoint)
        shutil.rmtree(folder_name)

	
    def test_restore_checkpoint(self):
        fake_parent = Individual({'network_structure': [["features", 1, 3]],
                                  'output': "softmax",
                                  'macro_structure': ["learning"]},
                                  0,
                                  0)
        run=0
        
        expected_checkpoint: Checkpoint = Checkpoint(run, None, None, None, 0, 0, 0, None, [], fake_parent)
        self.assertEqual(expected_checkpoint, self.restore_checkpoint.restore_checkpoint("tests/utils", run))

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
