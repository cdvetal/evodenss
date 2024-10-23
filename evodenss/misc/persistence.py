from __future__ import annotations

import csv
import glob
import os
import shutil
from typing import TYPE_CHECKING, Any, Callable, Optional

import dill

from evodenss.config.pydantic import Config, get_config
from evodenss.evolution.individual import Individual
from evodenss.metrics.evaluation_metrics import EvaluationMetrics
from evodenss.misc.constants import MODEL_FILENAME, OVERALL_BEST_FOLDER, STATS_FOLDER_NAME

if TYPE_CHECKING:
    from torch.utils.data import Subset

    from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetType
    from evodenss.evolution.grammar import Grammar
    from evodenss.misc.checkpoint import Checkpoint


__all__ = ['RestoreCheckpoint', 'SaveCheckpoint', 'save_overall_best_individual',
           'build_individual_path', 'build_overall_best_path']

class RestoreCheckpoint:

    def __init__(self,
                 f: Callable[[int, str, Grammar, Config, bool, Optional[Checkpoint]], Checkpoint]) -> None:
        self.f: Callable[[int, str, Grammar, Config, bool, Optional[Checkpoint]], Checkpoint] = f

    def __call__(self,
                 run: int,
                 dataset_name: str,
                 grammar: Grammar,
                 config: Config,
                 is_gpu_run: bool) -> None:
        self.f(run,
               dataset_name,
               grammar,
               config,
               is_gpu_run,
               self.restore_checkpoint(config.checkpoints_path, run))

    def restore_checkpoint(self, save_path: str, run: int) -> Optional[Checkpoint]:
        if os.path.exists(os.path.join(save_path, f"run_{run}", "checkpoint.pkl")):
            with open(os.path.join(save_path, f"run_{run}", "checkpoint.pkl"), "rb") as handle_checkpoint:
                checkpoint: Checkpoint = dill.load(handle_checkpoint)
            return checkpoint
        else:
            return None



class SaveCheckpoint:

    def __init__(self,
                 f: Callable[[int, int, dict[DatasetType, Subset[ConcreteDataset]], Grammar, Checkpoint],
                             Checkpoint]) -> None:
        self.f: Callable[[int, int, dict[DatasetType, Subset[ConcreteDataset]], Grammar, Checkpoint],
                         Checkpoint] = f

    def __call__(self, *args: Any, **kwargs: Any) -> Checkpoint:
        new_checkpoint: Checkpoint = self.f(*args)
        # we assume the config is the last parameter in the function decorated
        self._save_checkpoint(new_checkpoint,
                              get_config().checkpoints_path,
                              get_config().evolutionary.generations)
        return new_checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint, save_path: str, max_generations: int) -> None:
        assert checkpoint.population is not None
        assert checkpoint.parent is not None
        with open(os.path.join(save_path, f"run_{checkpoint.run}", "checkpoint.pkl"), "wb") as handle_checkpoint:
            dill.dump(checkpoint, handle_checkpoint)
        self._delete_unnecessary_files(checkpoint, save_path, max_generations)
        self._save_statistics(save_path, checkpoint)

    def _delete_unnecessary_files(self, checkpoint: Checkpoint, save_path: str, max_generations: int) -> None:
        assert checkpoint.population is not None
        # remove temporary files to free disk space
        files_to_delete = glob.glob(
            f"{save_path}/"
            f"run_{checkpoint.run}/"
            f"ind=*_generation={checkpoint.last_processed_generation}/*{MODEL_FILENAME}")
        for file in files_to_delete:
            os.remove(file)
        gen: int = checkpoint.last_processed_generation - 2
        if checkpoint.last_processed_generation > 1:
            folders_to_delete = glob.glob(f"{save_path}/run_{checkpoint.run}/ind=*_generation={gen}")
            for folder in folders_to_delete:
                shutil.rmtree(folder)
        #if checkpoint.last_processed_generation == max_generations-1:
        #    folders_to_delete = glob.glob(f"{save_path}/run_{checkpoint.run}/ind=*_generation=*")
        #    for folder in folders_to_delete:
        #        shutil.rmtree(folder)


    def _save_statistics(self, save_path: str, checkpoint: Checkpoint) -> None:
        assert checkpoint.population is not None
        with open(os.path.join(save_path,
                               f"run_{checkpoint.run}",
                               STATS_FOLDER_NAME,
                               f"generation_{checkpoint.last_processed_generation}.csv"), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(["id", "phenotype", "num_epochs", "total_training_time_allocated"] + \
                                EvaluationMetrics.list_fields())
            for ind in checkpoint.population:
                csvwriter.writerow([ind.id,
                                    ind.phenotype,
                                    ind.num_epochs,
                                    ind.total_allocated_train_time,
                                    *ind.metrics]) # type: ignore
        file_exists: bool = os.path.isfile(os.path.join(save_path,
                                                        f"run_{checkpoint.run}",
                                                        STATS_FOLDER_NAME,
                                                        "test_accuracies.csv"))
        with open(os.path.join(save_path,
                               f"run_{checkpoint.run}",
                               STATS_FOLDER_NAME,
                               "test_accuracies.csv"), 'a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if file_exists is False:
                csvwriter.writerow(["generation", "test_accuracy"])
            csvwriter.writerow([checkpoint.last_processed_generation, checkpoint.best_gen_ind_test_accuracy])

def save_overall_best_individual(best_individual_path: str, parent: Individual) -> None:
    shutil.copytree(best_individual_path,
                    os.path.join(best_individual_path, "..", OVERALL_BEST_FOLDER),
                    dirs_exist_ok=True)
    with open(os.path.join(best_individual_path, "..", OVERALL_BEST_FOLDER, "parent.pkl"), "wb") as handle:
        dill.dump(parent, handle)


def build_individual_path(checkpoint_base_path: str,
                          run: int,
                          generation: int,
                          individual_id: int) -> str:
    return os.path.join(f"{checkpoint_base_path}",
                        f"run_{run}",
                        f"ind={individual_id}_generation={generation}")


def build_overall_best_path(checkpoint_base_path: str, run: int) -> str:
    return os.path.join(f"{checkpoint_base_path}", f"run_{run}", OVERALL_BEST_FOLDER)
