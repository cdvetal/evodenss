from __future__ import annotations

import logging
from typing import Callable, cast, Optional, TYPE_CHECKING

import numpy as np

from evodenss.metrics.evaluation_metrics import EvaluationMetrics
from evodenss.misc.utils import InputLayerId, LayerId
from evodenss.evolution.genotype import IndividualGenotype


if TYPE_CHECKING:
    from evodenss.config.pydantic import ArchitectureConfig
    from evodenss.evolution.grammar import Grammar
    from evodenss.metrics.fitness_metrics import Fitness
    from evodenss.dataset.dataset_loader import ConcreteDataset, DatasetType
    from evodenss.networks.evaluators import BaseEvaluator
    from torch.utils.data import Subset

__all__ = ['Individual']


logger = logging.getLogger(__name__)


class Individual:

    def __init__(self,
                 grammar: Grammar,
                 network_architecture_config: ArchitectureConfig,
                 ind_id: int,
                 track_mutations: bool) -> None:

        self.id: int = ind_id
        self.individual_genotype: IndividualGenotype = \
            IndividualGenotype(grammar, network_architecture_config, track_mutations)
        self.phenotype: Optional[str] = None
        self.phenotype_projector: Optional[str] = None
        self.fitness: Optional[Fitness] = None
        self.metrics: Optional[EvaluationMetrics] = None
        self.num_epochs: int = 0
        self.current_time: float = 0.0
        self.total_allocated_train_time: float = 0.0
        self.total_training_time_spent: float = 0.0


    def __eq__(self, other: object) -> bool:
        if isinstance(other, Individual):
            return self.__dict__ == other.__dict__
        return False


    def _decode(self, grammar: Grammar, static_projector_config: Optional[list[int]]) -> str:
        phenotype: str = ''
        static_projector_phenotype: str = ''
        module_offset: int = 0

        for module_idx, module in enumerate(self.individual_genotype.modules_dict.values()):
            for layer_idx, layer_genotype in enumerate(module.layers):
                phenotype_layer: str = f" {grammar.decode(module.module_name, layer_genotype)}"
                connections_array: list[InputLayerId]
                phenotype_connections: str
                if (module_idx == 0 and layer_idx == 0) or \
                        (module.module_name == "projector" and layer_idx == 0):
                    module_offset = 0
                    connections_array = cast(list[InputLayerId], [-1])
                    phenotype_connections = f"{','.join(map(str, connections_array))}"
                else:
                    connections_array = module.connections[LayerId(layer_idx)]
                    phenotype_connections = \
                        f"{','.join(map(str, np.array(connections_array) + module_offset))}"
                phenotype += f"{phenotype_layer} input:{phenotype_connections}"
            module_offset += module.count_layers()

        # Build phenotype for projector network if this is static
        # (if it is dynamic, then it was already decoded above)
        if static_projector_config is not None:
            dense: Callable[..., str] = \
                lambda out, act, i: f"projector_layer:fc act:{act} out_features:{out} bias:True input:{i}" # noqa: E731
            batch: Callable[..., str] = \
                lambda act, i: f"projector_layer:batch_norm_proj act:{act} input:{i}" # noqa: E731
            for i in range(len(static_projector_config)*2):
                activation: str = "linear" if i >= len(static_projector_config)*2-2 else "relu"
                if i % 2 == 0:
                    static_projector_phenotype += " " + dense(static_projector_config[i//2], "linear", i-1)
                else:
                    static_projector_phenotype += " " + batch(activation, i-1)
        phenotype += static_projector_phenotype

        # Build phenotype for output layer
        final_input_layer_id: int
        if "projector" in self.individual_genotype.modules_dict.keys():
            assert static_projector_config is None
            final_input_layer_id = self.individual_genotype.modules_dict['projector'].count_layers() - 1
        else:
            if static_projector_config is None:
                # in this case we connect the output layer to the encoder instead of the projector
                # this because the only valid way we enter this if statement is because
                # we are doing evolution using supervised learning
                final_input_layer_id = module_offset - 1
            else:
                final_input_layer_id = len(static_projector_config) * 2 - 1
        
        final_phenotype_layer: str = grammar.decode(self.individual_genotype.output_layer_start_symbol_name,
                                                    self.individual_genotype.output_layer)
        phenotype += " " + final_phenotype_layer + " input:" + str(final_input_layer_id)

        # Build phenotype for non-topological stuff (learning, pretext task...)
        for start_symbol_name, extra_genotype in zip(self.individual_genotype.extra_genotype_start_symbol_names,
                                                     self.individual_genotype.extra_genotype):
            phenotype += " " + grammar.decode(start_symbol_name, extra_genotype)

        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype


    def reset_keys(self, *keys: str) -> None:
        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, (int, float)):
                setattr(self, key, 0)
            else:
                setattr(self, key, None)


    def evaluate(self,
                 grammar: Grammar,
                 dataset: dict['DatasetType', 'Subset[ConcreteDataset]'],
                 cnn_eval: BaseEvaluator,
                 static_projector_config: Optional[list[int]],
                 model_saving_dir: str,
                 parent_dir: Optional[str]=None) -> Fitness: #pragma: no cover

        phenotype: str
        phenotype = self._decode(grammar, static_projector_config)

        reuse_parent_weights: bool
        reuse_parent_weights = True
        if self.current_time == 0:
            reuse_parent_weights = False

        allocated_train_time: float = self.total_allocated_train_time - self.current_time
        logger.info(f"-----> Starting evaluation for individual {self.id} for {allocated_train_time} secs")
        evaluation_metrics: EvaluationMetrics = cnn_eval.evaluate(phenotype,
                                                                  dataset,
                                                                  model_saving_dir,
                                                                  parent_dir,
                                                                  reuse_parent_weights,
                                                                  allocated_train_time,
                                                                  self.num_epochs)
        if self.metrics is None:
            self.metrics = evaluation_metrics
        else:
            self.metrics += evaluation_metrics
        self.fitness = self.metrics.fitness
        self.num_epochs += self.metrics.n_epochs
        self.current_time += allocated_train_time
        self.total_training_time_spent += self.metrics.training_time_spent
        logger.info(f"Evaluation results for individual {self.id}: {self.metrics}\n")

        assert self.fitness is not None
        return self.fitness
