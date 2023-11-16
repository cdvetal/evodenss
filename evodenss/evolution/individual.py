from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from evodenss.networks import Module
from evodenss.misc.evaluation_metrics import EvaluationMetrics
from evodenss.misc.fitness_metrics import Fitness


if TYPE_CHECKING:
    from evodenss.evolution.grammar import Genotype, Grammar
    from evodenss.networks.torch.evaluators import BaseEvaluator
    from evodenss.networks.module import ModuleConfig

__all__ = ['Individual']


logger = logging.getLogger(__name__)


class Individual:
    """
        Candidate solution.


        Attributes
        ----------
        network_structure : list
            ordered list of tuples formated as follows
            [(non-terminal, min_expansions, max_expansions), ...]

        output_rule : str
            output non-terminal symbol

        macro_rules : list
            list of non-terminals (str) with the marco rules (e.g., learning)

        modules : list
            list of Modules (genotype) of the layers

        output : dict
            output rule genotype

        macro : list
            list of Modules (genotype) for the macro rules

        phenotype : str
            phenotype of the candidate solution

        fitness : float
            fitness value of the candidate solution

        metrics : dict
            training metrics

        num_epochs : int
            number of performed epochs during training

        trainable_parameters : int
            number of trainable parameters of the network

        time : float
            network training time

        current_time : float
            performed network training time

        train_time : float
            maximum training time

        id : int
            individual unique identifier


        Methods
        -------
            initialise(grammar, levels_back, reuse)
                Randomly creates a candidate solution

            decode(grammar)
                Maps the genotype to the phenotype

            evaluate(grammar, cnn_eval, weights_save_path, parent_weights_path='')
                Performs the evaluation of a candidate solution
    """


    def __init__(self, network_architecture_config: Dict[str, Any], ind_id: int, seed: int) -> None:

        self.seed: int = seed
        self.modules_configurations: Dict[str, ModuleConfig] = network_architecture_config['modules']
        self.output_rule: str = network_architecture_config['output']
        self.macro_rules: List[str] = network_architecture_config['macro_structure']
        self.modules: List[Module] = []
        self.output: Optional[Genotype] = None
        self.macro: List[Genotype] = []
        self.phenotype: Optional[str] = None
        self.phenotype_projector: Optional[str] = None
        self.fitness: Optional[Fitness] = None
        self.metrics: Optional[EvaluationMetrics] = None
        self.num_epochs: int = 0
        self.current_time: float = 0.0
        self.total_allocated_train_time: float = 0.0
        self.total_training_time_spent: float = 0.0
        self.id: int = ind_id


    def __eq__(self, other: object) -> bool:
        if isinstance(other, Individual):
            return self.__dict__ == other.__dict__
        return False


    def initialise(self, grammar: Grammar, reuse: float) -> "Individual":
        """
            Randomly creates a candidate solution

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            levels_back : dict
                number of previous layers a given layer can receive as input

            reuse : float
                likelihood of reusing an existing layer

            Returns
            -------
            candidate_solution : Individual
                randomly created candidate solution
        """
        for module_name, module_config in self.modules_configurations.items():
            new_module: Module = Module(module_name, module_config)
            new_module.initialise(grammar, reuse)
            self.modules.append(new_module)

        # Initialise output
        self.output = grammar.initialise(self.output_rule)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

        return self


    def _decode(self, grammar: Grammar, static_projector_config: Optional[List[int]]) -> str:
        """
            Maps the genotype to the phenotype

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            Returns
            -------
            phenotype : str
                phenotype of the individual to be used in the mapping to the keras model.
        """

        phenotype: str = ''
        offset: int = 0
        layer_counter: int = 0
        projector_phenotype: str = ""
        projection_layer_count: int = 0
        for i, module in enumerate(self.modules):
            offset = layer_counter
            for layer_idx, layer_genotype in enumerate(module.layers):
                layer_counter += 1
                phenotype_layer: str = f" {grammar.decode(module.module_name, layer_genotype)}"
                current_connections = deepcopy(module.connections[layer_idx])
                # ADRIANO HACK
                if "relu_agg" in phenotype_layer and -1 not in module.connections[layer_idx]:
                    current_connections = [-1] + current_connections
                # END
                final_offset: int = 0 if "projector_layer" in phenotype_layer else offset
                phenotype += (
                    f"{phenotype_layer}"
                    f" input:{','.join(map(str, np.array(current_connections) + final_offset))}"
                )
        # If a projector needs to be generated,
        # final layer id needs to be generated based on the projector module
        final_input_layer_id: int
        assert self.output is not None
        final_phenotype_layer: str = grammar.decode(self.output_rule, self.output)

        if "projector" in final_phenotype_layer and len(self.modules) == 1:
            offset = layer_counter

        if final_offset == 0: # projector to be generated
            final_input_layer_id = layer_counter - 1 - offset
        else:
            final_input_layer_id = layer_counter - 1

        if static_projector_config is not None:
            for i in static_projector_config[:-1]:
                projector_phenotype += f" projector_layer:fc act:linear out_features:{i} bias:True" + \
                    f" input:{projection_layer_count-1}"
                projector_phenotype += " projector_layer:batch_norm_proj act:relu" + \
                    f" input:{projection_layer_count}"
                projection_layer_count += 2
            projector_phenotype += f" projector_layer:fc act:linear out_features:{static_projector_config[-1]}" + \
                f" bias:True input:{projection_layer_count-1}"
            projector_phenotype += " projector_layer:batch_norm_proj act:linear" + \
                f" input:{projection_layer_count}"
            projection_layer_count += 2

        phenotype += projector_phenotype + \
            " " + final_phenotype_layer + " input:" + str(projection_layer_count + final_input_layer_id)

        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += " " + grammar.decode(macro_rule, self.macro[rule_idx])

        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype


    def evaluate(self,
                 grammar: Grammar,
                 cnn_eval: BaseEvaluator,
                 static_projector_config: Optional[List[int]],
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
