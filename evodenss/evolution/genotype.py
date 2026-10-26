from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeVar

from evodenss.config.pydantic import ArchitectureConfig
from evodenss.networks.module import Module

if TYPE_CHECKING:
    from evodenss.evolution.grammar import Derivation, Grammar, NonTerminal, Symbol
    from evodenss.evolution.operators.mutation_tracker import MutationLog

T = TypeVar('T')
K = TypeVar('K')


@dataclass
class Genotype:
    expansions: dict[NonTerminal, list[Derivation]]
    codons: dict[Symbol, list[int]]

    @classmethod
    def empty(cls) -> 'Genotype':
        return cls({}, {})

    def _concatenate_to_dict(self,
                             dict: dict[K, list[T]],
                             key: K,
                             element: T,
                             mode: str='append') -> dict[K, list[T]]:
        if key not in dict.keys():
            dict[key] = [element]
        else:
            if mode == 'append':
                dict[key] = dict[key] + [element]
            elif mode == 'prepend':
                dict[key] = [element] + dict[key]
            else:
                raise ValueError(f"Unrecognised value: [{mode}]. Only 'append' and 'prepend are accepted")
        return dict

    def add_to_genome(self, non_terminal: NonTerminal, codon: int, derivation: Derivation, mode: str) -> None:
        self.codons = self._concatenate_to_dict(self.codons, non_terminal, codon, mode)
        self.expansions = self._concatenate_to_dict(self.expansions, non_terminal, derivation, mode)

    def __iadd__(self, other: 'Genotype') -> 'Genotype':
        for k in other.expansions.keys():
            if k not in self.expansions.keys():
                self.expansions[k] = other.expansions[k]
            else:
                self.expansions[k] += other.expansions[k]
        for i in other.codons.keys():
            if i not in self.codons.keys():
                self.codons[i] = other.codons[i]
            else:
                self.codons[i] += other.codons[i]
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Genotype):
            return self.__dict__ == other.__dict__
        return False


class IndividualGenotype:

    def __init__(self,
                 grammar: Grammar,
                 network_architecture_config: ArchitectureConfig,
                 track_mutations: bool=False) -> None:
        # Modules' genotypes
        self.modules_dict: OrderedDict[str, Module] = OrderedDict({
            module_config.name: Module(module_config, grammar, network_architecture_config.reuse_layer)
            for module_config in network_architecture_config.modules
        })
        # Final layer
        self.output_layer_start_symbol_name: str = network_architecture_config.output
        self.output_layer: Genotype = grammar.initialise(self.output_layer_start_symbol_name)
        # Extra info to allocate other non-topological components
        # that can be evolved (e.g. data augmentation, learning)
        self.extra_genotype_start_symbol_names: list[str] = network_architecture_config.extra_components
        self.extra_genotype: list[Genotype] = [
            grammar.initialise(s_name) for s_name in self.extra_genotype_start_symbol_names
        ]
        self.mutation_tracker: Optional[list[MutationLog]] = [
        ] if track_mutations else None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IndividualGenotype):
            return self.__dict__ == other.__dict__
        return False
