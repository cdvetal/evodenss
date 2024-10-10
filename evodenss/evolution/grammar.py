from copy import deepcopy
from dataclasses import dataclass, field
import filecmp
import os
from random import randint, uniform
import shutil
import sys
from typing import Callable, Generic, NewType, Protocol, Optional, TypeVar

from evodenss.evolution.genotype import Genotype
from evodenss.misc.enums import AttributeType


class Comparable(Protocol):
    def __le__(self: 'T', __other: 'T') -> bool: ...
    def __ge__(self: 'T', __other: 'T') -> bool: ...

T = TypeVar('T', bound=Comparable)
K = TypeVar('K')


class Attribute(Generic[T]):

    def __init__(self,
                 var_type: str,
                 num_values: int,
                 min_value: T,
                 max_value: T,
                 generator: Callable[[int,T,T], list[T]]) -> None:
        self.var_type: str = var_type
        self.num_values: int = num_values
        self.min_value: T = min_value
        self.max_value: T = max_value
        self.generator: Callable[[int,T,T], list[T]] = generator
        self.values: Optional[list[T]] = None

    def generate(self) -> None:
        self.values = self.generator(self.num_values, self.min_value, self.max_value)
        assert self.values is not None

    def override_value(self, values_to_override: list[T]) -> None:
        for v in values_to_override:
            assert v >= self.min_value, f"error overriding value for attribute. {v} < {self.min_value}"
            assert v <= self.max_value, f"error overriding value for attribute. {v} > {self.max_value}"

    def __repr__(self) -> str:
        return f"Attribute(num_values={self.num_values}," + \
            f" min_value={self.min_value}," + \
            f" max_value={self.max_value}," + \
            f" values={self.values})"

    def __str__(self) -> str:
        string: str = f"{self.var_type},{self.num_values},{self.min_value},{self.max_value}"
        if self.values is not None:
            string += f",{self.values}"
        return string

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Attribute):
            return self.var_type == other.var_type and \
                self.num_values == other.num_values and \
                self.min_value == other.min_value and \
                self.max_value == other.max_value and \
                self.values == other.values
        return False


@dataclass
class Symbol:
    name: str

    @staticmethod
    def create_symbol(symbol_str: str) -> 'Symbol':
        symbol_name: str
        attribute_type: str
        num_values: str
        min_value: str
        max_value: str
        if "<" in symbol_str and ">" in symbol_str:
            symbol_name = symbol_str.replace('<', '').replace('>', '').rstrip().lstrip()
            return NonTerminal(symbol_name)
        elif "[" in symbol_str and "]" in symbol_str:
            attribute: Attribute[int]|Attribute[float]
            symbol_name, attribute_type, num_values, min_value, max_value = \
                symbol_str.replace('[', '')\
                          .replace(']', '')\
                          .split(',')
            if AttributeType(attribute_type) == AttributeType.INT:
                attribute = Attribute[int](attribute_type, int(num_values), int(min_value), int(max_value),
                                           lambda n, min, max: [randint(min, max) for _ in range(n)])
            elif AttributeType(attribute_type) == AttributeType.FLOAT:
                attribute = Attribute[float](attribute_type, int(num_values), float(min_value), float(max_value),
                                             lambda n, min, max: [uniform(min, max) for _ in range(n)])
            elif AttributeType(attribute_type) == AttributeType.INT_POWER2:
                attribute = Attribute[int](attribute_type, int(num_values), int(min_value), int(max_value),
                                           lambda n, min, max: [2 ** randint(min, max) for _ in range(n)])
            elif AttributeType(attribute_type) == AttributeType.INT_POWER2_INV:
                attribute = Attribute[int](attribute_type, int(num_values), int(min_value), int(max_value),
                                           lambda n, min, max: [1/(2 ** randint(min, max)) for _ in range(n)])
            else:
                raise AttributeError(f"Invalid Attribute type: [{attribute_type}]")
            return Terminal(symbol_name, attribute)
        else:
            return Terminal(symbol_str)

    def __lt__(self, other: 'Symbol') -> bool:
        return self.name < other.name

    def __le__(self, other: 'Symbol') -> bool:
        return self.name <= other.name

    def __gt__(self, other: 'Symbol') -> bool:
        return self.name > other.name

    def __ge__(self, other: 'Symbol') -> bool:
        return self.name >= other.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Symbol):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self) -> int:
        return hash(repr(self))

Derivation = NewType('Derivation', list[Symbol])


class NonTerminal(Symbol):
    def __str__(self) -> str:
        return f"<{self.name}>"


@dataclass
class Terminal(Symbol):
    attribute: Optional[Attribute[int]|Attribute[float]] = field(default=None)

    def __hash__(self) -> int:
        return hash(repr(self))

    def __str__(self) -> str:
        if self.attribute is None:
            return self.name
        else:
            return f"[{self.name},{str(self.attribute)}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Terminal):
            return self.__dict__ == other.__dict__
        return False


class Grammar:

    def __init__(self, path: str, backup_path: Optional[str]=None):
        self.grammar = self.get_grammar(path)
        if backup_path is not None:
            self._backup_used_grammar(path, backup_path)


    def _backup_used_grammar(self, origin_filepath: str, destination: str) -> None:
        destination_filepath: str =  os.path.join(destination, "used_grammar.yaml")
        # if there is a config file backed up already and it is different than the one we are trying to backup
        if os.path.isfile(destination_filepath) and \
            filecmp.cmp(origin_filepath, destination_filepath) is False:
            raise ValueError("You are probably trying to continue an experiment "
                             "with a different grammar than the one you used initially. "
                             "This is a gentle reminder to double-check the grammar you "
                             "have just passed as parameter.")
        if not shutil._samefile(origin_filepath, destination_filepath): # type: ignore
            shutil.copyfile(origin_filepath, destination_filepath)


    def get_grammar(self, path: str) -> dict[NonTerminal, list[Derivation]]:
        raw_grammar: Optional[list[str]] = self.read_grammar(path)
        if raw_grammar is None:
            print('Grammar file does not exist.')
            sys.exit(-1)
        return self.parse_grammar(raw_grammar)


    def read_grammar(self, path: str) -> Optional[list[str]]:
        try:
            with open(path, 'r') as f_in:
                raw_grammar = f_in.readlines()
                return raw_grammar
        except IOError:
            return None


    def parse_grammar(self, raw_grammar: list[str]) -> dict[NonTerminal, list[Derivation]]:
        grammar = {}
        for rule in raw_grammar:
            non_terminal_name, raw_rule_expansions = rule.rstrip('\n').split('::=')
            nt_symbol: Symbol = Symbol.create_symbol(non_terminal_name)
            assert isinstance(nt_symbol, NonTerminal)

            rule_expansions: list[Derivation] = []
            for production_rule in raw_rule_expansions.split('|'):
                rule_expansions.append(Derivation(
                    [Symbol.create_symbol(symbol_name)
                     for symbol_name in production_rule.rstrip().lstrip().split(' ')]
                ))
            grammar[nt_symbol] = rule_expansions
        return grammar


    def __str__(self) -> str:
        print_str = ''
        for _key_ in sorted(self.grammar):
            production_list: list[str] = []
            for production in self.grammar[_key_]:
                symbols: list[str] = [str(symbol) for symbol in production]
                production_list.append(" ".join(symbols))
            print_str += f"{str(_key_)} ::= {' | '.join(production_list)}\n"
        return print_str


    def initialise(self, start_symbol_name: str) -> Genotype:
        start_symbol: NonTerminal = NonTerminal(start_symbol_name)
        genotype: Genotype = self.initialise_recursive(start_symbol)
        #print("===============================================================")
        #print(genotype)
        #print("===============================================================")
        return genotype


    def initialise_recursive(self, symbol_to_expand: Symbol) -> Genotype:
        genotype: Genotype = Genotype(expansions={}, codons={})
        if isinstance(symbol_to_expand, NonTerminal):
            expansion_possibility: int = randint(0, len(self.grammar[symbol_to_expand]) - 1)
            derivation: Derivation = deepcopy(self.grammar[symbol_to_expand][expansion_possibility])
            for expanded_symbol in derivation:
                if isinstance(expanded_symbol, Terminal) and expanded_symbol.attribute is not None:
                    assert expanded_symbol.attribute.values is None
                    # this method has side-effects. The Derivation object is altered because of this
                    expanded_symbol.attribute.generate()
                genotype += self.initialise_recursive(expanded_symbol)
            genotype.add_to_genome(symbol_to_expand, expansion_possibility, derivation, mode='prepend')
        return genotype


    def decode(self,
               start_symbol_name: str,
               genotype: Genotype) -> str:
        start_symbol: NonTerminal = NonTerminal(start_symbol_name)
        phenotype_tokens: list[str]
        unconsumed_genotype: Genotype = deepcopy(genotype)
        extra_genotype: Genotype = Genotype.empty() # to keep track of any extra codons/expansions that were used
        phenotype_tokens = self.decode_recursive(start_symbol, unconsumed_genotype, extra_genotype)
        # if we decoded an individual that has suffered a DSGE mutation we will update the genotype accordingly
        # therefore, we will remove unconsumed codons/expansions and add extra codons/expansions that were used

        # TODO: This logic should be improved in the future
        for k in unconsumed_genotype.expansions.keys():
            n_expansions: int = len(genotype.expansions[k])
            n_unconsumed_expansions: int = len(unconsumed_genotype.expansions[k])
            genotype.expansions[k] = genotype.expansions[k][:n_expansions - n_unconsumed_expansions]
            genotype.codons[k] = genotype.codons[k][:n_expansions - n_unconsumed_expansions]

            if k in extra_genotype.expansions.keys():
                genotype.expansions[k] += extra_genotype.expansions[k]
            if k in extra_genotype.codons.keys():
                genotype.codons[k] += extra_genotype.codons[k]

            if not genotype.expansions[k]:
                genotype.expansions.pop(k)
            if not genotype.codons[k]:
                genotype.codons.pop(k)

        for k in extra_genotype.expansions.keys():
            if k not in genotype.expansions.keys():
                genotype.expansions[k] = extra_genotype.expansions[k]
                genotype.codons[k] = extra_genotype.codons[k]

        phenotype: str = " ".join(phenotype_tokens)
        return phenotype


    def decode_recursive(self,
                         symbol: Symbol,
                         unconsumed_geno: Genotype,
                         extra_genotype: Genotype) -> list[str]:
        phenotype: list[str] = []
        #print(f"-- symbol: {symbol} == unconsumed_geno: {unconsumed_geno}")
        #print(f"-- symbol: {symbol} == extra_genotype: {extra_genotype}")
        if isinstance(symbol, NonTerminal):
            # consume expansion
            expansion: Optional[Derivation] = None
            if symbol in unconsumed_geno.expansions.keys() and \
                len(unconsumed_geno.expansions[symbol]) > 0:
                expansion = unconsumed_geno.expansions[symbol].pop(0)
            # In case there has been a DSGE mutation, a symbol might not have enough codons
            # to continue expanding, thus throwing an error
            if expansion is None:
                expansion_possibility: int = randint(0, len(self.grammar[symbol]) - 1)
                expansion = deepcopy(self.grammar[symbol][expansion_possibility])
                extra_genotype.add_to_genome(symbol, expansion_possibility, expansion, mode='append')
            for expanded_symbol in expansion:
                phenotype += self.decode_recursive(expanded_symbol, unconsumed_geno, extra_genotype)
        else:
            assert isinstance(symbol, Terminal)
            if symbol.attribute is None:
                return [f"{symbol.name}"]
            else:
                if symbol.attribute.values is None:
                    #print(symbol.attribute.values)
                    symbol.attribute.generate()
                    #print(symbol.attribute.values)
                assert symbol.attribute.values is not None
                return [f"{symbol.name}:{','.join(map(str, symbol.attribute.values))}"]
        return phenotype
