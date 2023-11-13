# type: ignore
from copy import deepcopy
import os
from parameterized import parameterized
import textwrap
import unittest
# from unittest.mock import patch
import warnings

from evodenss.evolution.grammar import Genotype, Grammar
from tests.resources.genotype_examples import *
from tests.resources.phenotype_examples import *

import numpy as np
import random

class Test(unittest.TestCase):

    def setUp(self):
        self.nn_grammar_path: str = "tests/resources/example.grammar"
        self.simple_grammar_path: str = "tests/resources/simple_grammar.grammar"
        warnings.simplefilter('ignore', category=DeprecationWarning)

    
    def test_read_grammar_nn(self):
        expected_filename: str = "used_grammar.yaml"
        expected_output: str = """\
        <activation_function> ::= act:linear | act:relu | act:sigmoid
        <bias> ::= bias:True | bias:False
        <convolution> ::= layer:conv [out_channels,int,1,32,256] [kernel_size,int,1,2,5] [stride,int,1,1,3] <padding> <activation_function> <bias>
        <features> ::= <convolution> | <pooling>
        <identity> ::= layer:identity
        <padding> ::= padding:same | padding:valid
        <pool_type> ::= layer:pool_avg | layer:pool_max
        <pooling> ::= <pool_type> [kernel_size,int,1,2,5] [stride,int,1,1,3] <padding>
        """
        grammar1: Grammar = Grammar(self.nn_grammar_path)
        self.assertFalse(os.path.exists(expected_filename))
        self.assertEqual(grammar1.__str__(), textwrap.dedent(expected_output))

        grammar2: Grammar = Grammar(self.nn_grammar_path, backup_path='.')
        self.assertTrue(os.path.exists(expected_filename))
        self.assertEqual(grammar2.__str__(), textwrap.dedent(expected_output))
        os.remove(expected_filename)


    def test_read_grammar_simple(self):
        expected_filename: str = "used_grammar.yaml"
        expected_output: str = """\
        <expr> ::= <expr> <op> <expr> | ( <expr> ) | <value>
        <num> ::= 1 | 2 | 3
        <op> ::= + | - | / | *
        <value> ::= <num> | <var>
        <var> ::= X
        """
        grammar1: Grammar = Grammar(self.simple_grammar_path)
        self.assertFalse(os.path.exists(expected_filename))
        self.assertEqual(grammar1.__str__(), textwrap.dedent(expected_output))

        grammar2: Grammar = Grammar(self.simple_grammar_path, backup_path='.')
        self.assertTrue(os.path.exists(expected_filename))
        self.assertEqual(grammar2.__str__(), textwrap.dedent(expected_output))
        os.remove(expected_filename)


    def test_read_invalid_grammar(self):
        with self.assertRaises(SystemExit) as cm:
            _ = Grammar('invalid_path')
            self.assertEqual(cm.exception.code, -1, "Error: read invalid grammar")


    @parameterized.expand([
        (nn_sample1, 0),
        (nn_sample2, 1)
    ])
    def test_initialise_nn(self, expected_output: Genotype, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        grammar = Grammar(self.nn_grammar_path)
        output = grammar.initialise('features')
        self.assertEqual(output, expected_output)


    @parameterized.expand([
        (simple_sample1, 0),
        (simple_sample2, 1)
    ])
    #@patch('evodenss.evolution.grammar.randint')
    def test_initialise_simple(self, expected_output: Genotype, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        #mocked.side_effect = [0, 2, 0, 1, 0, 1, 1, 1, 2, 1, 0]
        grammar = Grammar(self.simple_grammar_path)
        output = grammar.initialise('expr')
        self.assertEqual(output, expected_output)

    
    @parameterized.expand([
        (nn_sample1, nn_phenotype1, nn_sample1),
        (nn_sample2, nn_phenotype2, nn_sample2)
    ])
    def test_decode_nn(self, genotype: Genotype, expected_output: str, fixed_genotype: Genotype):
        grammar = Grammar(self.nn_grammar_path)
        start_symbol = 'features'
        genotype_to_use: Genotype = deepcopy(genotype)
        phenotype = grammar.decode(start_symbol, genotype_to_use)
        self.assertEqual(phenotype, expected_output, "Error: phenotypes differ")
        self.assertEqual(genotype_to_use, fixed_genotype)


    @parameterized.expand([
        (simple_sample1, simple_phenotype1, simple_sample1),
        (simple_sample2, simple_phenotype2, simple_sample2),
        (simple_sample3, simple_phenotype3, dgse_mutated_simple_sample3)
    ])
    def test_decode_simple(self, genotype: Genotype, expected_output: str, fixed_genotype: Genotype):
        grammar = Grammar(self.simple_grammar_path)
        start_symbol = 'expr'
        genotype_to_use: Genotype = deepcopy(genotype)
        phenotype = grammar.decode(start_symbol, genotype_to_use)
        self.assertEqual(phenotype, expected_output, "Error: phenotypes differ")
        # check if after the decoding process, the genotype
        # was updated with new genome and unused genome
        self.assertEqual(genotype_to_use, fixed_genotype)
if __name__ == '__main__':
    unittest.main()
