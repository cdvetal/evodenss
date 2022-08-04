# type: ignore
import textwrap
import unittest
import warnings

from fast_denser.evolution.grammar import Grammar

import numpy as np
import random

class Test(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=DeprecationWarning)

    
    def test_read_grammar(self):
        grammar = Grammar('tests/utils/example.grammar')
        output = """\
        <activation_function> ::= act:linear |  act:relu |  act:sigmoid
        <bias> ::= bias:True |  bias:False
        <convolution> ::= layer:conv [out_channels,int,1,32,256] [kernel_size,int,1,2,5] [stride,int,1,1,3] <padding> <activation_function> <bias>
        <features> ::= <convolution>
        <padding> ::= padding:same |  padding:valid
        """
        grammar._str_()
        self.assertEqual(grammar.__str__(), textwrap.dedent(output), "Error: grammars differ")
    

    def test_read_invalid_grammar(self):
        with self.assertRaises(SystemExit) as cm:
            grammar = Grammar('invalid_path')
            self.assertEqual(cm.exception.code, -1, "Error: read invalid grammar")


    def test_initialise(self):
        random.seed(0)
        np.random.seed(0)

        output = {
            'features': [{'ge': 0, 'ga': {}}],
            'convolution': [{'ge': 0, 'ga': {'out_channels': ('int', 32.0, 256.0, [42]), 'kernel_size': ('int', 2.0, 5.0, [4]), 'stride': ('int', 1.0, 3.0, [3])}}],
            'padding': [{'ge': 1, 'ga': {}}],
            'activation_function': [{'ge': 1, 'ga': {}}],
            'bias': [{'ge': 1, 'ga': {}}]
        }

        grammar = Grammar('tests/utils/example.grammar')
        self.assertEqual(grammar.initialise('features'), output, "Error: initialise not equal")


    def test_decode(self):
        grammar = Grammar('tests/utils/example.grammar')
        start_symbol = 'features'
        genotype = {
            'padding': [{'ge': 1, 'ga': {}}],
            'bias': [{'ge': 0, 'ga': {}}],
            'features': [{'ge': 0, 'ga': {}}],
            'activation_function': [{'ge': 2, 'ga': {}}],
            'convolution': [{'ge': 0, 'ga': {'out_channels': ('int', 32.0, 256.0, [242]), 'kernel_size': ('int', 2.0, 5.0, [5]), 'stride': ('int', 1.0, 3.0, [2])}}]
        }
        output = "layer:conv out_channels:242 kernel_size:5 stride:2 padding:valid act:sigmoid bias:True"
        phenotype = grammar.decode(start_symbol, genotype)
        self.assertEqual(phenotype, output, "Error: phenotypes differ")


if __name__ == '__main__':
    unittest.main()
