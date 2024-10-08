from evodenss.evolution.grammar import Attribute, Derivation, Genotype, NonTerminal, Terminal


def generate_attribute(genotype: Genotype,
                       non_terminal: NonTerminal,
                       derivation_idx: int,
                       symbol_idx: int) -> None:
    terminal_symbol = genotype.expansions[non_terminal][derivation_idx][symbol_idx]
    assert isinstance(terminal_symbol, Terminal)
    assert terminal_symbol.attribute is not None
    terminal_symbol.attribute.generate()


nn_sample1: Genotype = Genotype(
    expansions={
        NonTerminal("features"): [Derivation([NonTerminal("pooling")])],
        NonTerminal("pooling"): [Derivation(
           [NonTerminal("pool_type"),
            Terminal(name='kernel_size',
                     attribute=Attribute[int]("int", 1, 2, 5, lambda x,y,z: [4])),
            Terminal(name='stride',
                     attribute=Attribute[int]("int", 1, 1, 3, lambda x,y,z: [3])),
            NonTerminal(name='padding')])
        ],
        NonTerminal("pool_type"): [Derivation([Terminal(name='layer:pool_avg', attribute=None)])],
        NonTerminal(name='padding'): [Derivation([Terminal(name='padding:valid', attribute=None)])],
        NonTerminal(name='padding'): [Derivation([Terminal(name='padding:valid', attribute=None)])]
    },
    codons={
        NonTerminal(name='features'): [1],
        NonTerminal(name='pooling'): [0],
        NonTerminal(name='pool_type'): [0],
        NonTerminal(name='padding'): [1]
    }
)
generate_attribute(nn_sample1, NonTerminal("pooling"), 0, 1)
generate_attribute(nn_sample1, NonTerminal("pooling"), 0, 2)


nn_sample2: Genotype = Genotype(
    expansions={
        NonTerminal("features"): [Derivation([NonTerminal("convolution")])],
        NonTerminal("convolution"): [Derivation(
            [Terminal(name='layer:conv', attribute=None),
             Terminal(name='out_channels',
                      attribute=Attribute[int]("int", 1, 32, 256, lambda x,y,z: [97])),
             Terminal(name='kernel_size',
                      attribute=Attribute[int]("int", 1, 2, 5, lambda x,y,z: [2])),
             Terminal(name='stride',
                      attribute=Attribute[int]("int", 1, 1, 3, lambda x,y,z: [2])),
             NonTerminal(name='padding'),
             NonTerminal(name='activation_function'), NonTerminal(name='bias')])
        ],
        NonTerminal(name='padding'): [Derivation([Terminal(name='padding:valid', attribute=None)])],
        NonTerminal(name='activation_function'): [Derivation([Terminal(name='act:relu', attribute=None)])],
        NonTerminal(name='bias'): [Derivation([Terminal(name='bias:False', attribute=None)])]
    },
    codons={
        NonTerminal(name='features'): [0],
        NonTerminal(name='convolution'): [0],
        NonTerminal(name='padding'): [1],
        NonTerminal(name='activation_function'): [1],
        NonTerminal(name='bias'): [1]}
)
generate_attribute(nn_sample2, NonTerminal("convolution"), 0, 1)
generate_attribute(nn_sample2, NonTerminal("convolution"), 0, 2)
generate_attribute(nn_sample2, NonTerminal("convolution"), 0, 3)


simple_sample1 = Genotype(
    expansions={
        NonTerminal(name='var'): [Derivation([Terminal(name='X', attribute=None)])],
		NonTerminal(name='value'): [Derivation([NonTerminal(name='var')]), Derivation([NonTerminal(name='num')])],
		NonTerminal(name='expr'): [
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
			Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
			Derivation([NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')]),
			Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
			Derivation([NonTerminal(name='value')]),
			Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
			Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
			Derivation([NonTerminal(name='value')])
		],
		NonTerminal(name='op'): [Derivation([Terminal(name='/', attribute=None)])],
		NonTerminal(name='num'): [Derivation([Terminal(name='3', attribute=None)])]
    },
    codons={
		NonTerminal(name='var'): [0],
		NonTerminal(name='value'): [1, 0],
		NonTerminal(name='expr'): [1, 1, 0, 1, 2, 1, 1, 2],
		NonTerminal(name='op'): [2],
		NonTerminal(name='num'): [2]
	}
)


simple_sample2 = Genotype(
    expansions={
        NonTerminal(name='num'): [Derivation([Terminal(name='2', attribute=None)])],
        NonTerminal(name='value'): [Derivation([NonTerminal(name='num')]), Derivation([NonTerminal(name='var')])],
        NonTerminal(name='expr'): [
            Derivation([NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')]),
            Derivation([NonTerminal(name='value')]),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation([NonTerminal(name='value')])
        ],
        NonTerminal(name='op'): [Derivation([Terminal(name='+', attribute=None)])],
        NonTerminal(name='var'): [Derivation([Terminal(name='X', attribute=None)])]
    },
    codons={
        NonTerminal(name='num'): [1],
        NonTerminal(name='value'): [0, 1],
        NonTerminal(name='expr'): [0, 2, 1, 1, 1, 2],
        NonTerminal(name='op'): [0],
        NonTerminal(name='var'): [0]
    }
)

simple_sample3 = Genotype(
    expansions={
        NonTerminal(name='var'): [Derivation([Terminal(name='X', attribute=None)])],
        NonTerminal(name='value'): [Derivation([NonTerminal(name='var')]), Derivation([NonTerminal(name='var')])],
        NonTerminal(name='expr'): [
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
            Derivation([NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')]),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
            Derivation([NonTerminal(name='value')]),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]),
            Derivation([NonTerminal(name='value')])
            ],
        NonTerminal(name='op'): [Derivation([Terminal(name='/', attribute=None)])
        ],
        NonTerminal(name='num'): [Derivation([Terminal(name='3', attribute=None)])]
    },
    codons={
        NonTerminal(name='var'): [0],
        NonTerminal(name='value'): [1, 1],
        NonTerminal(name='expr'): [1, 1, 0, 1, 2, 1, 1, 2],
        NonTerminal(name='op'): [2],
        NonTerminal(name='num'): [2]
    }
)


ind_test_output1 = Genotype(
    expansions={NonTerminal(name='softmax'): [Derivation([
        Terminal(name='layer:fc', attribute=None),
        Terminal(name='act:softmax', attribute=None),
        Terminal(name='out_features:10', attribute=None),
        Terminal(name='bias:True', attribute=None)
    ])]},
    codons={NonTerminal(name='softmax'): [0]}
)


ind_test_learning1 = Genotype(
    expansions={
        NonTerminal(name='adam'): [Derivation(
            [Terminal(name='learning:adam', attribute=None),
             Terminal(name='lr', attribute=Attribute("float", 1, 0.0001, 0.1, lambda x,y,z: [0.010994878808517258])),
             Terminal(name='beta1', attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.7755784963206666])),
             Terminal(name='beta2', attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.8532100487924581])),
             Terminal(name='weight_decay',
                      attribute=Attribute("float", 1, 1e-06, 0.001, lambda x,y,z: [0.0005478934704170954]))])
        ],
        NonTerminal(name='early_stop'): [
            Derivation([Terminal(name='early_stop', attribute=Attribute("int", 1, 10, 20, lambda x,y,z: [19]))])
        ],
        NonTerminal(name='learning'): [Derivation(
            [NonTerminal(name='adam'),
             NonTerminal(name='early_stop'),
             Terminal(name='batch_size', attribute=Attribute("int_power2", 1, 5, 12, lambda x,y,z: [512])),
             Terminal(name='epochs:100', attribute=None)])]
    },
    codons={
        NonTerminal(name='adam'): [0],
        NonTerminal(name='early_stop'): [0],
        NonTerminal(name='learning'): [0]
    }
)
generate_attribute(ind_test_learning1, NonTerminal("adam"), 0, 1)
generate_attribute(ind_test_learning1, NonTerminal("adam"), 0, 2)
generate_attribute(ind_test_learning1, NonTerminal("adam"), 0, 3)
generate_attribute(ind_test_learning1, NonTerminal("adam"), 0, 4)
generate_attribute(ind_test_learning1, NonTerminal("early_stop"), 0, 0)
generate_attribute(ind_test_learning1, NonTerminal("learning"), 0, 2)


ind_test_output1 = ind_test_output2 = Genotype(
    expansions={NonTerminal(name='softmax'): [Derivation([
        Terminal(name='layer:fc', attribute=None),
        Terminal(name='act:softmax', attribute=None),
        Terminal(name='out_features:10', attribute=None),
        Terminal(name='bias:True', attribute=None)
    ])]},
    codons={NonTerminal(name='softmax'): [0]}
)

ind_test_learning2 = Genotype(
    expansions={
        NonTerminal(name='adam'): [Derivation(
            [Terminal(name='learning:adam', attribute=None),
             Terminal(name='lr', attribute=Attribute("float", 1, 0.0001, 0.1, lambda x,y,z: [0.09265801171620802])),
             Terminal(name='beta1', attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.7080483514532836])),
             Terminal(name='beta2',
                      attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.9580432907690966])),
             Terminal(name='weight_decay',
                      attribute=Attribute("float", 1, 1e-06, 0.001, lambda x,y,z: [0.0009222663739074177]))])
        ],
        NonTerminal(name='early_stop'): [
            Derivation([Terminal(name='early_stop', attribute=Attribute("int", 1, 10, 20, lambda x,y,z: [12]))])
        ],
        NonTerminal(name='learning'): [Derivation(
            [NonTerminal(name='adam'),
             NonTerminal(name='early_stop'),
             Terminal(name='batch_size', attribute=Attribute("int_power2", 1, 5, 12, lambda x,y,z: [512])),
             Terminal(name='epochs:100', attribute=None)])]
    },
    codons={
        NonTerminal(name='adam'): [0],
        NonTerminal(name='early_stop'): [0],
        NonTerminal(name='learning'): [0]
    }
)
generate_attribute(ind_test_learning2, NonTerminal("adam"), 0, 1)
generate_attribute(ind_test_learning2, NonTerminal("adam"), 0, 2)
generate_attribute(ind_test_learning2, NonTerminal("adam"), 0, 3)
generate_attribute(ind_test_learning2, NonTerminal("adam"), 0, 4)
generate_attribute(ind_test_learning2, NonTerminal("early_stop"), 0, 0)
generate_attribute(ind_test_learning2, NonTerminal("learning"), 0, 2)


mutation_added_layer_genotype = Genotype(
    expansions={
        NonTerminal(name='padding'): [
            Derivation([Terminal(name='padding:valid', attribute=None)])
        ],
        NonTerminal(name='activation_function'): [
            Derivation([Terminal(name='act:relu', attribute=None)])
        ],
        NonTerminal(name='bias'): [
            Derivation([Terminal(name='bias:True', attribute=None)])
        ],
        NonTerminal(name='convolution'): [Derivation(
            [Terminal(name='layer:conv', attribute=None),
             Terminal(name='out_channels',
                      attribute=Attribute(var_type='int',
                                          num_values=1,
                                          min_value=32,
                                          max_value=256,
                                          generator=lambda x,y,z: [213])),
             Terminal(name='kernel_size',
                      attribute=Attribute(var_type='int',
                                          num_values=1,
                                          min_value=2,
                                          max_value=5,
                                          generator=lambda x,y,z: [2])),
             Terminal(name='stride',
                      attribute=Attribute(var_type='int',
                                          num_values=1,
                                          min_value=1,
                                          max_value=3,
                                          generator=lambda x,y,z: [3])),
             NonTerminal(name='padding'), NonTerminal(name='activation_function'), NonTerminal(name='bias')])
        ],
        NonTerminal(name='features'): [
            Derivation([NonTerminal(name='convolution')])
        ]
    },
    codons={
        NonTerminal(name='padding'): [1],
        NonTerminal(name='activation_function'): [1],
        NonTerminal(name='bias'): [0],
        NonTerminal(name='convolution'): [0],
        NonTerminal(name='features'): [0]
    }
)
generate_attribute(mutation_added_layer_genotype, NonTerminal("convolution"), 0, 1)
generate_attribute(mutation_added_layer_genotype, NonTerminal("convolution"), 0, 2)
generate_attribute(mutation_added_layer_genotype, NonTerminal("convolution"), 0, 3)


dgse_mutated_simple_sample3 = Genotype(
    expansions={
        NonTerminal(name='var'): [
            Derivation([Terminal(name='X', attribute=None)]), Derivation([Terminal(name='X', attribute=None)])],
        NonTerminal(name='value'): [Derivation([NonTerminal(name='var')]), Derivation([NonTerminal(name='var')])],
        NonTerminal(name='expr'): [
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation([NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')]),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation([NonTerminal(name='value')]),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation(
                [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)]
            ),
            Derivation([NonTerminal(name='value')])
        ],
        NonTerminal(name='op'): [Derivation([Terminal(name='/', attribute=None)])]
    },
    codons={
        NonTerminal(name='var'): [0, 0],
        NonTerminal(name='value'): [1, 1],
        NonTerminal(name='expr'): [1, 1, 0, 1, 2, 1, 1, 2],
        NonTerminal(name='op'): [2]
    }
)
