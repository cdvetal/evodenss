# type: ignore

from evodenss.evolution.grammar import Attribute, Genotype, NonTerminal, Terminal


nn_sample1: Genotype = Genotype(
    expansions={
        NonTerminal("features"): [[NonTerminal("pooling")]],
        NonTerminal("pooling"): [
           [NonTerminal("pool_type"),
            Terminal(name='kernel_size',
                     attribute=Attribute[int]("int", 1, 2, 5, lambda x,y,z: [4])),
            Terminal(name='stride',
                     attribute=Attribute[int]("int", 1, 1, 3, lambda x,y,z: [3])),
            NonTerminal(name='padding')]
        ],
        NonTerminal("pool_type"): [[Terminal(name='layer:pool_avg', attribute=None)]],
        NonTerminal(name='padding'): [[Terminal(name='padding:valid', attribute=None)]],
        NonTerminal(name='padding'): [[Terminal(name='padding:valid', attribute=None)]]
    },
    codons={
        NonTerminal(name='features'): [1],
        NonTerminal(name='pooling'): [0],
        NonTerminal(name='pool_type'): [0],
        NonTerminal(name='padding'): [1]
    }
)
nn_sample1.expansions[NonTerminal("pooling")][0][1].attribute.generate()
nn_sample1.expansions[NonTerminal("pooling")][0][2].attribute.generate()


nn_sample2: Genotype = Genotype(
    expansions={
        NonTerminal("features"): [[NonTerminal("convolution")]],
        NonTerminal("convolution"): [
            [Terminal(name='layer:conv', attribute=None),
             Terminal(name='out_channels',
                      attribute=Attribute[int]("int", 1, 32, 256, lambda x,y,z: [97])),
             Terminal(name='kernel_size',
                      attribute=Attribute[int]("int", 1, 2, 5, lambda x,y,z: [2])),
             Terminal(name='stride',
                      attribute=Attribute[int]("int", 1, 1, 3, lambda x,y,z: [2])),
             NonTerminal(name='padding'),
             NonTerminal(name='activation_function'), NonTerminal(name='bias')]
        ],
        NonTerminal(name='padding'): [[Terminal(name='padding:valid', attribute=None)]],
        NonTerminal(name='activation_function'): [[Terminal(name='act:relu', attribute=None)]],
        NonTerminal(name='bias'): [[Terminal(name='bias:False', attribute=None)]]
    },
    codons={
        NonTerminal(name='features'): [0],
        NonTerminal(name='convolution'): [0],
        NonTerminal(name='padding'): [1],
        NonTerminal(name='activation_function'): [1],
        NonTerminal(name='bias'): [1]}
)
nn_sample2.expansions[NonTerminal("convolution")][0][1].attribute.generate()
nn_sample2.expansions[NonTerminal("convolution")][0][2].attribute.generate()
nn_sample2.expansions[NonTerminal("convolution")][0][3].attribute.generate()


simple_sample1 = Genotype(
    expansions={
        NonTerminal(name='var'): [[Terminal(name='X', attribute=None)]],
		NonTerminal(name='value'): [[NonTerminal(name='var')], [NonTerminal(name='num')]],
		NonTerminal(name='expr'): [
	        [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
			[Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
			[NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')],
			[Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
			[NonTerminal(name='value')],
			[Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
			[Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
			[NonTerminal(name='value')]
		],
		NonTerminal(name='op'): [[Terminal(name='/', attribute=None)]],
		NonTerminal(name='num'): [[Terminal(name='3', attribute=None)]]
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
        NonTerminal(name='num'): [[Terminal(name='2', attribute=None)]],
        NonTerminal(name='value'): [[NonTerminal(name='num')], [NonTerminal(name='var')]],
        NonTerminal(name='expr'): [
            [NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')],
            [NonTerminal(name='value')],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [NonTerminal(name='value')]
        ],
        NonTerminal(name='op'): [[Terminal(name='+', attribute=None)]],
        NonTerminal(name='var'): [[Terminal(name='X', attribute=None)]]
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
        NonTerminal(name='var'): [[Terminal(name='X', attribute=None)]],
        NonTerminal(name='value'): [[NonTerminal(name='var')], [NonTerminal(name='var')]],
        NonTerminal(name='expr'): [
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [NonTerminal(name='value')],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [NonTerminal(name='value')]], NonTerminal(name='op'): [[Terminal(name='/', attribute=None)]
        ],
        NonTerminal(name='num'): [[Terminal(name='3', attribute=None)]]
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
    expansions={NonTerminal(name='softmax'): [[
        Terminal(name='layer:fc', attribute=None),
        Terminal(name='act:softmax', attribute=None),
        Terminal(name='out_features:10', attribute=None),
        Terminal(name='bias:True', attribute=None)
    ]]},
    codons={NonTerminal(name='softmax'): [0]}
)


ind_test_macro1 = Genotype(
    expansions={
        NonTerminal(name='adam'): [
            [Terminal(name='learning:adam', attribute=None),
             Terminal(name='lr', attribute=Attribute("float", 1, 0.0001, 0.1, lambda x,y,z: [0.010994878808517258])),
             Terminal(name='beta1', attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.7755784963206666])),
             Terminal(name='beta2', attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.8532100487924581])),
             Terminal(name='weight_decay',
                      attribute=Attribute("float", 1, 1e-06, 0.001, lambda x,y,z: [0.0005478934704170954]))]
        ],
        NonTerminal(name='early_stop'): [
            [Terminal(name='early_stop', attribute=Attribute("int", 1, 10, 20, lambda x,y,z: [19]))]
        ],
        NonTerminal(name='learning'): [
            [NonTerminal(name='adam'),
             NonTerminal(name='early_stop'),
             Terminal(name='batch_size', attribute=Attribute("int_power2", 1, 5, 12, lambda x,y,z: [512])),
             Terminal(name='epochs:100', attribute=None)]]
    },
    codons={
        NonTerminal(name='adam'): [0],
        NonTerminal(name='early_stop'): [0],
        NonTerminal(name='learning'): [0]
    }
)
ind_test_macro1.expansions[NonTerminal(name='adam')][0][1].attribute.generate()
ind_test_macro1.expansions[NonTerminal(name='adam')][0][2].attribute.generate()
ind_test_macro1.expansions[NonTerminal(name='adam')][0][3].attribute.generate()
ind_test_macro1.expansions[NonTerminal(name='adam')][0][4].attribute.generate()
ind_test_macro1.expansions[NonTerminal(name='early_stop')][0][0].attribute.generate()
ind_test_macro1.expansions[NonTerminal(name='learning')][0][2].attribute.generate()


ind_test_output1 = ind_test_output2 = Genotype(
    expansions={NonTerminal(name='softmax'): [[
        Terminal(name='layer:fc', attribute=None),
        Terminal(name='act:softmax', attribute=None),
        Terminal(name='out_features:10', attribute=None),
        Terminal(name='bias:True', attribute=None)
    ]]},
    codons={NonTerminal(name='softmax'): [0]}
)

ind_test_macro2 = Genotype(
    expansions={
        NonTerminal(name='adam'): [
            [Terminal(name='learning:adam', attribute=None),
             Terminal(name='lr', attribute=Attribute("float", 1, 0.0001, 0.1, lambda x,y,z: [0.09265801171620802])),
             Terminal(name='beta1', attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.7080483514532836])),
             Terminal(name='beta2',
                      attribute=Attribute("float", 1, 0.5, 0.9999, lambda x,y,z: [0.9580432907690966])),
             Terminal(name='weight_decay',
                      attribute=Attribute("float", 1, 1e-06, 0.001, lambda x,y,z: [0.0009222663739074177]))]
        ],
        NonTerminal(name='early_stop'): [
            [Terminal(name='early_stop', attribute=Attribute("int", 1, 10, 20, lambda x,y,z: [12]))]
        ],
        NonTerminal(name='learning'): [
            [NonTerminal(name='adam'),
             NonTerminal(name='early_stop'),
             Terminal(name='batch_size', attribute=Attribute("int_power2", 1, 5, 12, lambda x,y,z: [512])),
             Terminal(name='epochs:100', attribute=None)]]
    },
    codons={
        NonTerminal(name='adam'): [0],
        NonTerminal(name='early_stop'): [0],
        NonTerminal(name='learning'): [0]
    }
)
ind_test_macro2.expansions[NonTerminal(name='adam')][0][1].attribute.generate()
ind_test_macro2.expansions[NonTerminal(name='adam')][0][2].attribute.generate()
ind_test_macro2.expansions[NonTerminal(name='adam')][0][3].attribute.generate()
ind_test_macro2.expansions[NonTerminal(name='adam')][0][4].attribute.generate()
ind_test_macro2.expansions[NonTerminal(name='early_stop')][0][0].attribute.generate()
ind_test_macro2.expansions[NonTerminal(name='learning')][0][2].attribute.generate()


mutation_added_layer_genotype = Genotype(
    expansions={
        NonTerminal(name='padding'): [
            [Terminal(name='padding:same', attribute=None)]
        ],
        NonTerminal(name='activation_function'): [
            [Terminal(name='act:sigmoid', attribute=None)]
        ],
        NonTerminal(name='bias'): [
            [Terminal(name='bias:False', attribute=None)]
        ],
        NonTerminal(name='convolution'): [
            [Terminal(name='layer:conv', attribute=None),
             Terminal(name='out_channels',
                      attribute=Attribute(var_type='int', num_values=1, min_value=32, max_value=256, generator=lambda x,y,z: [216])),
             Terminal(name='kernel_size',
                      attribute=Attribute(var_type='int', num_values=1, min_value=2, max_value=5, generator=lambda x,y,z: [5])),
             Terminal(name='stride',
                      attribute=Attribute(var_type='int', num_values=1, min_value=1, max_value=3, generator=lambda x,y,z: [3])),
             NonTerminal(name='padding'), NonTerminal(name='activation_function'), NonTerminal(name='bias')]
        ],
        NonTerminal(name='features'): [
            [NonTerminal(name='convolution')]
        ]
    },
    codons={
        NonTerminal(name='padding'): [0],
        NonTerminal(name='activation_function'): [2],
        NonTerminal(name='bias'): [1],
        NonTerminal(name='convolution'): [0],
        NonTerminal(name='features'): [0]
    }
)
mutation_added_layer_genotype.expansions[NonTerminal(name='convolution')][0][1].attribute.generate()
mutation_added_layer_genotype.expansions[NonTerminal(name='convolution')][0][2].attribute.generate()
mutation_added_layer_genotype.expansions[NonTerminal(name='convolution')][0][3].attribute.generate()

dgse_mutated_simple_sample3 = Genotype(
    expansions={
        NonTerminal(name='var'): [[Terminal(name='X', attribute=None)], [Terminal(name='X', attribute=None)]],
        NonTerminal(name='value'): [[NonTerminal(name='var')], [NonTerminal(name='var')]],
        NonTerminal(name='expr'): [
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [NonTerminal(name='expr'), NonTerminal(name='op'), NonTerminal(name='expr')],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [NonTerminal(name='value')],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [Terminal(name='(', attribute=None), NonTerminal(name='expr'), Terminal(name=')', attribute=None)],
            [NonTerminal(name='value')]], NonTerminal(name='op'): [[Terminal(name='/', attribute=None)]
        ],
        NonTerminal(name='num'): []
    },
    codons={
        NonTerminal(name='var'): [0, 0],
        NonTerminal(name='value'): [1, 1],
        NonTerminal(name='expr'): [1, 1, 0, 1, 2, 1, 1, 2],
        NonTerminal(name='op'): [2],
        NonTerminal(name='num'): []
    }
)
