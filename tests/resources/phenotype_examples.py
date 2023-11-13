nn_phenotype1: str = "layer:pool_avg kernel_size:4 stride:3 padding:valid"
nn_phenotype2: str = "layer:conv out_channels:97 kernel_size:2 stride:2 padding:valid act:relu bias:False"

simple_phenotype1: str = "( ( ( X ) / ( ( 3 ) ) ) )"
simple_phenotype2: str = "2 + ( ( ( X ) ) )"
simple_phenotype3: str = "( ( ( X ) / ( ( X ) ) ) )"

ind_phenotype1: str = \
    "layer:conv out_channels:98 kernel_size:5 stride:2 padding:valid act:relu bias:False input:-1 " + \
    "layer:conv out_channels:104 kernel_size:3 stride:1 padding:valid act:sigmoid bias:True input:0 " + \
    "layer:conv out_channels:152 kernel_size:2 stride:2 padding:valid act:relu bias:True input:1 " + \
    "layer:conv out_channels:253 kernel_size:4 stride:1 padding:same act:linear bias:False input:2 " + \
    "layer:conv out_channels:243 kernel_size:4 stride:1 padding:valid act:sigmoid bias:True input:3 " + \
    "layer:conv out_channels:243 kernel_size:4 stride:1 padding:valid act:sigmoid bias:True input:4 " + \
    "layer:fc act:softmax out_features:10 bias:True input:5 " + \
    "learning:adam lr:0.010994878808517258 beta1:0.7755784963206666 beta2:0.8532100487924581 " + \
    "weight_decay:0.0005478934704170954 early_stop:19 batch_size:512 epochs:100"
ind_phenotype2: str = \
    "layer:conv out_channels:62 kernel_size:5 stride:2 padding:valid act:sigmoid bias:False input:-1 " + \
    "layer:conv out_channels:39 kernel_size:5 stride:2 padding:same act:sigmoid bias:False input:0 " + \
    "layer:conv out_channels:113 kernel_size:2 stride:1 padding:same act:sigmoid bias:True input:1 " + \
    "layer:conv out_channels:140 kernel_size:2 stride:3 padding:same act:relu bias:False input:2 " + \
    "layer:fc act:softmax out_features:10 bias:True input:3 " + \
    "learning:adam lr:0.09265801171620802 beta1:0.7080483514532836 beta2:0.9580432907690966 " + \
    "weight_decay:0.0009222663739074177 early_stop:12 batch_size:512 epochs:100"

ind_phenotype3: str = \
    "layer:conv out_channels:62 kernel_size:5 stride:2 padding:valid act:sigmoid bias:False input:-1 " + \
    "layer:conv out_channels:39 kernel_size:5 stride:2 padding:same act:sigmoid bias:False input:0 " + \
    "layer:conv out_channels:113 kernel_size:2 stride:1 padding:same act:sigmoid bias:True input:1 " + \
    "layer:conv out_channels:140 kernel_size:2 stride:3 padding:same act:relu bias:False input:2 " + \
    "projector_layer:fc act:linear out_features:512 bias:True input:-1 " + \
    "projector_layer:batch_norm_proj act:relu input:0 " + \
    "projector_layer:fc act:linear out_features:32 bias:True input:1 " + \
    "projector_layer:batch_norm_proj act:relu input:2 " + \
    "projector_layer:fc act:linear out_features:10 bias:True input:3 " + \
    "projector_layer:batch_norm_proj act:linear input:4 " + \
    "projector_layer:identity input:5 " + \
    "learning:adam lr:0.09265801171620802 beta1:0.7080483514532836 beta2:0.9580432907690966 " + \
    "weight_decay:0.0009222663739074177 early_stop:12 batch_size:512 epochs:100"

ind_phenotype4: str = \
    "layer:conv out_channels:62 kernel_size:5 stride:2 padding:valid act:sigmoid bias:False input:-1 " + \
    "layer:conv out_channels:39 kernel_size:5 stride:2 padding:same act:sigmoid bias:False input:0 " + \
    "layer:conv out_channels:113 kernel_size:2 stride:1 padding:same act:sigmoid bias:True input:1 " + \
    "layer:conv out_channels:140 kernel_size:2 stride:3 padding:same act:relu bias:False input:2 " + \
    "projector_layer:fc act:linear out_features:5 bias:True input:-1 " + \
    "projector_layer:batch_norm_proj act:linear input:0 " + \
    "projector_layer:identity input:1 " + \
    "learning:adam lr:0.09265801171620802 beta1:0.7080483514532836 beta2:0.9580432907690966 " + \
    "weight_decay:0.0009222663739074177 early_stop:12 batch_size:512 epochs:100"
