# EvoDENSS
[![](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/) [![](https://img.shields.io/badge/PyTorch-2.0.0-blue.svg)](https://pytorch.org/get-started/previous-versions/) [![](https://img.shields.io/badge/cudatoolkit-11.3-blue.svg)](https://developer.nvidia.com/cuda-downloads/)

[![](https://img.shields.io/badge/License-Apache_2.0-green.svg)]()

<!---
![t](https://img.shields.io/badge/status-maintained-green.svg)
[![](https://img.shields.io/github/license/adrianovinhas/fast-denser-adriano.svg)](https://github.com/adrianovinhas/fast-denser-adriano/blob/master/LICENSE.md)
-->

EvoDENSS stands for **Evo**lution of **DE**ep **N**etworks through **S**elf **S**upervision. EvoDENSS performs Neuro-Evolution by evolving the structure of the networks and optimiser aspects (the optimiser instance to use and relevant hyperparameters). Being inspired on Fast-DENSER, it uses self-supervised learning to train the generated networks in order to measure their fitness.

## Installing

In order to run EvoDENSS, one needs to install the relevant dependencies. There are two ways to install the framework:

##### 1. Conda
A conda environment can be created from an exported yml file that contains all the required dependences:
```
conda env create -f environment.yml
```

After the environment is created, just activate it in order to be able to run your code:
```
conda activate evodenss
```

##### 2. pip
Alternatively, you can use the `requirements.txt` file, but you will be on your own to install cudatoolkit and other libraries that might be required to enable GPU acceleration.
```
pip install -r requirements.txt
```

**Note:** Installing EvoDENSS as a Python library is not yet supported

## Running EvoDENSS

- In order to run fast EvoDENSS, you need to run the evodenss module as a script:

```
python3 -m evodenss.main \
    -d <dataset_name> \
    -c <config_path> \
    -g <grammar_path> \
    -r <#run>
```

Example:
```
python3 -m evodenss.main \
    -d mnist \
    -c example/example_config.yaml \
    -g example/example.grammar \
    --run 0 \
    --gpu-enabled
```

In case several seeds are needed to be run, that can be done with Bash:
```
for i in {7..9}; do \
python3 -m evodenss.main \
-d cifar10 \
-c config_files/bt_10.yaml \
-g grammars/bt.grammar \
-r $i --gpu-enabled; \
done
```

Externally to the code itself, two main files are required to execute any run.
1. A grammar that shapes the search space by setting the possibilities within each macro block.
2. A configuration file that sets miscellaneous parameters that affect the outcome of the evolutionary run. These can se related with the evolutionary process itself, or the networks that are generated.

## Testing

Unit tests can be executed via `pytest`:
```
pytest tests
```
In case one wants to do it with coverage report:
```
coverage run --source evodenss -m pytest -v tests
coverage report
```

#### Command-line flags

- `-c`/`--config-path`: Sets the path to the config file to be used;
- `-d`/`--dataset-name`: Name of the dataset to be used. At the moment, `mnist`, `fashion-mnist`, `cifar10` and `cifar100` are supported.
- `-g`/`--grammar-path`: Sets the path to the grammar to be used;
- `-r`/`--run`: Identifies the run id and seed to be used;
- `--gpu-enabled`: When used, it enables GPU processing.


#### Framework modes

EvoDENSS can be run using the supervised learning mode behaviour and the self-supervised learning mode. In the case of self-supervised learning there are a few variations that influence which components are targeted by Evolutionary Computation.

###### 1. Supervised learning mode

Config files that should be used:
- `supervised_10.yaml` to evaluate evolved networks by training them on 10% labelled data
- `supervised_100.yaml` to evaluate evolved networks by training them on 100% labelled data

Grammar files that should be used:
- `supervised.grammar` for CIFAR-10
- `supervised_cifar100.grammar` for CIFAR-100

###### 2. Self-Supervised learning mode

Config files that should be used:
- Any config file that starts with `bt_`

Grammar files that should be used:
- `bt_with_projector.grammar` can be used without problems because the projector related grammar derivations will never be expanded by the evolutionary engine if the right config file is provided.

###### 3. Self-Supervised learning mode with evolvable projector

Config files that should be used:
- Any config file that starts with contains `projector`

Grammar files that should be used:
- `bt_with_projector.grammar`


#### Extending train to best individuals

##### Command-line flags

This small cli was create to refine a specific SSL model that was evolved and it takes advantages of the evolution outputs from EvoDeNSS

- `--model-path`: Path to the model file created by Pytorch;
- `--weights-path`: Path to the weights file created by Pytorch;
- `--metadata-path`: Path to the metadata file created by EvoDeNSS with relevant details/params used to evaluate the individual;
- `--output-model-path`: The output folder destination for the refined model;
- `--pretext-epochs`: Number of training epochs for the pretext task. If this number is lower than the number of epochs already trained by a certain individual in the pretext task, it skips the pretext task;
- `--downstream-epochs`: Number of training epochs for the downstream task;
- `--gpu-enabled`: When used, it enables GPU processing;
- `--config-path`: Path to the config file used during the evolution phase;
- `--individual-path`: Path the to the individual pickle file;
- `--downstream-mode`: Decides whether to `freeze` representation weights or `finetune`;
- `-r`: Identifies the run id and seed to be used. 

##### Example
```
python3 -m evodenss.train_longer  \
    --model-path path_to_model.pt  \
    --weights-path path_to_weightsweights.pt  \
    --metadata-path path_to_metadata \
    --output-model-path output_path_of_refined_model \
    --pretext-epochs 100 \
    --downstream-epochs 600 \
    --gpu-enabled \
    --config-path path_to_config_used_to_evolve_the_model.yaml \
    --individual-path path_to_the_individual_pickle_obj.pkl \
    --downstream-mode freeze \
    -r $i
```

Note: To refine models using a supervised learning paradigm, use evodenss.train_longer_supervised