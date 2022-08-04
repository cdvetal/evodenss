# Fast-DENSER torch

## Installing dependencies

Fast Denser Torch works with:

- Python: 3.9
- Pytorch: 1.12.0
- Cudatoolkit: 11.3

Assuming you've got conda installed, you can install the dependencies by typing:
- `conda env create -f environment.yml`

Alternatively, you can use the requirements.txt file, but, you will be on your own to install cudatoolkit and other libraries that might be required to enable GPU acceleration

## Run fast denser

- In order to run fast denser torch, you need to run the fast_denser module as a script, in the following way:
    - `python3 -m fast_denser.main -d <dataset_name> -c <config_path> -g <grammar_path> -r <#run>`   
    Example: `python3 -m fast_denser.main -d mnist -c ../example/config_legacy.yaml -g ../example/cnn.grammar --run 0 --cuda-enabled`

