# CHANGELOG

## 3.0.0 (2023-11-13)


Features:

- Configuration files are now adapted to a new schema
- New grammars were added, to adapt to the case of evolving projector networks
- Added capability of including extra parameters in the config such as the number of downstream epochs, static projectors, data splits or training mode in the downstream task 
- Projector can be evolved through the framework. This is a breaking change
- More unit tests added
- Structure of the project has been slightly changed. This is a breaking change
- Update framework to use Pytorch 2.0


Documentation updates:

- Added instructions on README about how to run unit tests and get code coverage;
- Update README with new badges and corrected a few typos.


## 2.0.0 (2023-09-03)


Features:

- Structure of the project was changed. This is a breaking change
- Project can be imported as a library as an alternative to CLI


Documentation updates:

- Repo renamed to its new project name - EvoDENSS
- README file updated
- CHANGELOG document was created


## 1.1 (2022-11-24)

Features:

- Adapt code for the possibility of early stop not being defined

## 1.0 (2022-08-18)

Reimplementation of Fast-DENSER on PyTorch