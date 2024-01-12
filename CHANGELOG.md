# CHANGELOG

## 3.1.0 (2024-01-12)

Features:

- Model artifacts have now an associated metadata file associated which can be used to further extend the train of the model
- Add new CLI to extend the train of individuals trained with Barlow Twins

Bug Fixes:

- Removed model processing checkppoint as it producing incorrect results

## 3.0.2 (2023-12-10)


Bug Fixes:

- Genotype was not being correctly updated after it suffers DSGE mutation
- Refactoring introduced a bug in the remove layer mutation which is being addressed in this new version


## 3.0.1 (2023-11-16)


Bug Fixes:

- Fixed calculation of a fitness fuction based on loss, for the supervised learning case
- Set up Github action to put linting and unit testing in place, to enforce better code quality. This allowed to fix several miscellaneous syntax problems.

Documentation updates:

- Update README by fixing typos.


## 3.0.0 (2023-11-13)


Features:

- Configuration files are now adapted to a new schema
- New grammars were added, to adapt to the case of evolving projector networks
- Added capability of including extra parameters in the config such as the number of downstream epochs, static projectors, data splits or training mode in the downstream task 
- Projector can be evolved through the framework. This is a breaking change
- More unit tests added
- Structure of the project has been slightly changed. This is a breaking change
- Update framework to use Pytorch 2.0. With the support of Pytorch 2.0, a dependency of `llvm-openmp<16` was added, as advised per this [issue](https://github.com/pytorch/pytorch/issues/102269)

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