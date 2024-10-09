# CHANGELOG

## 4.0.0 (2024-10-09)

Features:

- Each individual tracks the mutations that it got from the very beginning until present point
- In case of evolution with SSL learning, projector networks can either be statically defined or evolved
- lambda parameter from BarlowTwins is now being evolved as well rather than statically defined
- Downstream parameters can now be fully defined from configuration (not just the number of epochs, but optimiser details too)
- Control over each data subset can be done with more granularity in the config. The size of each subset can be defined by ratios or absolute numbers
- User can now define if a data subset needs to be static regardless of the seed/run, or if the data subset depends on the seed/run that is passed via CLI
- Configuration parameters can be overriden via CLI using `--override`
- Evolution using supervised learning allows for partial evolution. One can do so by loading representation model as an input to an evolved network. Representation weights can be either frozen or finetuned

Bug Fixes:

- Logging is now working more appropriately by redirecting to both stdin and a log file. It works at the INFO level for stdin, and at DEBUG level for a file
- Fixed bug where config option to freeze or finetune representations was not really working
- In experiments with limited labeled data, we ensure that the run only uses that limited data and nothing more. Previously, evo_test subset was not being included in the percentage of labeled data defined by the user

Other improvements:

- Configuration parameters are now defined and handled with [Pydantic](https://docs.pydantic.dev/)
- Barlow Twins Loss decoupled from the BarlowTwinsNetwork class to allow the inclusion of different loss functions in the future


## 3.1.0 (2024-01-12)

Features:

- Model artifacts have now an associated metadata file associated which can be used to further extend the train of the model
- Add new CLI to extend the train of individuals trained with Barlow Twins

Bug Fixes:

- Removed model processing checkppoint as it was producing incorrect results

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