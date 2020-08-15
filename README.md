# chainladder (python)
[![PyPI version](https://badge.fury.io/py/chainladder.svg)](https://badge.fury.io/py/chainladder)
![Build Status](https://github.com/casact/chainladder-python/workflows/Unit%20Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/chainladder-python/badge/?version=latest)](http://chainladder-python.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jbogaardt/chainladder-python/branch/master/graph/badge.svg)](https://codecov.io/gh/jbogaardt/chainladder-python)

## chainladder - Property and Casualty Loss Reserving in Python
This package gets inpiration from the popular [R ChainLadder package](https://github.com/mages/ChainLadder).

This package strives to be minimalistic in needing its own API.
Think in [pandas](https://pandas.pydata.org/) for data manipulation and [scikit-learn](https://scikit-learn.org/stable/index.html) for model construction. An actuary already versed in these tools will pick up this package with ease. Save your mental energy for actuarial work.

## Available Estimators
`chainladder` has an ever growing list of estimators:

|Loss Development|Tails Factors|IBNR Models|Adjustments & Workflow|
| - | - | - | - |
|[Development](https://chainladder-python.readthedocs.io/en/latest/modules/development.html#basic-development)|[TailCurve](https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#ldf-curve-fitting)	   |[Chainladder](https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#basic-chainladder)          |[BootstrapODPSample](https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#bootstrap-sampling)
|[DevelopmentConstant](https://chainladder-python.readthedocs.io/en/latest/modules/development.html#external-patterns)| [TailConstant](https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#external-data) |[MackChainladder](https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#mack-chainladder)      |[BerquistSherman](https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#berquist-sherman)
|[MunichAdjustment](https://chainladder-python.readthedocs.io/en/latest/modules/development.html#munich-adjustment)| [TailBondy](https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#the-bondy-tail)    |[BornhuettterFerguson](https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#bornhuetter-ferguson)|	[Pipeline](https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#pipeline)
|[ClarkLDF](https://chainladder-python.readthedocs.io/en/latest/modules/development.html#growth-curve-fitting)|	[TailClark](https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#growth-curve-extrapolation)	|[Benktander](https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#benktander)|[GridSearch](https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#gridsearch)
|[IncrementalAdditive](https://chainladder-python.readthedocs.io/en/latest/modules/development.html#incremental-additive)|              |[CapeCod](https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#cape-cod)

## Documentation
Please visit the [Documentation](https://chainladder-python.readthedocs.io/en/latest/) page for examples, how-tos, and source
code documentation.

## Getting Started Tutorials
Tutorial notebooks are available for download [here](https://github.com/casact/chainladder-python/tree/master/docs/tutorials).
* [Working with Triangles](https://chainladder-python.readthedocs.io/en/latest/tutorials/triangle-tutorial.html)
* [Selecting Development Patterns](https://chainladder-python.readthedocs.io/en/latest/tutorials/development-tutorial.html)
* [Extending Development Patterns with Tails](https://chainladder-python.readthedocs.io/en/latest/tutorials/tail-tutorial.html)
* [Applying Deterministic Methods](https://chainladder-python.readthedocs.io/en/latest/tutorials/deterministic-tutorial.html)
* [Applying Stochastic Methods](https://chainladder-python.readthedocs.io/en/latest/tutorials/stochastic-tutorial.html)
* [Large Datasets](https://chainladder-python.readthedocs.io/en/latest/tutorials/large-datasets.html)

## Installation
To install using pip:
`pip install chainladder`

Alternatively, install directly from github:
`pip install git+https://github.com/casact/chainladder-python/`

Note: This package requires Python 3.5 and later, numpy 1.12.0 and later,
pandas 0.23.0 and later, scikit-learn 0.18.0 and later.

## Questions?
Feel free to reach out on [Gitter](https://gitter.im/chainladder-python/community).

## Want to contribute?
Check out our [contributing guidelines](https://github.com/casact/chainladder-python/blob/master/CONTRIBUTING.md).
