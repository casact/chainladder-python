# chainladder (python)
[![PyPI version](https://badge.fury.io/py/chainladder.svg)](https://badge.fury.io/py/chainladder)
![Build Status](https://github.com/casact/chainladder-python/workflows/Unit%20Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/chainladder-python/badge/?version=master)](http://chainladder-python.readthedocs.io/en/latest/?badge=master)
[![codecov](https://codecov.io/gh/jbogaardt/chainladder-python/branch/master/graph/badge.svg)](https://codecov.io/gh/jbogaardt/chainladder-python)

## chainladder - Property and Casualty Loss Reserving in Python
This package is highly inspired by the popular [R ChainLadder package](https://github.com/mages/ChainLadder) and where equivalent procedures exist, has been heavily tested against the R package.

A goal of this package is to be minimalistic in needing its own API.  To that end,
we've adopted as much of the pandas API for data manipulation and the scikit-learn API for model construction as possible.  The idea here is to allow an actuary already versed in these tools to easily pick up this package.
We figure an actuary who uses python has reasonable familiarity with pandas and
scikit-learn, so they can spend as little mental energy as possible learning yet
another API.

## Documentation
Please visit the [Documentation](https://chainladder-python.readthedocs.io/en/latest/) page for examples, how-tos, and source
code documentation.

## Have a question?
Feel free to reach out on [Gitter](https://gitter.im/chainladder-python/community).

## Want to contribute?
Check out our [contributing guidelines](https://github.com/casact/chainladder-python/blob/master/CONTRIBUTING.md).

## Installation
To install using pip:
`pip install chainladder`

Alternatively, install directly from github:
`pip install git+https://github.com/casact/chainladder-python/`

Note: This package requires Python 3.5 and later, numpy 1.12.0 and later,
pandas 0.23.0 and later, scikit-learn 0.18.0 and later.
