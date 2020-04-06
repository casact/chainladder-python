# chainladder (python)
[![PyPI version](https://badge.fury.io/py/chainladder.svg)](https://badge.fury.io/py/chainladder)
![Build Status](https://github.com/casact/chainladder-python/workflows/Unit%20Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/chainladder-python/badge/?version=latest)](http://chainladder-python.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jbogaardt/chainladder-python/branch/master/graph/badge.svg)](https://codecov.io/gh/jbogaardt/chainladder-python)

## chainladder - Property and Casualty Loss Reserving in Python
This package gets inpiration from the popular [R ChainLadder package](https://github.com/mages/ChainLadder).

A goal of this package is to be minimalistic in needing its own API.
Think in pandas for data manipulation and scikit-learn for model construction. The idea here is to allow an actuary already versed in these tools to pick up this package with ease. Save your mental energy for actuarial work.


## Documentation
Please visit the [Documentation](https://chainladder-python.readthedocs.io/en/latest/) page for examples, how-tos, and source
code documentation.

## Tutorials
Tutorial notebooks are available for download [here](https://github.com/casact/chainladder-python/tree/master/docs/tutorials).
* [Working with Triangles](https://chainladder-python.readthedocs.io/en/latest/tutorials/triangle-tutorial.html)
* [Selecting Development Patterns](https://chainladder-python.readthedocs.io/en/latest/tutorials/development-tutorial.html)
* [Extending Development Patterns with Tails](https://chainladder-python.readthedocs.io/en/latest/tutorials/tail-tutorial.html)
* [Applying Deterministic Methods](https://chainladder-python.readthedocs.io/en/latest/tutorials/deterministic-tutorial.html)
* [Applying Stochastic Methods](https://chainladder-python.readthedocs.io/en/latest/tutorials/stochastic-tutorial.html)


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

## GPU support
New in version `0.5.0` - `chainladder` now supports CUDA-based GPU computations by way of [CuPY](https://github.com/cupy/cupy).  You can now swap `array_backend` between `numpy` and `cupy` to switch between CPU and GPU-based computations.

Array backends can be set globally:
```python
import chainladder as cl
cl.array_backend('cupy')
```
Alternatively, they can be set per `Triangle` instance.
```python
cl.Triangle(..., array_backend='cupy')
```
**Note** you must have a CUDA-enabled graphics card and [CuPY](https://github.com/cupy/cupy) installed to use the GPU backend.
