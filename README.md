# chainladder (python)
[![Build Status](https://travis-ci.org/jbogaardt/chainladder-python.svg?branch=master)](https://travis-ci.org/jbogaardt/chainladder-python)
[![Documentation Status](https://readthedocs.org/projects/chainladder-python/badge/?version=latest)](http://chainladder-python.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jbogaardt/chainladder-python/branch/master/graph/badge.svg)](https://codecov.io/gh/jbogaardt/chainladder-python)

## chainladder - Property and Casualty Loss Reserving in Python
This package is highly inspired by the popular [R ChainLadder package](https://github.com/mages/ChainLadder) and where equivalent procedures exist, has been heavily tested against the R package.

A goal of this package is to be minimalistic in needing its own API.  To that end,
we've adopted as much of the pandas API for data manipulation and the scikit-learn API for model construction as possible.  The goal here is to allow an actuary already versed in these tools to easily pick up this package.

## The API
### Triangle
  Basic object for manipulating reserving data.  We've adopted as much of the pandas API for accessing/manipulating data structures in the package as possible.

### Models
While scikit-learn requires numpy arrays for fitting models, the models within
this package require Triangle instances.

#### Development
  - Development - core LDF development techniques
  - MunichAdjustment - paid-to-Incurred Munch Chainladder adjustment

#### Tails
  - TailConstant
  - TailCurve (Exponential, Inverse Power)

#### IBNR Methods
  - Chainladder - Deterministic method
  - BornhuetterFerguson - Deterministic Method
  - Benktander - Deterministic Method
  - CapeCod - Deterministic Method
  - MackChainladder

## Installation
To install using pip:
`pip install chainladder`

Alternatively, install directly from github:
`pip install git+https://github.com/jbogaardt/chainladder-python/`

Note: This package requires Python 3.6 and later, numpy 1.12.0 and later,
pandas 0.23.0 and later, scikit-learn 0.19.0 and later.


## Documentation
Please refer to the [Documentation](http://chainladder-python.readthedocs.io/) for source code documentation

## Usage
|Tutorial|Live Notebook|
|--------|-----|
|[Triangle API](https://github.com/jbogaardt/chainladder-python/blob/triangle_rewrite/notebooks/triangle_demo.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jbogaardt/chainladder-python/blob/triangle_rewrite/notebooks/triangle_demo.ipynb#scrollTo=JTvUhh3GBxrf)|
|Modeling API|TBD|
|Excel Exhibits|TBD|
