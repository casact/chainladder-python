# chainladder (python)
[![Build Status](https://travis-ci.org/jbogaardt/chainladder-python.svg?branch=master)](https://travis-ci.org/jbogaardt/chainladder-python)
[![Documentation Status](https://readthedocs.org/projects/chainladder-python/badge/?version=latest)](http://chainladder-python.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jbogaardt/chainladder-python/branch/master/graph/badge.svg)](https://codecov.io/gh/jbogaardt/chainladder-python)

This is a rewrite of the popular [R ChainLadder package](https://github.com/mages/ChainLadder) in Python.



This code is actively under development and contains the following functionality:

### Stochastic Methods
  - Mack Chain Ladder
  - Munich Chain Ladder
  - Bootstrap Chain Ladder
### Deterministic Methods
  - Development Factor Method
  - Bornhuetter-Ferguson Method
  - Benktander Method
  - Generalized Cape Cod Method


### The API
We've adopted as much of the pandas API for accessing/manipulating data structures
in the package, and scikit-learn for the model fitting routines.  The goal here is to
allow an actuary already versed in these tools to easily pick up this package.


## Installation
To install using pip:
`pip install chainladder`

Note: This package requires Python 3.6 and later.


## Documentation
Please refer to the [Documentation](http://chainladder-python.readthedocs.io/) for source code documentation

## Tutorials
 -[Quickstart Guide](http://chainladder-python.readthedocs.io/en/master/quickstart.html)  
 -[Using Mack Chainladder](http://chainladder-python.readthedocs.io/en/master/MackExample.html)  
 -[Using Munich Chainladder](http://chainladder-python.readthedocs.io/en/master/MunichExample.html)  
 -[Using Bootstrap Chainladder](http://chainladder-python.readthedocs.io/en/master/BootstrapExample.html)
