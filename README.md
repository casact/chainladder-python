# chainladder (python)
[![Build Status](https://travis-ci.org/jbogaardt/chainladder-python.svg?branch=master)](https://travis-ci.org/jbogaardt/chainladder-python)
[![Documentation Status](https://readthedocs.org/projects/chainladder-python/badge/?version=latest)](http://chainladder-python.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/jbogaardt/chainladder-python/branch/master/graph/badge.svg)](https://codecov.io/gh/jbogaardt/chainladder-python)

### chainladder - Property and Casualty Loss Reserving in Python
This package is highly inspired by the popular [R ChainLadder package](https://github.com/mages/ChainLadder).

This code is actively under development and contains the following functionality:

### Stochastic Methods
  - Mack Chainladder
  - Munich Chainladder
  - Bootstrap Chainladder
### Deterministic Methods
  - Development Factor Method
  - Bornhuetter-Ferguson Method
  - Benktander Method
  - Generalized Cape Cod Method


### The API
We've adopted as much of the pandas API for accessing/manipulating data structures
in the package as possible.  For fitting actuarial models, the scikit-learn API
was adopted.  The goal here is to allow an actuary already versed in these tools
to easily pick up this package.


## Installation
To install using pip:
`pip install chainladder`

Note: This package requires Python 3.6 and later.


## Documentation
Please refer to the [Documentation](http://chainladder-python.readthedocs.io/) for source code documentation

## Tutorials
-[Getting Started with Triangles](https://github.com/jbogaardt/chainladder-python/blob/triangle_rewrite/notebooks/triangle_demo.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jbogaardt/chainladder-python/blob/triangle_rewrite/notebooks/triangle_demo.ipynb#scrollTo=JTvUhh3GBxrf)

 -[Quickstart Guide](http://chainladder-python.readthedocs.io/en/master/quickstart.html)  
 -[Using Mack Chainladder](http://chainladder-python.readthedocs.io/en/master/MackExample.html)  
 -[Using Munich Chainladder](http://chainladder-python.readthedocs.io/en/master/MunichExample.html)  
 -[Using Bootstrap Chainladder](http://chainladder-python.readthedocs.io/en/master/BootstrapExample.html)
