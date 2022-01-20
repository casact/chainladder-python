.. -*- mode: rst -*-

.. |PyPI version| image:: https://badge.fury.io/py/chainladder.svg
   :target: https://badge.fury.io/py/chainladder

.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/chainladder.svg
   :target: https://anaconda.org/conda-forge/chainladder

.. |Build Status| image:: https://github.com/casact/chainladder-python/workflows/Unit%20Tests/badge.svg

.. |Documentation Status| image:: https://readthedocs.org/projects/chainladder-python/badge/?version=latest
   :target: http://chainladder-python.readthedocs.io/en/latest/?badge=latest

.. |codecov io| image:: https://codecov.io/github/casact/chainladder-python/coverage.svg?branch=latest
   :target: https://codecov.io/github/casact/chainladder-python?branch=latest

chainladder (python)
====================

|PyPI version| |Conda Version| |Build Status| |codecov io| |Documentation Status|

chainladder: Property and Casualty Loss Reserving in Python
------------------------------------------------------------

Welcome! The chainladder package was built to be able to handle all of your actuarial needs in python. It consists of popular actuarial tools, such as triangle data manipulation, link ratios calculation, and IBNR estimates with both deterministic and stochastic models. We build this package so you no longer have to rely on outdated softwares and tools when performing actuarial pricing or reserving indications.

This package strives to be minimalistic in needing its own API. The syntax mimics popular packages `pandas`_ for data manipulation and `scikit-learn`_ for model
construction. An actuary that is already familiar with these tools will be able to pick up this package with ease. You will be able to save your mental energy for actual actuarial work.

Chainladder is built by a group of volunteers, and we need YOUR help!

.. _pandas: https://pandas.pydata.org/

.. _scikit-learn: https://scikit-learn.org/stable/





Installation
------------

There are two ways to install the chainladder package, using `pip` or `conda`:

1) Using `pip`:

``pip install chainladder``

2) Using `conda`:

``conda install -c conda-forge chainladder``

If you would like to try pre-release features, install the package directly from GitHub.

``pip install git+https://github.com/casact/chainladder-python/``




Getting Started
-------------------------

The package comes pre-loaded with sample insurance datasets that are publicly available. We have also drafted tutorials that use the chainladder package on these datasets to demonstrate some of the commonly used functionalities that the package offers.

Once you have the package installed, we recommend that you follow the starter tutorial and work alongside with the pre-loaded datasets.

`Starter Tutorial`_

.. _Starter Tutorial: https://chainladder-python.readthedocs.io/en/latest/tutorials/triangle-tutorial.html



Documentation and Discussions
-----------------------------

Please visit the `documentation`_ page for examples, how-tos, and source
code documentation.

Do you have a question, a new idea, or a feature request? Join the `discussions`_ on GitHub.  Your question is more likely to get answered here than on Stack Overflow. We are always happy to answer any usage questions or hear ideas on how to make ``chainladder`` better.

.. _documentation: https://chainladder-python.readthedocs.io/en/latest/
.. _discussions: https://github.com/casact/chainladder-python/discussions



Want to Contribute?
-------------------
We welcome volunteers for all aspects of the project. Whether you are new to actuarial reserving, new to python, or both; feedback, questions, suggestions and, of course, contributions are all welcomed. We can all learn from each other, together.

Check out our `contributing guidelines`_.


.. _contributing guidelines: https://chainladder-python.readthedocs.io/en/latest/library/contributing.html


Licenses
-------------------
This package is released under `Mozilla Public License 2.0`_.

.. _Mozilla Public License 2.0: https://github.com/casact/chainladder-python/blob/master/LICENSE
