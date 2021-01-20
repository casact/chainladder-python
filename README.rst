.. -*- mode: rst -*-

chainladder (python)
====================

|PyPI version| |Conda Version| |Build Status| |codecov io| |Documentation Status|

chainladder - Property and Casualty Loss Reserving in Python
------------------------------------------------------------

This package gets inpiration from the popular `R ChainLadder package`_.

This package strives to be minimalistic in needing its own API. Think in
`pandas`_ for data manipulation and `scikit-learn`_ for model
construction. An actuary already versed in these tools will pick up this
package with ease. Save your mental energy for actuarial work.

Available Estimators
--------------------

``chainladder`` has an ever growing list of estimators that work seemlessly together:

.. _R ChainLadder package: https://github.com/mages/ChainLadder
.. _pandas: https://pandas.pydata.org/
.. _scikit-learn: https://scikit-learn.org/stable/index.html

.. |PyPI version| image:: https://badge.fury.io/py/chainladder.svg
   :target: https://badge.fury.io/py/chainladder

.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/chainladder.svg
   :target: https://anaconda.org/conda-forge/chainladder

.. |Build Status| image:: https://github.com/casact/chainladder-python/workflows/Unit%20Tests/badge.svg

.. |Documentation Status| image:: https://readthedocs.org/projects/chainladder-python/badge/?version=latest
   :target: http://chainladder-python.readthedocs.io/en/latest/?badge=latest

.. |codecov io| image:: https://codecov.io/github/casact/chainladder-python/coverage.svg?branch=master
   :target: https://codecov.io/github/casact/chainladder-python?branch=master



+------------------------------+------------------+-------------------------+-----------------------+
| Loss                         | Tails Factors    | IBNR Models             | Adjustments &         |
| Development                  |                  |                         | Workflow              |
+==============================+==================+=========================+=======================+
| `Development`_               | `TailCurve`_     | `Chainladder`_          | `BootstrapODPSample`_ |
+------------------------------+------------------+-------------------------+-----------------------+
| `DevelopmentConstant`_       | `TailConstant`_  | `MackChainladder`_      | `BerquistSherman`_    |
+------------------------------+------------------+-------------------------+-----------------------+
| `MunichAdjustment`_          | `TailBondy`_     | `BornhuettterFerguson`_ | `Pipeline`_           |
+------------------------------+------------------+-------------------------+-----------------------+
| `ClarkLDF`_                  | `TailClark`_     | `Benktander`_           | `GridSearch`_         |
+------------------------------+------------------+-------------------------+-----------------------+
| `IncrementalAdditive`_       |                  | `CapeCod`_              | `ParallelogramOLF`_   |
+------------------------------+------------------+-------------------------+-----------------------+
|                              |                  |                         | `Trend`_              |
+------------------------------+------------------+-------------------------+-----------------------+

Documentation
-------------

Please visit the `Documentation`_ page for examples, how-tos, and source
code documentation.

.. _Development: https://chainladder-python.readthedocs.io/en/latest/modules/development.html#basic-development
.. _TailCurve: https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#ldf-curve-fitting
.. _Chainladder: https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#basic-chainladder
.. _BootstrapODPSample: https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#bootstrap-sampling
.. _DevelopmentConstant: https://chainladder-python.readthedocs.io/en/latest/modules/development.html#external-patterns
.. _TailConstant: https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#external-data
.. _MackChainladder: https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#mack-chainladder
.. _BerquistSherman: https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#berquist-sherman
.. _MunichAdjustment: https://chainladder-python.readthedocs.io/en/latest/modules/development.html#munich-adjustment
.. _TailBondy: https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#the-bondy-tail
.. _BornhuettterFerguson: https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#bornhuetter-ferguson
.. _Pipeline: https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#pipeline
.. _ClarkLDF: https://chainladder-python.readthedocs.io/en/latest/modules/development.html#growth-curve-fitting
.. _TailClark: https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#growth-curve-extrapolation
.. _Benktander: https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#benktander
.. _GridSearch: https://chainladder-python.readthedocs.io/en/latest/modules/workflow.html#gridsearch
.. _IncrementalAdditive: https://chainladder-python.readthedocs.io/en/latest/modules/development.html#incremental-additive
.. _CapeCod: https://chainladder-python.readthedocs.io/en/latest/modules/methods.html#cape-cod
.. _ParallelogramOLF: https://chainladder-python.readthedocs.io/en/latest/modules/generated/chainladder.ParallelogramOLF.html#chainladder.ParallelogramOLF
.. _Trend: https://chainladder-python.readthedocs.io/en/latest/modules/generated/chainladder.Trend.html#chainladder.Trend
.. _Documentation: https://chainladder-python.readthedocs.io/en/latest/

Getting Started Tutorials
-------------------------

Tutorial notebooks are available for download `here`_.

* `Working with Triangles`_
* `Selecting Development Patterns`_
* `Extending Development Patterns with Tails`_
* `Applying Deterministic Methods`_
* `Applying Stochastic Methods`_
* `Large Datasets`_

Installation
------------

To install using pip: ``pip install chainladder``

To instal using conda: ``conda install -c conda-forge chainladder``

Alternatively, install directly from github:
``pip install git+https://github.com/casact/chainladder-python/``

Note: This package requires Python 3.5 and later, numpy 1.12.0 and
later, pandas 0.23.0 and later, scikit-learn 0.18.0 and later.

Questions?
----------

Feel free to reach out on `Gitter`_.

Want to contribute?
-------------------

Check out our `contributing guidelines`_.

.. _here: https://github.com/casact/chainladder-python/tree/master/docs/tutorials
.. _Working with Triangles: https://chainladder-python.readthedocs.io/en/latest/tutorials/triangle-tutorial.html
.. _Selecting Development Patterns: https://chainladder-python.readthedocs.io/en/latest/tutorials/development-tutorial.html
.. _Extending Development Patterns with Tails: https://chainladder-python.readthedocs.io/en/latest/tutorials/tail-tutorial.html
.. _Applying Deterministic Methods: https://chainladder-python.readthedocs.io/en/latest/tutorials/deterministic-tutorial.html
.. _Applying Stochastic Methods: https://chainladder-python.readthedocs.io/en/latest/tutorials/stochastic-tutorial.html
.. _Large Datasets: https://chainladder-python.readthedocs.io/en/latest/tutorials/large-datasets.html
.. _Gitter: https://gitter.im/chainladder-python/community
.. _contributing guidelines: https://github.com/casact/chainladder-python/blob/master/CONTRIBUTING.md
