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

Welcome! The chainladder package was built to be able to handle all of your actuarial needs in python. It consists of popular of actuarial tools, such as triangle data manipulation, link ratios calculation, and IBNR estimates with both deterministic and stochastic models. We build this package so you no longer have to rely on outdated softwares and tools when performing actuarial pricing or reserving indications.

This package strives to be minimalistic in needing its own API. The syntax mimics popular packages `pandas`_ for data manipulation and `scikit-learn`_ for model
construction. An actuary that is already familiar with these tools will be able to pick up this package with ease. You will be able to save your mental energy for actual actuarial work.

Chainladder is built by a group of volunteers, and we need YOUR help!

.. _pandas: https://pandas.pydata.org/

.. _scikit-learn: https://scikit-learn.org/stable/





Installation
------------

There are two ways to install the chainladder package, using `pip` or `conda`:

1) Using `pip`:

`pip install chainladder`

2) Using `conda`:

`conda install -c conda-forge chainladder`

If you would like to try pre-release features, install the package directly from GitHub.

`pip install git+https://github.com/casact/chainladder-python/`




Getting Started
-------------------------

The package comes pre-load with sample insurance datasets that are publicly available. We have also drafted tutorials that use the chainladder package on these datasets to demonstrate some of the commonly used functionalities that the package offers.

Once you have the package installed, we recommend that you follow the starter tutorial and work alongside with the pre-loaded datasets.

`Starter Tutorial`_

.. _Starter Tutorial: https://chainladder-python.readthedocs.io/en/latest/tutorials/triangle-tutorial.html





Estimators
--------------------

chainladder has an ever growing list of estimators that work seamlessly together, here are some examples:

Loss Development:

* `Development`_
* `DevelopmentConstant`_
* `MunichAdjustment`_
* `ClarkLDF`_
* `IncrementalAdditive`_
* `CaseOutstanding`_
* `TweedieGLM`_
* `DevelopmentML`_
* `BarnettZehnwirth`_

Tail Factors:

* `TailCurve`_
* `TailConstant`_
* `TailBondy`_
* `TailClark`_

Adjustments:

* `BootstrapODPSample`_
* `BerquistSherman`_
* `ParallelogramOLF`_
* `Trend`_

IBNR Models:

* `Chainladder`_
* `MackChainladder`_
* `BornhuettterFerguson`_
* `Benktander`_
* `CapeCod`_

Workflow:

* `VotingChainladder`_
* `Pipeline`_
* `GridSearch`_


.. _Development: https://chainladder-python.readthedocs.io/en/latest/development.html#development
.. _TailCurve: https://chainladder-python.readthedocs.io/en/latest/tails.html#tailcurve
.. _Chainladder: https://chainladder-python.readthedocs.io/en/latest/methods.html#chainladder
.. _BootstrapODPSample: https://chainladder-python.readthedocs.io/en/latest/adjustments.html#bootstrapodpsample
.. _DevelopmentConstant: https://chainladder-python.readthedocs.io/en/latest/development.html#developmentconstant
.. _TailConstant: https://chainladder-python.readthedocs.io/en/latest/tails.html#tailconstant
.. _MackChainladder: https://chainladder-python.readthedocs.io/en/latest/methods.html#mackchainladder
.. _BerquistSherman: https://chainladder-python.readthedocs.io/en/latest/adjustments.html#berquistsherman
.. _MunichAdjustment: https://chainladder-python.readthedocs.io/en/latest/development.html#munichadjustment
.. _TailBondy: https://chainladder-python.readthedocs.io/en/latest/tails.html#tailbondy
.. _BornhuettterFerguson: https://chainladder-python.readthedocs.io/en/latest/methods.html#bornhuetterferguson
.. _Pipeline: https://chainladder-python.readthedocs.io/en/latest/workflow.html#pipeline
.. _ClarkLDF: https://chainladder-python.readthedocs.io/en/latest/development.html#clarkldf
.. _TailClark: https://chainladder-python.readthedocs.io/en/latest/tails.html#tailclark
.. _Benktander: https://chainladder-python.readthedocs.io/en/latest/methods.html#benktander
.. _GridSearch: https://chainladder-python.readthedocs.io/en/latest/workflow.html#gridsearch
.. _IncrementalAdditive: https://chainladder-python.readthedocs.io/en/latest/development.html#incrementaladditive
.. _CapeCod: https://chainladder-python.readthedocs.io/en/latest/methods.html#capecod
.. _ParallelogramOLF: https://chainladder-python.readthedocs.io/en/latest/adjustments.html#parallelogramolf
.. _VotingChainladder: https://chainladder-python.readthedocs.io/en/latest/workflow.html#votingchainladder
.. _Trend: https://chainladder-python.readthedocs.io/en/latest/adjustments.html#trend
.. _CaseOutstanding: https://chainladder-python.readthedocs.io/en/latest/development.html#caseoutstanding
.. _TweedieGLM: https://chainladder-python.readthedocs.io/en/latest/development.html#tweedieglm
.. _DevelopmentML: https://chainladder-python.readthedocs.io/en/latest/development.html#developmentml
.. _BarnettZehnwirth: https://chainladder-python.readthedocs.io/en/latest/development.html#barnettzehnwirth





Documentation
-------------

Please visit the `Documentation`_ page for examples, how-tos, and source
code documentation.

.. _Documentation: https://chainladder-python.readthedocs.io/en/latest/




Discussion Board
--------------------

Do you have a question, a new idea, or a feature request? Join the discussions on `GitHub`_.  Your question is more likely to get answered here than on Stack Overflow. We are always happy to answer any usage questions or hear ideas on how to make ``chainladder`` better.

.. _GitHub: https://github.com/casact/chainladder-python/discussions



Want to Contribute?
-------------------
We welcome volunteers for all aspects of the project. Whether you are new to actuarial reserving, new to python, or both; feedback, questions, suggestions and, of course, contributions are all welcomed. We can all learn from each other, together.

Check out our `contributing guidelines`_.


.. _contributing guidelines: https://chainladder-python.readthedocs.io/en/latest/library/contributing.html
