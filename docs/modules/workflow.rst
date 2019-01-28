.. _workflow:

.. currentmodule:: chainladder

==========
Workflow
==========

.. _pipeline:
Pipeline
========
The :class:`Pipeline` class implements utilities to build a composite
estimator, as a chain of transforms and estimators.

.. _gridsearch:
GridSearch
==========
The grid search provided by :class:`GridSearch` exhaustively generates
candidates from a grid of parameter values specified with the ``param_grid``
parameter.
