# {octicon}`code-square` API Reference
This is the class and function reference of chainladder. Please refer to
the full user guide for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

```{eval-rst}

:mod:`chainladder.core`: Triangle
=================================

.. automodule:: chainladder.core
  :no-members:
  :no-inherited-members:


Classes
-------
.. currentmodule:: chainladder

.. autosummary::
  :toctree: generated/
  :template: class_inherited.rst

  Triangle
  DevelopmentCorrelation
  ValuationCorrelation


.. _development_ref:

:mod:`chainladder.development`: Development Patterns
====================================================

.. automodule:: chainladder.development
  :no-members:
  :no-inherited-members:


Classes
-------
.. currentmodule:: chainladder

.. autosummary::
  :toctree: generated/
  :template: class.rst

  Development
  DevelopmentConstant
  MunichAdjustment
  IncrementalAdditive
  ClarkLDF
  CaseOutstanding
  TweedieGLM
  DevelopmentML
  BarnettZehnwirth


.. _tails_ref:

:mod:`chainladder.tails`: Tail Factors
========================================

.. automodule:: chainladder.tails
  :no-members:
  :no-inherited-members:


Classes
-------
.. currentmodule:: chainladder

.. autosummary::
  :toctree: generated/
  :template: class.rst

  TailConstant
  TailCurve
  TailBondy
  TailClark


.. _methods_ref:

:mod:`chainladder.methods`: IBNR Methods
========================================

.. automodule:: chainladder.methods
   :no-members:
   :no-inherited-members:


Classes
-------
.. currentmodule:: chainladder

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Chainladder
   MackChainladder
   BornhuetterFerguson
   Benktander
   CapeCod

.. _adjustments_ref:

:mod:`chainladder.workflow`: Adjustments
========================================
.. automodule:: chainladder.workflow
 :no-members:
 :no-inherited-members:

Classes
-------
.. currentmodule:: chainladder

.. autosummary::
 :toctree: generated/
 :template: class.rst

 BootstrapODPSample
 BerquistSherman
 Trend
 ParallelogramOLF

.. _workflow_ref:

:mod:`chainladder.workflow`: Workflow
=====================================
.. automodule:: chainladder.workflow
  :no-members:
  :no-inherited-members:

Classes
-------
.. currentmodule:: chainladder

.. autosummary::
  :toctree: generated/
  :template: class.rst

  Pipeline
  VotingChainladder
  GridSearch


.. _utils_ref:

:mod:`chainladder.utils`: Utilities
===================================
.. automodule:: chainladder.utils
   :no-members:
   :no-inherited-members:


Functions
---------
.. currentmodule:: chainladder

.. autosummary::
   :toctree: generated/
   :template: function.rst

   load_sample
   read_pickle
   read_json
   concat
   load_sample
   minimum
   maximum

Classes
-------
.. currentmodule:: chainladder

.. autosummary::
 :toctree: generated/
 :template: class.rst

 PatsyFormula

```