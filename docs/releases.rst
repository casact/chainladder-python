=========
Releases
=========

Version 0.8
===========

Version 0.8.1
--------------
Release Date: Feb 28, 2021

**Enhancements**

-  Included a ``truncation_age`` in the ``TailClark`` estimator to
   replicate examples from the paper
-  `#129 <https://github.com/casact/chainladder-python/issues/129>`__
   Included new development estimators ``TweedieGLM`` and
   ``DevelopmentML`` to take advantage of a broader set of
   (sklearn-compliant) regression frameworks.
-  Included a helper Transformer ``PatsyFormula`` to make working
   with ML algorithms easier.

**Bug Fixes**

-  `#128 <https://github.com/casact/chainladder-python/issues/128>`__
   Made ``IncrementalAdditive`` method comply with the rest of the
   package API

Version 0.8.0
--------------
Release Date: Feb 15, 2021.

**Enhancements**

-  `#112 <https://github.com/casact/chainladder-python/issues/112>`__
   Advanced ``groupby`` support.
-  `#113 <https://github.com/casact/chainladder-python/issues/113>`__
   Added ``reg_threshold`` argument to ``TailCurve`` for
   finer control of fit. Huge thanks to
   `Brian-13 <https://github.com/Brian-13>`__ for the
   contribution.
-  `#115 <https://github.com/casact/chainladder-python/issues/115>`__
   Introducing ``VotingChainladder`` workflow estimator to
   allow for averaging multiple IBNR estimators Huge thanks
   to `cbalona <https://github.com/cbalona>`__ for the
   contribution.
-  Introduced ``CaseOutstanding`` development estimator
-  `#124 <https://github.com/casact/chainladder-python/issues/124>`__
   Standardized ``CapCod`` functionality to that of
   ``Benktander`` and ``BornhuetterFerguson``

**Bug Fixes**

-  `#83 <https://github.com/casact/chainladder-python/issues/83>`__
   Fixed ``grain`` issues when using the ``trailing``
   argument
-  `#119 <https://github.com/casact/chainladder-python/issues/119>`__
   Fixed pickle compatibility issue introduced in
   ``v0.7.12``
-  `#120 <https://github.com/casact/chainladder-python/issues/120>`__
   Fixed ``Trend`` estimator API
-  `#121 <https://github.com/casact/chainladder-python/issues/121>`__
   Fixed numpy==1.20.1 compatibility issue
-  `#123 <https://github.com/casact/chainladder-python/issues/123>`__
   Fixed ``BerquistSherman`` estimator
-  `#125 <https://github.com/casact/chainladder-python/issues/125>`__
   Fixed various issues with compatibility of estimators in
   a ``Pipeline``

**Deprecations**

-  `#118 <https://github.com/casact/chainladder-python/issues/118>`__
   Deprecation warnings on ``xlcompose``. Will be removed as
   a depdendency in v0.9.0


Version 0.7
===========

Version 0.7.12
--------------
Release Date: Jan 19, 2021

No code changes from ``0.7.11``. Bump release to fix conda
packaging.


Version 0.7.11
--------------
Release Date: Jan 18, 2021

**Enhancements**

-  `#110 <https://github.com/casact/chainladder-python/issues/110>`__
   Added ``virtual_column`` functionality

**Bug Fixes**

-  Minor bug fix on ``sort_index`` not accepting kwargs
-  Minor bug fix in ``DevelopmentConstant`` when using a
   callable

**Misc**

-  Bringing release up to parity with docs.

Version 0.7.10
--------------
Release Date: Jan 16, 2021


**Bug Fixes**

-  `#108 <https://github.com/casact/chainladder-python/issues/108>`__
   - ``sample_weight`` error handling on predict - thank you
   `cbalona <https://github.com/cbalona>`__
-  `#107 <https://github.com/casact/chainladder-python/issues/107>`__
   - Latest diagonal of empty triangle now resolves
-  `#106 <https://github.com/casact/chainladder-python/issues/106>`__
   - Improved ``loc`` consistency with pandas
-  `#105 <https://github.com/casact/chainladder-python/issues/105>`__
   - Addressed broadcasting error during triangle arithmetic
-  `#103 <https://github.com/casact/chainladder-python/issues/103>`__
   - Fixed Index alignment with triangle arithmetic
   consistent with ``pd.Series``

Version 0.7.9
--------------
Release Date: Nov 5, 2020

**Bug Fixes**

-  `#101 <https://github.com/casact/chainladder-python/issues/101>`__
   Bug where LDF labels were not aligned with underlying LDF
   array

**Enhancements**

-  `#66 <https://github.com/casact/chainladder-python/issues/66>`__
   Allow for onleveling with new ``ParallelogramOLF``
   transformer
-  `#98 <https://github.com/casact/chainladder-python/issues/98>`__
   Allow for more complex trends in estimators with
   ``Trend`` transformer
   Refer to this
   `example <https://chainladder-python.readthedocs.io/en/latest/auto_examples/plot_capecod_onlevel.html#sphx-glr-auto-examples-plot-capecod-onlevel-py>`__
   on how to apply the new estimators.

Version 0.7.8
--------------
Release Date: Oct 22, 2020

**Bug Fixes**

-  Resolved
   `#87 <https://github.com/casact/chainladder-python/issues/87>`__
   val_to_dev with malformed triangle

**Enhancements**

-  Major overhaul of Triangle internals for better code
   clarity and more efficiency
-  Made sparse operations more efficient for larger
   triangles
-  ``to_frame`` now works on Triangles that are 3D or 4D.
   For example ``clrd.to_frame()``
-  Advanced ``groupby`` operations supported. For (trivial)
   example:


  >>> clrd = cl.load_sample('clrd')
  >>> # Split companies with names less than 15 characters vs those above:
  >>> clrd.groupby(clrd.index['GRNAME'].str.len()<15).sum()


Version 0.7.7
--------------
Release Date: Sep 13, 2020

**Enhancements**

-  `#97 <https://github.com/casact/chainladder-python/issues/97>`__,
   loc and iloc now support Ellipsis
-  ``Development`` can now take a float value for averaging.
   When float value is used, it corresponds to weight
   exponent (delta in Barnett/Zenwirth). Only special cases
   had previously existed -
   ``{"regression": 0.0, "volume": 1.0, "simple": 2.0}``
-  Major improvements in slicing performance.

**Bug Fixes**

-  `#96 <https://github.com/casact/chainladder-python/issues/96>`__,
   Fix for TailBase transform
-  `#94 <https://github.com/casact/chainladder-python/issues/94>`__,
   ``n_periods`` with asymmetric triangles fixed


Version 0.7.6
--------------
Release Date: Aug 26, 2020

**Enhancements**

-  Four Dimensional slicing is now supported.

  >>> clrd = cl.load_sample('clrd')
  >>> clrd.iloc[[0,10, 3], 1:8, :5, :]
  >>> clrd.loc[:'Aegis Grp', 'CumPaidLoss':, '1990':'1994', :48]

-  `#92 <https://github.com/casact/chainladder-python/issues/92>`__
   to_frame() now takes optional ``origin_as_datetime``
   argument for better compatibility with various plotting
   libraries (Thank you
   `johalnes <https://github.com/johalnes>`__ )

   >>> tri.to_frame(origin_as_datetime=True)

**Bug Fixes**

-  Patches to the interaction between ``sparse`` and
   ``numpy`` arrays to accomodate more scenarios.
-  Patches to multi-index broadcasting
-  Improved performance of ``latest_diagonal`` for sparse
   backends
-  `#91 <https://github.com/casact/chainladder-python/issues/91>`__
   Bug fix to ``MackChainladder`` which errored on
   asymmetric triangles (Thank you
   `johalnes <https://github.com/johalnes>`__ for
   reporting)

Version 0.7.5
--------------
Release Date: Aug 15, 2020

**Enhancements**

-  Enabled multi-index broadcasting.

 >>> clrd = cl.load_sample('clrd')
 >>> clrd / clrd.groupby('LOB').sum()  # LOB alignment works now instead of throwing error

-  Added sparse representation of triangles which substantially
increases the size limit of in-memory triangles. Check out
the new `Large
Datasets <https://chainladder-python.readthedocs.io/en/latest/tutorials/large-datasets.html>`__
tutorial for details

**Bug Fixes**

-  Fixed cupy backend which had previously been neglected
-  Fixed xlcompose issue where Period fails when included as
   column header

Version 0.7.4
--------------
Release Date: Jul 26, 2020

**Bug Fixes**

-  Fixed a bug where Triangle did not support full accident
   dates at creation
-  Fixed an inappropriate index mutation in Triangle index

**Enhancements**

-  Added ``head`` and ``tail`` methods to Triangle
-  Prepped Triangle class to support sparse backend
-  Added prism sample dataset for sparse demonstrations and
   unit tests

Version 0.7.3
--------------
Release Date: Jul 11, 2020

**Enhancements**

-  Improved performance of ``valuation`` axis
-  Improved performance of ``groupby``
-  Added ``sort_index`` method to ``Triangle`` consistent
   with pandas
-  Allow for ``fit_predict`` to be called on a ``Pipeline``
   estimator

**Bug Fixes**

-  Fixed issue with Bootstrap process variance where it was
   being applied more than once
-  Fixed but where Triangle.index did not ingest numeric
   columns appropriately.

Version 0.7.2
--------------
Release Date: Jul 1, 2020

**Bug Fixes**

-  Index slicing not compatible with pandas
   `#84 <https://github.com/casact/chainladder-python/issues/84>`__
   fixed
-  arithmetic fail
   `#68 <https://github.com/casact/chainladder-python/issues/68>`__
   - Substantial reworking of how arithmetic works.
-  JSON IO on sub-triangles now works
-  ``predict`` and ``fit_predict`` methods added to all IBNR
   models and now function as expected

**Enhancements**

-  Allow ``DevelopmentConstant`` to take on more than one
   set of patterns by passing in a callable
-  ``MunichAdjustment``\ Allow \` does not work when P/I or
   I/P ratios cannot be calculated. You can now optionally
   back-fill zero values with expectaton from simple
   chainladder so that Munich can be performed on sparser
   triangles.

**Refactors**

-  Performance optimized several triangle functions
   including slicing and ``val_to_dev``
-  Reduced footprint of ``ldf_``, ``sigma``, and
   ``std_err_`` triangles
-  Standardized IBNR model methods
-  Changed ``cdf_``, ``full_triangle_``,
   ``full_expectation_``, ``ibnr_`` to function-based
   properties instead of in-memory objects to reduce memory
   footprint

Version 0.7.1
--------------
Release Date: Jun 22, 2020

**Enhancements**

-  Added heatmap method to Triangle - allows for
   conditionally formatting a 2D triangle. Useful for
   detecting ``link_ratio`` outliers
-  Introduced BerquistSherman estimator
-  Better error messaging when triangle columns are
   non-numeric
-  Broadened the functionality of ``Triangle.trend``
-  Allow for nested estimators in ``to_json``. Required
   addition for the new ``BerquistSherman`` method
-  Docs, docs, and more docs.

**Bug Fixes**

-  Mixed an inappropriate mutation in
  ``MunichAdjustment.transform``
-  Triangle column slicing now supports pd.Index objects
   instead of just lists

**Misc**

-  Moved ``BootstrapODPSample`` to workflow section as it is
   not a development estimator.

Version 0.7.0
--------------
Release Date: Jun 2, 2020

**Bug Fixes**

-  ``TailBondy`` now works with multiple (4D) triangles
-  ``TailBondy`` computes correctly when ``earliest_age`` is
   selected
-  Sub-triangles now honor index and column slicing of the
   parent.
-  ``fit_transform`` for all tail estimators now correctly
   propagate all estimator attributes
-  ``Bondy`` decay now uses the generalized Bondy formula
   instead of exponential decay

**Enhancements**

-  Every tail estimator now has a ``tail_`` attribute
   representing the point estimate of the tail
-  Every tail estimator how has an ``attachment_age``
   parameter to allow for attachment before the end of the
   triangle
-  ``TailCurve`` now has ``slope_`` and ``intercept_``
   attributes for a diagnostics of the estimator.
-  ``TailBondy`` now has ``earliest_ldf_`` attributes to
   allow for diagnostics of the estimator.
-  Substantial improvement to the `documents <https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#tails>`__ on Tails.
-  Introduced the deterministic components of `ClarkLDF <https://chainladder-python.readthedocs.io/en/latest/modules/generated/chainladder.ClarkLDF.html#chainladder.ClarkLDF>`__ and `TailClark <https://chainladder-python.readthedocs.io/en/latest/modules/generated/chainladder.TailClark.html#chainladder.TailClark>`__ estimators to allow for growth curve selection of development patterns.

Version 0.6
=============

Version 0.6.3
--------------
Release Date: May 21, 2020

**Enhancements (courtesy of gig67)**

-  Added ``Triangle.calendar_correlation`` method and
   companion class ``CalendarCorrelation`` to support
   detecting calendar year correlations in triangles.
-  Added ``Triangle.developmen_correlation`` method and
   companion class ``DevelopmentCorrelation`` to support
   detecting development correlations in triangles.

Version 0.6.2
--------------
Release Date: Apr 27, 2020

patch to 0.6.1

Version 0.6.1
--------------
Release Date: Apr 25, 2020

**Bug Fixes**

-  Corrected a bug where ``TailConstant`` couldn't decay
   when the contant is set to 1.0
-  `#71 <https://github.com/casact/chainladder-python/issues/71>`__
   Fixed issue where
   \``Pipeline.predict\ ``would not honor the``\ sample_weight\`
   argument

**Enhancements**

-  `#72 <https://github.com/casact/chainladder-python/issues/72>`__
   Added ``drop`` method to ``Triangle`` similar to
   ``pd.DataFrame.drop`` for dropping columns
-  Added ``xlcompose`` yaml templating
-  `#74 <https://github.com/casact/chainladder-python/issues/74>`__
   Dropped link ratios now show as ommitted when callinng
   ``link_ratio`` on a ``Development`` transformed triangle
-  `#73 <https://github.com/casact/chainladder-python/issues/73>`__
   ``Triangle.grain`` now has a ``trailing`` argument that
   will aggregate triangle on a trailing basis

Version 0.6.0
--------------
Release Date: Mar 17, 2020

**Enhancements**

-  Added ``TailBondy`` method
-  Propagate ``std_err_`` and ``sigma_`` on determinsitic
   tails in line with Mack for better compatibility with
   ``MackChainladder``
-  Improved consistency between ``to_frame`` and
   ``__repr__`` for 2D triangles.

**Bug Fixes**

-  Fixed a bug where the latest origin period was dropped from ``Triangle`` initialization when sure data was present
-  resolves `#69 <https://github.com/casact/chainladder-python/issues/69>`__ where ``datetime`` was being mishandled when ingested
   into ``Triangle``.
