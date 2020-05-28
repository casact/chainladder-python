.. _development:

===========================
Development Estimators
===========================


.. currentmodule:: chainladder

Estimator Basics
================

Fitting data: the main modeling API implemented by chainladder follows that of
the scikit-learn estimator. An estimator is any object that learns from data.

All estimator objects expose a fit method that takes a `Triangle()` object:

  >>> estimator.fit(data)

Estimator parameters: All the parameters of an estimator can be set when it is
instantiated or by modifying the corresponding attribute:

  >>> estimator = Estimator(param1=1, param2=2)
  >>> estimator.param1
  1

Estimated parameters: When data is fitted with an estimator, parameters are
estimated from the data at hand. All the estimated parameters are attributes
of the estimator object ending by an underscore:

  >>> estimator.estimated_param_

In many cases the estimated paramaters are themselves triangles and can be
manipulated using the same methods we learned about in the :class:`Triangle` class.

  >>> dev = cl.Development().fit(cl.load_sample('ukmotor'))
  >>> type(dev.cdf_)
  <class 'chainladder.core.triangle.Triangle'>

.. _dev:

Basic Development
==================

:class:`Development` allows for the selection of loss development patterns. Many
of the typical averaging techniques are available in this class. As well as the
ability to exclude certain patterns from the LDF calculation.

Single Development Adjustment vs Entire Triangle adjustment
-----------------------------------------------------------

Most of the arguments of the ``Development`` class can be specified for each
development period separately.  When adjusting individual development periods
a list is required that defines the argument for each development.

**Example:**
   >>> import chainladder as cl
   >>> raa = cl.load_sample('raa')
   >>> cl.Development(average=['volume']+['simple']*8).fit(raa)

This approach works for ``average``, ``n_periods``, ``drop_high`` and ``drop_low``.

Omitting link ratios
--------------------
There are several arguments for dropping individual cells from the triangle as
well as excluding whole valuation periods or highs and lows.  Any combination
of the 'drop' arguments is permissible.

**Example:**
   >>> import chainladder as cl
   >>> raa = cl.load_sample('raa')
   >>> cl.Development(drop_high=True, drop_low=True).fit(raa)
   >>> cl.Development(drop_valuation='1985').fit(raa)
   >>> cl.Development(drop=[('1985', 12), ('1987', 24)]).fit(raa)
   >>> cl.Development(drop=('1985', 12), drop_valuation='1988').fit(raa)

.. note::
  ``drop_high`` and ``drop_low`` are ignored in cases where the number of link
  ratios available for a given development period is less than 3.

Properties
----------
:class:`Development` uses the regression approach suggested by Mack to estimate
development patterns.  Using the regression framework, we not only get estimates
for our patterns (``cdf_``, and ``ldf_``), but also measures of variability of
our estimates (``sigma_``, ``std_err_``).  These variability propeperties are
used to develop the stochastic featuers in the `MackChainladder()` method.


.. _dev_const:

External patterns
=================

The :class:`DevelopmentConstant` method simply allows you to hard code development
patterns into a Development Estimator.  A common example would be to include a
set of industry development patterns in your workflow that are not directly
estimated from any of your own data.

For more info refer to the docstring of:class:`DevelopmentConstant`.


.. _incremental:

Incremental Additive
====================

The :class:`IncrementalAdditive` method uses both the triangle of incremental
losses and the exposure vector for each accident year as a base. Incremental
additive ratios are computed by taking the ratio of incremental loss to the
exposure (which has been adjusted for the measurable effect of inflation), for
each accident year. This gives the amount of incremental loss in each year and
at each age expressed as a percentage of exposure, which we then use to square
the triangle.

.. topic:: References

  .. [S2006] K Schmidt, "Methods and Models of Loss Reserving Based on Run–Off Triangles: A Unifying Survey"

.. _munich:

Munich Adjustment
==================
The :class:`MunichAdjustment` is a bivariate adjustment to loss development factors.
There is a fundamental correlation between the paid and the case incurred data
**(P/I)** of a triangle. The ratio of paid to incurred has information that can
be used to simultaneously adjust the basic development factor selections for the
two separate triangles.

Depending on whether the momentary **(P/I)** ratio is below or above average,
one should use an above-average or below-average paid development factor and/or
a below-average or above-average incurred development factor.  In doing so, the
model replaces a set of development patterns that would be used for all
`origins` with individualized development curves that reflect the unique levels
of **(P/I)** per origin period.

The :class:`MunichAdjustment` uses the correlation between the residuals of the
univariate (basic) model and the (P/I) model.

.. figure:: /auto_examples/images/sphx_glr_plot_munich_resid_001.png
   :target: ../auto_examples/plot_munich_resid.html
   :align: center
   :scale: 50%

With the correlations, ``lambda_`` known, the basic development patterns can
be adjusted based on the **(P/I)** ratio at any given cell of the ``Triangle``.

.. topic:: References

  .. [QM2004] `G Quarg, Gerhard, and T Mack, "Munich Chain Ladder: A Reserving Method that Reduces the Gap between IBNR" <http://www.variancejournal.org/issues/02-02/266.pdf>`__


.. _bootstrap:

Bootstrap Sampling
==================

:class:`BootstrapODPSample` simulates new triangles according to the ODP Bootstrap
model. The Estimator can only apply to single triangles.  That is both the
``index`` and ``column`` of the Triangle must be of unity length.  Upon fitting the
Estimator, the ``index`` will contain the individual simulations.

  >>> import chainladder as cl
  >>> raa = cl.load_sample('raa')
  >>> cl.BootstrapODPSample(n_sims=500).fit_transform(raa)
  Valuation: 1990-12
  Grain:     OYDY
  Shape:     (500, 1, 10, 10)
  Index:      ['Total']
  Columns:    ['values']

The class only simulates new triangles from which you can generate
statistics about parameter uncertainty. Estimates of ultimate along with process
uncertainty would occur with the various :ref:`IBNR Models<methods_toc>`.

An example of using the :class:`BootstrapODPSample` with the :class:`BornhuetterFerguson`
method:

.. figure:: /auto_examples/images/sphx_glr_plot_stochastic_bornferg_001.png
   :target: ../auto_examples/plot_stochastic_bornferg.html
   :align: center
   :scale: 50%


.. topic:: References

  .. [SM2016] `M Shapland, "Using the ODP Bootstrap Model: A Practitioner's Guide", CAS Monograph No.4 <https://www.casact.org/pubs/monographs/papers/04-shapland.pdf>`__


LDF Curve Fitting
=================
:class:`ClarkLDF` is an application of Maximum Likelihood Estimation (MLE) theory
for modeling the distribution of loss development. This model is used to estimate
future loss emergence, and the variability around that estimate. The value of
using an exposure base to supplement the data in a development triangle is
demonstrated as a means of reducing variability.

Growth curves
-------------
:class:`ClarkLDF` estimates growth curves of the form 'loglogistic' or 'weibull'
for the incremental loss development of a `Triangle`.  These growth curves are
monotonic increasing and are more relevant for paid data.  While the model can
be used for case incurred data, if there is too much "negative" development,
other Estimators should be used.

The `Loglogistic` Growth Function:

.. math::
   G(x|\omega, \theta) =\frac{x^{\omega }}{x^{\omega } + \theta^{\omega }}

The `Weibull` Growth Function:

.. math::
  G(x|\omega, \theta) =1-exp(-\left (\frac{x}{\theta}  \right )^\omega)

.. figure:: /auto_examples/images/sphx_glr_plot_clarkldf_001.png
   :target: ../auto_examples/plot_clarkldf.html
   :align: center
   :scale: 50%

Parameterized growth curves can produce patterns for any age and can even be used
to estimate a tail beyond the latest age in a Triangle.  In general, the
`loglogistic` growth curve produces a larger tail than the `weibull` growth curve.

LDF and Cape Cod methods
------------------------
Clark approaches curve fitting with two different methods, an LDF approach and
a Cape Cod approach.  The LDF approach only requires a loss triangle whereas
the Cape Cod approach would also need a premium vector.  Choosing between the
two methods occurs at the time you fit the estimator.  When a premium vector
is included, the Cape Cod method is invoked.

A simple example of using :class:`ClarkLDF` LDF Method. Upon fitting the Estimator,
we obtain both ``omega_`` and ``theta_``.

  >>> import chainladder as cl
  >>> clrd = cl.load_sample('clrd').groupby('LOB').sum()
  >>> cl.ClarkLDF(growth='weibull').fit(clrd['CumPaidLoss']).omega_
            CumPaidLoss
  LOB
  comauto      0.928921
  medmal       1.569647
  othliab      1.330084
  ppauto       0.831528
  prodliab     1.456167
  wkcomp       0.898282

Another example showing the usage of the :class:`ClarkLDF` Cape Cod approach. With
the Cape Cod, an Expected Loss Ratio is included as an extra feature in the
``elr_`` property.

  >>> cl.ClarkLDF().fit(clrd['CumPaidLoss'], sample_weight=clrd['EarnedPremDIR'].latest_diagonal).elr_
            CumPaidLoss
  LOB
  comauto      0.680297
  medmal       0.701447
  othliab      0.623781
  ppauto       0.825931
  prodliab     0.671010
  wkcomp       0.697926

Residuals
---------
Clark's model assumes Incremental losses are independent and identically
distributed.  To ensure compatibility with this assumption, he suggests
reviewing the "Normalized Residuals" of the fitted incremental losses to ensure
the assumption is not violated.

.. figure:: /auto_examples/images/sphx_glr_plot_clarkldf_resid_001.png
   :target: ../auto_examples/plot_clarkldf_resid.html
   :align: center
   :scale: 50%


Stochastics
-----------
Using MLE to solve for the growth curves, we can produce statistics about the
parameter and process uncertainty of our model.


.. topic:: References

  .. [CD2003] `Clark, David R., "LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood Approach",
               Casualty Actuarial Society Forum, Fall, 2003 <https://www.casact.org/pubs/forum/03fforum/03ff041.pdf>`__
