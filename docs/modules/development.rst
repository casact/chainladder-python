.. _development_docs:

===========================
Development Estimators
===========================

.. currentmodule:: chainladder

Basics and Commonalities
=========================

Before stepping into fitting development patterns, its worth reviewing the basics
of Estimators. The main modeling API implemented by chainladder follows that of
the scikit-learn estimator. An estimator is any object that learns from data.

Scikit-Learn API
----------------
The scikit-learn API is a common modeling interface that is used to construct and
fit a countless variety of machine learning algorithms.  The common interface
allows for very quick swapping between models with minimal code changes.  The
``chainladder`` package has adopted the interface to promote a standardized approach
to fitting reserving models.

All estimator objects can optionally be configured with parameters to uniquely
specify the model being built.  This is done ahead of pushing any data through
the model.

  >>> estimator = Estimator(param1=1, param2=2)

All estimator objects expose a ``fit`` method that takes a `Triangle` as input, ``X``:

  >>> estimator.fit(X=data)

All estimators include a ``sample_weight`` option to the ``fit`` method to specify
an exposure basis.  If an exposure base is not applicable, then this argument is
ignored.

  >>> estimator.fit(X=data, sample_weight=weight)

All estimators either ``transform`` the input Triangle or ``predict`` an outcome.

Transformers
------------
All transformers include a ``transform`` method.  The method is used to transform a
Triangle and it will always return a Triangle with added features based on the
specifics of the transformer.

  >>> transformed_data = estimator.transform(data)

Other than final IBNR models, ``chainladder`` estimators are transformers.
That is, they return your `Triangle` back to you with additional properties.

Transforming can be done at the time of fit.

  >>> # Fitting and Transforming
  >>> estimator.fit(data)
  >>> transformed_data = estimator.transform(data)
  >>> # One line equivalent
  >>> transformed_data = estimator.fit_transform(data)
  >>> assert isinstance(transformed_data, cl.Triangle)

Predictors
----------
All predictors include a ``predict`` method.

  >>> prediction = estimator.predict(new_data)

Predictors are intended to create new predictions. It is not uncommon to fit a
model on a more aggregate view, say national level, of data and predict on a
more granular triangle, state or provincial.


Parameter Types
---------------
Estimator parameters: All the parameters of an estimator can be set when it is
instantiated or by modifying the corresponding attribute.  These parameters
define how you'd like to fit an estimator and are chosen before the fitting
process.  These are often referred to as hyperparameters in the context of
Machine Learning, and throughout these documents.  Most of the hyperparameters
of the ``chainladder`` package take on sensible defaults.

  >>> estimator = Estimator(param1=1, param2=2)
  >>> estimator.param1
  1

Estimated parameters: When data is fitted with an estimator, parameters are
estimated from the data at hand. All the estimated parameters are attributes
of the estimator object ending by an underscore.  The use of the underscore is
a key API design style of scikit-learn that allows for the quicker recognition
of fitted parameters vs hyperparameters:

  >>> estimator.estimated_param_

In many cases the estimated parameters are themselves Triangles and can be
manipulated using the same methods we learned about in the :class:`Triangle` class.

  >>> dev = cl.Development().fit(cl.load_sample('ukmotor'))
  >>> type(dev.cdf_)
  <class 'chainladder.core.triangle.Triangle'>

Commonalities
--------------

All "Development Estimators" are transformers and reveal common a set of properties
when they are fit.

1. ``ldf_`` represents the fitted age-to-age factors of the model.
2. ``cdf_`` represents the fitted age-to-ultimate factors of the model.
3. All "Development estimators" implement the ``transform`` method.


``cdf_`` is nothing more than the cumulative representation of the ``ldf_`` vectors.

  >>> import chainladder as cl
  >>> dev = cl.Development().fit(cl.load_sample('raa'))
  >>> dev.ldf_.incr_to_cum() == dev.cdf_
  True


.. _dev:

Development
============

:class:`Development` allows for the selection of loss development patterns. Many
of the typical averaging techniques are available in this class: ``simple``,
``volume`` and  ``regression`` through the origin. Additionally, :class:`Development`
includes patterns to allow for fine-tuned exclusion of link-ratios from the LDF
calculation.


Setting parameters
-------------------

Most of the arguments of the ``Development`` class can be specified for all
development ages by providing a single value:

  >>> import chainladder as cl
  >>> raa = cl.load_sample('raa')
  >>> cl.Development(average='simple')
  Development(average='simple', drop=None, drop_high=None, drop_low=None,
              drop_valuation=None, fillna=None, n_periods=-1,
              sigma_interpolation='log-linear')

Alternatively, you can provide a list to parameterize each development period
separately.  When adjusting individual development periods the list must be
the same length as your triangles ``link_ratio`` development axis.

  >>> len(raa.link_ratio.development)
  9
  >>> cl.Development(average=['volume']+['simple']*8)
  Development(average=['volume', 'simple', 'simple', 'simple', 'simple', 'simple',
                       'simple', 'simple', 'simple'],
              drop=None, drop_high=None, drop_low=None, drop_valuation=None,
              fillna=None, n_periods=-1, sigma_interpolation='log-linear')

This approach works for ``average``, ``n_periods``, ``drop_high`` and ``drop_low``.

Notice where you have not specified a parameter, a sensible default
is chosen for you.

.. _dropping:

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

When using ``drop``, the earliest age of the ``link_ratio`` should be referenced.
For example, use ``12`` to drop the ``12-24`` ratio.

.. note::
  ``drop_high`` and ``drop_low`` are ignored in cases where the number of link
  ratios available for a given development period is less than 3.

Properties
----------
:class:`Development` uses the regression approach suggested by Mack to estimate
development patterns.  Using the regression framework, we not only get estimates
for our patterns (``cdf_``, and ``ldf_``), but also measures of variability of
our estimates (``sigma_``, ``std_err_``).  These variability properties are
used to develop the stochastic features in the `MackChainladder` method, but for
deterministic exercises, can be ignored.

Transforming
------------
When transforming a `Triangle`, you will receive a copy of the original
triangle back along with the fitted properties of the `Development`
estimator.  Where the original Triangle contains all link ratios, the transformed
version recognizes any ommissions you specify.

  >>> import chainladder as cl
  >>> triangle = cl.load_sample('raa')
  >>> dev = cl.Development(drop=('1982', 12), drop_valuation='1988')
  >>> transformed_triangle = dev.fit_transform(triangle)
  >>> transformed_triangle.ldf_
            12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120
  (All)  2.662527  1.544686  1.297522  1.171947  1.113358  1.046817  1.029409  1.033088  1.009217
  >>> transformed_triangle.link_ratio.heatmap()


.. figure:: ../_static/images/transformed_heatmap.PNG
   :align: center
   :scale: 40%

By decoupling the ``fit`` and ``transform`` methods, we can apply our :class:`Development`
estimator to new data.  This is a common pattern of the scikit-learn API. In this
example we generate development patterns at an industry level and apply those
patterns to individual companies.

  >>> import chainladder as cl
  >>> clrd = cl.load_sample('clrd')
  >>> clrd = clrd[clrd['LOB']=='wkcomp']['CumPaidLoss']
  >>> # Summarize Triangle to industry level to estimate patterns
  ... dev = cl.Development().fit(clrd.sum())
  >>> # Apply Industry patterns to individual companies
  ... dev.transform(clrd)
  Valuation: 1997-12
  Grain:     OYDY
  Shape:     (132, 1, 10, 10)
  Index:      ['GRNAME', 'LOB']
  Columns:    ['CumPaidLoss']


.. _dev_const:

DevelopmentConstant
===================

External patterns
------------------

The :class:`DevelopmentConstant` estimator simply allows you to hard code development
patterns into a Development Estimator.  A common example would be to include a
set of industry development patterns in your workflow that are not directly
estimated from any of your own data.

  >>> triangle = cl.load_sample('ukmotor')
  >>> patterns={12: 2, 24: 1.25, 36: 1.1, 48: 1.08, 60: 1.05, 72: 1.02}
  >>> cl.DevelopmentConstant(patterns=patterns, style='ldf').fit(triangle).ldf_
      12-24  24-36  36-48  48-60  60-72  72-84
  (All)    2.0   1.25    1.1   1.08   1.05   1.02


By wrapping patterns in the :class:`DevelopmentConstant` estimator, we can integrate
into a larger workflow with tail extrapolation and IBNR calculations.

Function of patterns
--------------------

:class:`DevelopmentConstant` doesn't have to be limited to one set of fixed patterns.
It can take any arbitrary function similar to how ``pandas.DataFrame.apply`` works. Refer
to this detailed example:

.. figure:: /auto_examples/images/sphx_glr_plot_callable_dev_constant_001.png
   :target: ../auto_examples/plot_callable_dev_constant.html
   :align: center
   :scale: 75%


.. _incremental:

IncrementalAdditive
===================

The :class:`IncrementalAdditive` method uses both the triangle of incremental
losses and the exposure vector for each accident year as a base. Incremental
additive ratios are computed by taking the ratio of incremental loss to the
exposure (which has been adjusted for the measurable effect of inflation), for
each accident year. This gives the amount of incremental loss in each year and
at each age expressed as a percentage of exposure, which we then use to square
the incremental triangle.

  >>> import chainladder as cl
  >>> tri = cl.load_sample("ia_sample")
  >>> ia = cl.IncrementalAdditive().fit(
  ...     tri['loss'], sample_weight=tri['exposure'].latest_diagonal)
  >>> ia.incremental_.round(0)
            12      24      36      48     60     72
  2000  1001.0   854.0   568.0   565.0  347.0  148.0
  2001  1113.0   990.0   671.0   648.0  422.0  164.0
  2002  1265.0  1168.0   800.0   744.0  482.0  195.0
  2003  1490.0  1383.0  1007.0   849.0  543.0  220.0
  2004  1725.0  1536.0  1068.0   984.0  629.0  255.0
  2005  1889.0  1811.0  1256.0  1157.0  740.0  300.0

These ``incremental_`` values are then used to determine an implied set of
mutiplicative development patterns.  Because incremental additive values are
unique for each ``origin``, so too will be the ``ldf_``.

  >>> ia.ldf_
           12-24     24-36     36-48     48-60     60-72
  2000  1.853147  1.306199  1.233182  1.116131  1.044378
  2001  1.889488  1.319068  1.233598  1.123320  1.042624
  2002  1.923320  1.328812  1.230127  1.121179  1.043830
  2003  1.928188  1.350505  1.218848  1.114772  1.041751
  2004  1.890435  1.327647  1.227353  1.118406  1.042933
  2005  1.958577  1.339524  1.233506  1.121004  1.043773

Incremental calculation
------------------------
The estimation of the incremental triangle can be done with varying hyperparameters
of ``n_period`` and ``average`` similar to the `Development` estimator.  Additionally,
a ``trend`` in the origin period can also be selected.

Suppose there is a vector ``zeta_`` that represents an estimate of the incremental
losses, ``X`` for a development period as a percentage of some exposure or ``sample_weight``.
Using a 'volume' weighted estimate for all origin periods, we can manually estimate ``zeta_``

  >>> zeta_ = tri['loss'].cum_to_incr().sum('origin') / tri['exposure'].sum('origin')
  >>> zeta_
              12       24        36        48        60       72
  2000  0.243212  0.22196  0.153978  0.141853  0.090673  0.03677

The ``zeta_`` vector along with the ``sample_weight`` and optionally a `trend`
are used to propagate incremental losses to the lower half of the `Triangle`.
In the trivial case of no trend, we can estimate the incrementals for age 72.

  >>> zeta_.loc[..., 72] * tri['exposure'].latest_diagonal
                72
  2000  148.000000
  2001  163.847950
  2002  195.433540
  2003  220.106335
  2004  255.148323
  2005  299.971180
  >>> # These are the same incrementals that the IncrementalAdditive method produces
  >>> zeta_.loc[..., 72]*tri['exposure'].latest_diagonal == ia.incremental_.loc[..., 72]
  True

Trending
---------
The `IncrementalAdditive` method supports trending through the ``trend``
and the ``future_trend`` hyperparameters.  The ``trend`` parameter is used in the
fitting of ``zeta_`` and it trends all inner diagonals of the `Triangle` to its
``latest_diagonal`` before estimating ``zeta_``.

The ``future_trend`` hyperparameter is used to trend beyond the ``latest_diagonal``
into the lower half of the `Triangle`.  If no future trend is supplied, then
the ``future_trend`` is assumed to be that of the ``trend`` parameter.

  >>> cl.IncrementalAdditive(trend=0.02, future_trend=0.05).fit(
  ...     tri['loss'], sample_weight=tri['exposure'].latest_diagonal).incremental_.round(0)
            12      24      36      48     60     72
  2000  1001.0   854.0   568.0   565.0  347.0  148.0
  2001  1113.0   990.0   671.0   648.0  422.0  172.0
  2002  1265.0  1168.0   800.0   744.0  511.0  215.0
  2003  1490.0  1383.0  1007.0   908.0  604.0  255.0
  2004  1725.0  1536.0  1151.0  1105.0  735.0  310.0
  2005  1889.0  1967.0  1420.0  1364.0  907.0  383.0


.. note::
   These trend assumptions are applied to the incremental Triangle which produces
   drastically different answers from the same trends applied to a cumulative Triangle.


A nice property of this estimator is that it really only requires incremental amounts
so a `Triangle` that has cumulative data censored data in earlier diagonals can
leverage this method.  Another nice property is that it allows for more explicit recognition
of future inflation in your estimate via the `trend` factor.

.. topic:: References

  .. [S2006] K Schmidt, "Methods and Models of Loss Reserving Based on Run–Off Triangles: A Unifying Survey"

.. _munich:

MunichAdjustment
=================

The :class:`MunichAdjustment` is a bivariate adjustment to loss development factors.
There is a fundamental correlation between the paid and the case incurred data
of a triangle. The ratio of paid to incurred **(P/I)** has information that can
be used to simultaneously adjust the basic development factor selections for the
two separate triangles.

Depending on whether the momentary **(P/I)** ratio is below or above average,
one should use an above-average or below-average paid development factor and/or
a below-average or above-average incurred development factor.  In doing so, the
model replaces a set of development patterns that would be used for all
``origins`` with individualized development curves that reflect the unique levels
of **(P/I)** per origin period.

Residuals
----------
The :class:`MunichAdjustment` uses the correlation between the residuals of the
univariate (basic) model and the (P/I) model.  These correlations spin off a
property ``lambda_`` which is represented by the line through the origin of
the correlation plots.

.. figure:: /auto_examples/images/sphx_glr_plot_munich_resid_001.png
   :target: ../auto_examples/plot_munich_resid.html
   :align: center
   :scale: 50%

With the correlations, ``lambda_`` known, the basic development patterns can
be adjusted based on the **(P/I)** ratio at any given cell of the ``Triangle``.

BerquistSherman Comparison
---------------------------

This method is similar to the `BerquistSherman` approach in that it tries to
adjust for case reserve adequacy.  However it is different in two distinct ways.

  1.  The `BerquistSherman` method is a direct adjustment to the data whereas
      the `MunichAdjustment` keeps the `Triangle` intact and adjusts the development
      patterns.
  2.  The `MunichAdjustment` is built in the context of a stochastic framework.


.. topic:: References

  .. [QM2004] `G Quarg, Gerhard, and T Mack, "Munich Chain Ladder: A Reserving Method that Reduces the Gap between IBNR" <http://www.variancejournal.org/issues/02-02/266.pdf>`__


.. _clarkldf:

ClarkLDF
=========
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

The ``Loglogistic`` Growth Function:

.. math::
   G(x|\omega, \theta) =\frac{x^{\omega }}{x^{\omega } + \theta^{\omega }}

The ``Weibull`` Growth Function:

.. math::
  G(x|\omega, \theta) =1-exp(-\left (\frac{x}{\theta}  \right )^\omega)

.. figure:: /auto_examples/images/sphx_glr_plot_clarkldf_001.png
   :target: ../auto_examples/plot_clarkldf.html
   :align: center
   :scale: 50%

Parameterized growth curves can produce patterns for any age and can even be used
to estimate a tail beyond the latest age in a Triangle.  In general, the
``loglogistic`` growth curve produces a larger tail than the ``weibull`` growth curve.

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
  >>> dev = cl.ClarkLDF(growth='weibull').fit(clrd['CumPaidLoss'])
  >>> dev.omega_
            CumPaidLoss
  LOB
  comauto      0.928925
  medmal       1.569647
  othliab      1.330080
  ppauto       0.831530
  prodliab     1.456174
  wkcomp       0.898277

Perhaps more useful than the parameters is the growth curve ``G_`` function they
represent which can be used to deetermine the development factor at any age.

  >>> 1/dev.G_(37.5).to_frame()
  LOB
  comauto     1.270909
  medmal      1.707244
  othliab     1.619175
  ppauto      1.118799
  prodliab    2.126134
  wkcomp      1.311591
  dtype: float64


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


CaseOutstanding
================

The :class:`CaseOutstanding` method is a deterministic method that estimates
incremental payment patterns from prior lag carried case reserves.  Included
in this is also patterns for the carried case reserves based on the prior lag
carried case reserve.

Like the :class:`MunichAdjustment` and :class:`BerquistSherman`, this estimator
is useful when you want to incorporate information about case reserves into paid
ultimates.

To use it, a triangle with both paid and incurred amounts must be available.

  >>> import chainladder as cl
  >>> tri = cl.load_sample('usauto')
  >>> model = cl.CaseOutstanding(paid_to_incurred=('paid', 'incurred')).fit(tri)
  >>> model.paid_ldf_
            12-24     24-36    36-48     48-60     60-72     72-84     84-96    96-108   108-120
  (All)  0.842814  0.709981  0.70835  0.696826  0.637589  0.622016  0.553385  0.437373  0.524347

In the example above, the incremental paid losses during the period 12-24 is expected to be
84.28% of the outstanding case reserve at lag 12.  The set of patterns produced by
:class:`CaseOutstanding` don't follow the multiplicative approach commonly used in the
various IBNR methods making them not directly usable.  Because of this, the estimator
determines the 'implied' multiplicative pattern so that a broader set of IBNR
methods can be used.  Due to the origin period specifics on case reserves, each
origin gets its own set of multiplicative ``ldf_`` patterns.

  >>> model.ldf_['paid']
           12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120
  1998  1.792469  1.205560  1.095603  1.045669  1.018931  1.009749  1.004777  1.002288  1.001866
  1999  1.768268  1.198631  1.090150  1.043455  1.019377  1.009244  1.004998  1.002392  1.001904
  2000  1.761959  1.190201  1.089958  1.042975  1.019054  1.010127  1.004585  1.002444  1.001969
  2001  1.743900  1.191331  1.090565  1.043566  1.018677  1.009022  1.004171  1.002074  1.001672
  2002  1.734765  1.194005  1.089153  1.044183  1.018551  1.008452  1.004106  1.002042  1.001646
  2003  1.718935  1.185285  1.091988  1.043790  1.018635  1.009171  1.004452  1.002213  1.001784
  2004  1.702514  1.186716  1.092167  1.041492  1.017863  1.008798  1.004272  1.002124  1.001712
  2005  1.701237  1.186004  1.085897  1.041211  1.017746  1.008741  1.004245  1.002111  1.001701
  2006  1.702795  1.179664  1.085678  1.041114  1.017706  1.008722  1.004236  1.002106  1.001698
  2007  1.669287  1.180366  1.085961  1.041239  1.017758  1.008747  1.004248  1.002112  1.001702

Incremental patterns
---------------------

The incremental patterns of the :class:`CaseOutstanding` method are avilable as
additional properties for review. They are the ``paid_to_prior_case_`` and the
``case_to_prior_case_``. These are useful to review when deciding on the appropriate
hyperparameters for ``paid_n_periods`` and ``case_n_periods``.  Once you are satisfied
with your hyperparameter tuning, you can see the fitted selections in the
``paid_ldf_`` and ``case_ldf_`` incremental patterns.

  >>> model.case_to_prior_case_
           12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120
  1998  0.537820  0.554128  0.525253  0.498107  0.532934  0.537997  0.587702  0.697024  0.579812
  1999  0.536825  0.564892  0.544220  0.496910  0.502864  0.579975  0.641971  0.650552       NaN
  2000  0.546126  0.574211  0.539091  0.487208  0.537598  0.543222  0.665505       NaN       NaN
  2001  0.540564  0.566029  0.514819  0.501324  0.507736  0.541364       NaN       NaN       NaN
  2002  0.540900  0.554610  0.540609  0.480216  0.488133       NaN       NaN       NaN       NaN
  2003  0.526510  0.576514  0.536276  0.476418       NaN       NaN       NaN       NaN       NaN
  2004  0.529775  0.566523  0.506884       NaN       NaN       NaN       NaN       NaN       NaN
  2005  0.521531  0.553888       NaN       NaN       NaN       NaN       NaN       NaN       NaN
  2006  0.526122       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
  2007       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
  >>> model.case_ldf_
            12-24    24-36     36-48     48-60     60-72    72-84     84-96    96-108   108-120
  (All)  0.534019  0.56385  0.529593  0.490031  0.513853  0.55064  0.631726  0.673788  0.579812


.. topic:: References

  .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 12, 2010.

TweedieGLM
================
The `TweedieGLM` implements the GLM reserving structure discussed by Taylor and McGuire.
A nice property of the GLM framework is that it is highly flexible in terms of including
covariates that may be predictive of loss reserves while maintaining a close relationship
to traditional methods.  Additionally, the framework can be extended in a straightforward
way to incorporate various approaches to measuring prediction errors.  Behind the
scenes, `TweedieGLM` is using scikit-learn's ``TweedieRegressor`` estimator.

Long Format
-------------
GLMs are fit to triangles in "Long Format".  That is, they are converted to pandas
DataFrames behind the scenes.  Each axis of the `Triangle` is included in the
dataframe. The ``origin`` and ``development`` axes are in columns of the same name.
You can inspect what your `Triangle` looks like in long format by calling `to_frame`
with ``keepdims=True``


  >>> cl.load_sample('clrd').to_frame(keepdims=True).reset_index().head()
              GRNAME      LOB     origin  development  ...  EarnedPremCeded  EarnedPremDIR  EarnedPremNet  IncurLoss
  0  Adriatic Ins Co  othliab 1995-01-01           12  ...            131.0          139.0            8.0        8.0
  1  Adriatic Ins Co  othliab 1995-01-01           24  ...            131.0          139.0            8.0       11.0
  2  Adriatic Ins Co  othliab 1995-01-01           36  ...            131.0          139.0            8.0        7.0
  3  Adriatic Ins Co  othliab 1996-01-01           12  ...            359.0          410.0           51.0       40.0
  4  Adriatic Ins Co  othliab 1996-01-01           24  ...            359.0          410.0           51.0       40.0

  [5 rows x 10 columns]

.. warning::
  'origin', 'development', and 'valuation' are reserved keywords for the dataframe.  Declaring columns with these names separately will result in error.

While you can inspect the `Triangle` in long format, you will not directly convert
to long format yourself.  The `TweedieGLM` does this for you.  Additionally,
the `origin` of the design matrix is restated in years from the earliest origin
period.  That is, is if the earliest origin is '1995-01-01' then it gets replaced with
0.  Consequently, '1996-04-01' would be replaced with 1.25. This is done because
datetimes have limited support in scikit-learn. Finally, the `TweedieGLM`
will automatically convert the response to an incremental basis.

R-style formulas
-----------------
We use the ``patsy`` library to allow formulation of the the feature set ``X``
of the GLM.  Because ``X`` is a parameter that used extensively throughout
``chainladder``, the `TweedieGLM` refers to it as the ``design_matrix``.  Those
familiar with the R programming language will be familiar with the notation
used by ``patsy``.  For example, we can include both ``origin`` and ``development``
as terms in a model.

  >>> glm = cl.TweedieGLM(design_matrix='development + origin').fit(genins)
  >>> glm.coef_
                   coef_
  Intercept    13.516322
  development  -0.006251
  origin        0.033863


ODP Chainladder
-----------------
Replicating the results of the volume weighted chainladder development patterns
can be done by fitting a Poisson-log GLM to incremental paids.  To do this, we
can specify the ``power`` and ``link`` of the estimator as well as the ``design_matrix``.
The volume-weighted chainladder method can be replicated by including both
``origin`` and ``development`` as categorical features.

  >>> genins = cl.load_sample('genins')
  >>> dev = cl.TweedieGLM(
  ...     design_matrix='C(development) + C(origin)',
  ...     power=1, link='log'
  ...     ).fit(genins)

A trivial comparison against the traditional `Development` estimator shows
a comparable set of ``ldf_`` patterns.

.. figure:: /auto_examples/images/sphx_glr_plot_glm_ldf_001.png
   :target: ../auto_examples/plot_glm_ldf.html
   :align: center
   :scale: 75%

Parsimonious modeling
-----------------------
Having full access to all axes of the `Triangle` along with the powerful formulation
of ``patsy`` allows for substantial customization of the model fit.  For example,
we can include 'LOB' interactions with piecewise linear coefficients to reduce
model complexity.

  >>> clrd = cl.load_sample('clrd')['CumPaidLoss'].groupby('LOB').sum()
  >>> clrd=clrd[clrd['LOB'].isin(['ppauto', 'comauto'])]
  >>> dev = cl.TweedieGLM(
  ...     design_matrix='LOB+LOB:C(np.minimum(development, 36))+LOB:development+LOB:origin',
  ...     max_iter=1000).fit(clrd)
  >>> dev.coef_
                                                         coef_
  Intercept                                          12.549945
  LOB[T.ppauto]                                       3.202703
  LOB[comauto]:C(np.minimum(development, 36))[T.24]   0.578694
  LOB[ppauto]:C(np.minimum(development, 36))[T.24]    0.449832
  LOB[comauto]:C(np.minimum(development, 36))[T.36]   0.790516
  LOB[ppauto]:C(np.minimum(development, 36))[T.36]    0.321206
  LOB[comauto]:development                           -0.044627
  LOB[ppauto]:development                            -0.054814
  LOB[comauto]:origin                                 0.054581
  LOB[ppauto]:origin                                  0.057790

This model is limited to 10 coefficients across two lines of business.  The basic
chainladder model is known to be overparameterized with at least 18 parameters
requiring estimation. Despite drastically simplifying the model, the ``cdf_``
patterns of the GLM are within 1% of the traditional chainladder for every lag
and for both lines of business:

  >>> ((dev.cdf_.iloc[..., 0, :]/cl.Development().fit(clrd).cdf_)-1).to_frame().round(3)
  development  12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult
  LOB
  comauto       0.002   0.003   -0.01   0.003   0.011   0.008   0.005  -0.000   -0.002
  ppauto        0.006   0.003   -0.00   0.001   0.002   0.001   0.001   0.001    0.001


Like every other Development estimator, the `TweedieGLM` produces a set of ``ldf_``
patterns and can be used in a larger workflow with tail extrapolation and reserve
estimation.


.. topic:: References

  .. [T2016] Taylor, G. and McGuire G., "Stochastic Loss Reserving Using Generalized Linear Models", CAS Monograph #3

DevelopmentML
==============
`DevelopmentML` is a general development estimator that works as an interface to
scikit-learn compliant machine learning (ML) estimators.  The `TweedieGLM` is
a special case of `DevelopmentML` with the ML algorithm limited to scikit-learn's
``TweedieRegressor`` estimator.

The Interface
--------------
ML algorithms are designed to be fit against tabular data like a pandas DataFrame
or a 2D numpy array.  A `Triangle` does not meet the definition and so ``DevelopmentML``
is provided to incorporate ML into a broader reserving workflow. This includes:

  1. Automatic conversion of Triangle to a dataframe for fitting
  2. Flexibility in expressing any preprocessing as part of a scikit-learn ``Pipeline``
  3. Predictions through the terminal development age of a `Triangle` to fill in the
     lower half
  4. Predictions converted to ``ldf_`` patterns so that the results of the estimator
     are compliant with the rest of ``chainladder``, like tail selection and IBNR modeling.

Features
---------
Data from any axis of a `Triangle` is available to be used in the `DevelopmentML`
estimator.  For example, we can use many of the scikit-learn components to
generate development patterns from both the time axes as well as the ``index`` of
the `Triangle`.

  >>> import chainladder as cl
  >>> from sklearn.ensemble import RandomForestRegressor
  >>> from sklearn.pipeline import Pipeline
  >>> from sklearn.preprocessing import OneHotEncoder
  >>> from sklearn.compose import ColumnTransformer
  >>> clrd = cl.load_sample('clrd').groupby('LOB').sum()['CumPaidLoss']
  >>> # Decide how to preprocess the X (ML) dataset using sklearn
  >>> design_matrix = ColumnTransformer(transformers=[
  ...     ('dummy', OneHotEncoder(drop='first'), ['LOB', 'development']),
  ...     ('passthrough', 'passthrough', ['origin'])
  ... ])
  >>> # Wrap preprocessing and model in a larger sklearn Pipeline
  >>> estimator_ml = Pipeline(steps=[
  ...     ('design_matrix', design_matrix),
  ...     ('model', RandomForestRegressor())
  ... ])
  >>> # Fitting DevelopmentML fits the underlying ML model and gives access to ldf_
  >>> cl.DevelopmentML(estimator_ml=estimator_ml, y_ml='CumPaidLoss').fit(clrd).ldf_
             Triangle Summary
  Valuation:          2261-12
  Grain:                 OYDY
  Shape:        (6, 1, 10, 9)
  Index:                [LOB]
  Columns:      [CumPaidLoss]


Autoregressive
---------------
The time-series nature of loss development naturally lends to an urge for autoregressive
features.  That is, features that are based on predictions, albeit on a lagged basis.
`DevelopmentML` includes an ``autoregressive`` parameter that can be used to
express the response as a lagged feature as well.

.. note::
   When using ``autoregressive`` features, you must also declare it as a column
   in your ``estimator_ml`` Pipeline.

PatsyFormula
-------------
While the sklearn preprocessing API is powerful, it can be tedious work with in
some instances. In particular, modeling complex interactions is much easier to do
with Patsy.  The  ``chainladder`` package includes a custom sklearn estimator
to gain access to the patsy API.  This is done through the `PatsyFormula` estimator.

  >>> estimator_ml = Pipeline(steps=[
  ...     ('design_matrix', cl.PatsyFormula('LOB:C(origin)+LOB:C(development)+development')),
  ...     ('model', RandomForestRegressor())
  ... ])
  >>> cl.DevelopmentML(estimator_ml=estimator_ml, y_ml='CumPaidLoss').fit(clrd).ldf_.iloc[0, 0, 0].round(2)
         12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
  (All)   2.64   1.42   1.19   1.09   1.04   1.02   1.01    1.01     1.01


  .. note::
     `PatsyFormula` is not an estimator designed to work with triangles.  It is an sklearn
     transformer designed to work with pandas DataFrames allowing it to work directly
     in an sklearn Pipeline.
