.. _methods_toc:

============
IBNR Models
============

The IBNR Estimators are the final stage in analyzing reserve estimates in the
``chainladder`` package.  These Estimators have a ``predict`` method as opposed
to a ``transform`` method.


Basics and Commonalities
=========================

Ultimates
----------

All reserving methods determine some ultimate cost of insurance claims.  These
ultimates are captured in the ``ultimate_`` property of the estimator.

  >>> import chainladder as cl
  >>> cl.Chainladder().fit(cl.load_sample('raa')).ultimate_
                2261
  1981  18834.000000
  1982  16857.953917
  1983  24083.370924
  1984  28703.142163
  1985  28926.736343
  1986  19501.103184
  1987  17749.302590
  1988  24019.192510
  1989  16044.984101
  1990  18402.442529

Ultimates are measured at a valuation date way into the future.  The library is
extraordinarily conservative in picking this date, and sets it to December 31, 2261.
This is set globally and can be viewed by referencing the ``ULT_VAL``
constant.

  >>> cl.ULT_VAL
  '2261-12-31 23:59:59.999999999'

.. warning::
   Changing ULT_VAL can make the library exhibit unstable behavior.


The ``ultimate_`` along with most of the other properties of IBNR models are triangles
and can be manipulated.  However, it is important to note that the model itself
is not a Triangle, it is an scikit-learn style Estimator.  This distinction is
important when wanting to manipulate model attributes.

  >>> triangle = cl.load_sample('quarterly')
  >>> model = cl.Chainladder().fit(triangle)
  >>> # This works since we're slicing the ultimate Triangle
  >>> ult = model.ultimate_['paid']
  >>> # This throws an error since the model itself is not sliceable
  >>> ult = model['paid'].ultimate_


IBNR
-----

Any difference between an ``ultimate_`` and the ``latest_diagonal`` of a Triangle
is contained in the ``ibnr_`` property of an estimator.  While technically, as in
the example of a paid triangle, there can be case reserves included in the ``ibnr_``
estimate, the distinction is not made by the ``chainladder`` package and must be
managed by you.

  >>> triangle = cl.load_sample('quarterly')
  >>> model = cl.Chainladder().fit(triangle)
  >>> # Determine outstanding case reserves
  >>> case_reserves = (triangle['incurred']-triangle['paid']).latest_diagonal
  >>> # Net case reserves off of paid IBNR
  >>> true_ibnr = model.ibnr_['paid'] - case_reserves
  >>> true_ibnr.sum()
  2431.2695585474003

Complete Triangles
--------------------
The ``full_triangle_`` and ``full_expectation_`` attributes give a view of the
completed `Triangle`.  While the ``full_expectation_`` is entirely based on
``ultimate_`` values and development patterns, the ``full_triangle_`` is a
blend of the existing triangle.  These are useful for conducting an analysis
of actual results vs model expectations.

  >>> import chainladder as cl
  >>> model = cl.Chainladder().fit(cl.load_sample('ukmotor'))
  >>> residuals = model.full_expectation_ - model.full_triangle_
  >>> residuals
              12          24          36          48        60    72    84    96    9999
  2007  344.492346  557.928307  348.774627   10.847889 -11.40612   NaN   NaN   NaN   NaN
  2008  -21.882151 -185.514153 -340.715515 -102.582899  11.40612   NaN   NaN   NaN   NaN
  2009  -92.224026 -233.617500   94.508419   91.735009       NaN   NaN   NaN   NaN   NaN
  2010 -303.438287 -209.004780 -102.567531         NaN       NaN   NaN   NaN   NaN   NaN
  2011   67.162588   70.208127         NaN         NaN       NaN   NaN   NaN   NaN   NaN
  2012    5.889530         NaN         NaN         NaN       NaN   NaN   NaN   NaN   NaN
  2013         NaN         NaN         NaN         NaN       NaN   NaN   NaN   NaN   NaN


Another typical analysis is to forecast the IBNR run-off for future periods.

  >>> expected_3y_run_off = model.full_triangle_.dev_to_val().cum_to_incr().loc[..., '2014':'2016']
  >>> expected_3y_run_off
               2014         2015         2016
  2007          NaN          NaN          NaN
  2008   350.902024          NaN          NaN
  2009   661.620101   375.916667          NaN
  2010  1073.335187   619.525276   351.999397
  2011  1502.970266  1133.999503   654.540504
  2012  2724.981102  1820.419755  1373.516924
  2013  5587.058983  3351.884601  2239.221748



.. _chainladder_docs:
.. currentmodule:: chainladder

Chainladder
============
The distinguishing characteristic of the :class:`Chainladder` method is that ultimate claims for each
accident year are produced from recorded values assuming that future claims’ development is
similar to prior years’ development. In this method, the actuary uses the development triangles to
track the development history of a specific group of claims. The underlying assumption in the
development technique is that claims recorded to date will continue to develop in a similar manner
in the future – that the past is indicative of the future. That is, the development technique assumes
that the relative change in a given year’s claims from one evaluation point to the next is similar to
the relative change in prior years’ claims at similar evaluation points.

An implicit assumption in the development technique is that, for an immature accident year, the
claims observed thus far tell you something about the claims yet to be observed. This is in
contrast to the assumptions underlying the expected claims technique.

Other important assumptions of the development method include: consistent claim processing, a
stable mix of types of claims, stable policy limits, and stable reinsurance (or excess insurance)
retention limits throughout the experience period.

Though the algorithm underling the basic chainladder is trivial, the properties
of the `Chainladder` estimator allow for a concise access to relevant information.

As an example, we can use the estimator to determine actual vs expected run-off
of a subsequent valuation period.

.. figure:: /auto_examples/images/sphx_glr_plot_ave_analysis_001.png
   :target: ../auto_examples/plot_ave_analysis.html
   :align: center
   :scale: 70%


.. topic:: References

  .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 7, 2010.

.. _mack:

MackChainladder
================

The :class:`MackChainladder` model can be regarded as a special form of a
weighted linear regression through the origin for each development period. By using
a regression framework, statistics about the variability of the data and the parameter
estimates allows for the estimation of prediction errors. The Mack Chainladder
method is the most basic of stochastic methods.


.. figure:: /auto_examples/images/sphx_glr_plot_mack_001.png
   :target: ../auto_examples/plot_mack.html
   :align: center
   :scale: 70%

Compatibility
--------------
Because of the regression framework underlying the `MackChainladder`, it is not
compatible with all development and tail estimators of the library.  In fact,
it really should only be used with the `Development` estimator and `TailCurve`
tail estimator.

.. warning::
   While the MackChainladder might not error with other options for development and
   tail, the stochastic properties should be ignored, in which case the basic
   `Chainladder` should be used.

.. topic:: References

   .. [M1993] T Mack. Distribution-free calculation of the standard error of chain ladder reserve estimates. Astin Bulletin. Vol. 23. No 2. 1993. pp.213:225
   .. [M1994] T Mack. The standard error of chain ladder reserve estimates: Recursive calculation and inclusion of a tail factor. Astin Bulletin. Vol. 29. No 2. 1999. pp.361:366


.. _bornferg:

BornhuetterFerguson
====================
The :class:`BornhuetterFerguson` technique is essentially a blend of the
development and expected claims techniques. In the development technique, we multiply actual
claims by a cumulative claim development factor. This technique can lead to erratic, unreliable
projections when the cumulative development factor is large because a relatively small swing in
reported claims or the reporting of an unusually large claim could result in a very large swing in
projected ultimate claims. In the expected claims technique, the unpaid claim estimate is equal to
the difference between a predetermined estimate of expected claims and the actual payments.
This has the advantage of stability, but it completely ignores actual results as reported. The
Bornhuetter-Ferguson technique combines the two techniques by splitting ultimate claims into
two components: actual reported (or paid) claims and expected unreported (or unpaid) claims. As
experience matures, more weight is given to the actual claims and the expected claims become
gradually less important.

Exposure base
--------------
The :class:`BornhuetterFerguson` technique is the first we explore of the Expected
Loss techniques.  In this family of techniques, we need some measure of exposure.
This is handled by passing a `Triangle` representing the exposure to the ``sample_weight``
argument of the ``fit`` method of the Estimator.

All scikit-learn style estimators optionally support a ``sample_weight`` argument
and this is used by the ``chainladder`` package to capture the exposure base
of these Expected Loss techniques.

  >>> import chainladder as cl
  >>> raa = cl.load_sample('raa')
  >>> sample_weight = raa.latest_diagonal*0+40_000
  >>> cl.BornhuetterFerguson(apriori=0.7).fit(raa, sample_weight=sample_weight).ibnr_.sum()
  75203.23550854485


Apriori
---------
We've fit a :class:`BornhuetterFerguson` model with the assumption that our
prior belief, or ``apriori`` is a 70% Loss Ratio.  The method supports any constant
for the ``apriori`` hyperparameter.  The ``apriori`` then gets
multiplied into our sample weight to determine our prior belief on expected losses
prior to considering that actual emerged to date.

Because of the multiplicative nature of ``apriori`` and ``sample_weight`` we don't
have to limit ourselves to a single constant for the ``apriori``.  Instead, we
can exploit the model structure to make our ``sample_weight`` represent our
prior belief on ultimates while setting the ``apriori`` to 1.0.

For example, we can use the :class:`Chainladder` ultimates as our prior belief
in the :class:`BornhuetterFerguson` method.

  >>> cl_ult = cl.Chainladder().fit(raa).ultimate_ # Chainladder Ultimate
  >>> apriori = cl_ult*0+(cl_ult.sum()/10)[0] # Mean Chainladder Ultimate
  >>> cl.BornhuetterFerguson(apriori=1).fit(raa, sample_weight=apriori).ultimate_
            Ultimate
  1981  18834.000000
  1982  16898.632172
  1983  24012.333266
  1984  28281.843524
  1985  28203.700714
  1986  19840.005163
  1987  18840.362337
  1988  22789.948877
  1989  19541.155136
  1990  20986.022826

.. topic:: References

 .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 9, 2010.

.. _benktander_docs:

Benktander
==========

The :class:`Benktander` method is a credibility-weighted
average of the :class:`BornhuetterFerguson` technique and the development technique.
The advantage cited by the authors is that this method will prove more
responsive than the Bornhuetter-Ferguson technique and more stable
than the development technique.

Iterations
------------
The `Benktander` method is also known as the iterated :class:`BornhuetterFerguson`
method.  This is because it is a generalization of the :class:`BornhuetterFerguson`
technique.

The generalized formula based on ``n_iters``, n is:

.. math::
   Ultimate = Apriori\times (1-\frac{1}{CDF})^{n} + Latest\times \sum_{k=0}^{n-1}(1-\frac{1}{CDF})^{k}

``n=0`` yields the expected loss method, ``n=1`` yields the traditional :class:`BornhuetterFerguson`
method, and finally when ``n`` is sufficiently large, the method converges to the
traditional :class:`Chainladder` method.

.. figure:: /auto_examples/images/sphx_glr_plot_benktander_001.png
   :target: ../auto_examples/plot_benktander.html
   :align: center
   :scale: 70%

Expected Loss Method
---------------------
Setting ``n_iters`` to 0 will emulate that Expected Loss method.  That is to say,
the actual emerged loss experience of the Triangle will be completely ignored in
determining the ultimate.  While it is a trivial calculation, it allows for
run-off patterns to be developed, which is useful for new programs new lines
of businesses.

  >>> import chainladder as cl
  >>> triangle = cl.load_sample('ukmotor')
  >>> exposure = triangle.latest_diagonal*0 + 25_000
  >>> cl.Benktander(apriori=0.75, n_iters=0).fit(triangle, sample_weight=exposure).full_triangle_.round(0)
          12       24       36       48       60       72       84       96       9999
  2007  3511.0   6726.0   8992.0  10704.0  11763.0  12350.0  12690.0  18750.0  18750.0
  2008  4001.0   7703.0   9981.0  11161.0  12117.0  12746.0  18750.0  18750.0  18750.0
  2009  4355.0   8287.0  10233.0  11755.0  12993.0  18248.0  18750.0  18750.0  18750.0
  2010  4295.0   7750.0   9773.0  11093.0  17363.0  18248.0  18750.0  18750.0  18750.0
  2011  4150.0   7897.0  10217.0  15832.0  17363.0  18248.0  18750.0  18750.0  18750.0
  2012  5102.0   9650.0  13801.0  15832.0  17363.0  18248.0  18750.0  18750.0  18750.0
  2013  6283.0  10762.0  13801.0  15832.0  17363.0  18248.0  18750.0  18750.0  18750.0

Mack noted the `Benktander` method is found to have almost always a smaller mean
squared error than the other two methods and to be almost as precise as an exact
Bayesian procedure.

.. topic:: References

 .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 9, 2010.


.. _capecod_docs:

CapeCod
========

The :class:`CapeCod` method, also known as the Stanard-Buhlmann method, is similar to the
Bornhuetter-Ferguson technique.  The primary difference between the two methods is the
derivation of the expected claim ratio. In the Cape Cod technique, the expected claim ratio
or apriori is obtained from the triangle itself instead of an independent and often judgmental
selection as in the Bornhuetter-Ferguson technique.

  >>> import chainladder as cl
  >>> clrd = cl.load_sample('clrd')[['CumPaidLoss', 'EarnedPremDIR']].groupby('LOB').sum().loc['wkcomp']
  >>> loss = clrd['CumPaidLoss']
  >>> sample_weight=clrd['EarnedPremDIR'].latest_diagonal
  >>> m1 = cl.CapeCod().fit(loss, sample_weight=sample_weight)
  >>> m1.ibnr_.sum()
  3030598.3846801124

Apriori
--------
The default hyperparameters for the :class:`CapeCod` method can be emulated by
the :class:`BornhuetterFerguson` method.  We can manually derive the ``apriori``
implicit in the CapeCod estimate.

  >>> cl_ult = cl.Chainladder().fit(loss).ultimate_
  >>> apriori = loss.latest_diagonal.sum()/(sample_weight/(cl_ult/loss.latest_diagonal)).sum()
  >>> m2 = cl.BornhuetterFerguson(apriori).fit(clrd['CumPaidLoss'], sample_weight=clrd['EarnedPremDIR'].latest_diagonal)
  >>> m2.ibnr_.sum()
  3030598.384680113

A parameter `apriori_sigma` can also be specified to give sampling variance to the
estimated apriori.  This along with `random_state` can be used in conjuction with
the `BootstrapODPSample` estimator to build a stochastic `CapeCod` estimate.

Trend and On-level
-------------------
When using data implicit in the Triangle to derive the apriori, it is desirable
to bring the different origin periods to a common basis.  The `CapeCod` estimator
provides a ``trend`` hyperparameter to allow for trending everything to the latest
origin period.  However, the apriori used in the actual estimation of the IBNR is
the ``detrended_apriori_`` detrended back to each of the specific origin periods.

  >>> import pandas as pd
  >>> m1 = cl.CapeCod(trend=0.05).fit(loss, sample_weight=sample_weight)
  >>> pd.concat((
  ...     m1.detrended_apriori_.to_frame().iloc[:, 0].rename('Detrended Apriori'),
  ...     m1.apriori_.to_frame().iloc[:, 0].rename('Apriori')), axis=1)
        Detrended Apriori   Apriori
  1988           0.483539  0.750128
  1989           0.507716  0.750128
  1990           0.533102  0.750128
  1991           0.559757  0.750128
  1992           0.587745  0.750128
  1993           0.617132  0.750128
  1994           0.647989  0.750128
  1995           0.680388  0.750128
  1996           0.714407  0.750128
  1997           0.750128  0.750128

Simple one-part trends are supported directly in the hyperparameter selection.
If a more complex trend assumption is required or on-leveling, then passing
Triangles transformed by the :class:`Trend` and :class:`ParallelogramOLF`
estimators will capture these finer details as in this example from the
example gallery.

.. figure:: /auto_examples/images/sphx_glr_plot_capecod_onlevel_001.png
   :target: ../auto_examples/plot_capecod_onlevel.html
   :align: center
   :scale: 70%

Decay
------
The default behavior of the `CapeCod` is to include all origin periods in the
estimation of the ``apriori_``.  A more localized approach, giving lesser weight
to origin periods that are farther from a target origin period, can be achieved
by flexing the ``decay`` hyperparameter.

>>> cl.CapeCod(decay=0.8).fit(loss, sample_weight=sample_weight).apriori_.T
          1988      1989      1990      1991     1992      1993      1994      1995      1996      1997
2261  0.617945  0.613275  0.604879  0.591887  0.57637  0.559855  0.548615  0.542234  0.540979  0.541723

With a ``decay`` less than 1.0, we see ``apriori_`` estimates that vary by origin.






.. topic:: References

 .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 10, 2010.
