.. _ibnr:

============
IBNR Models
============

Common Properties
=================
All IBNR estimators have ``ibnr_``, ``ultimate_``, ``full_triangle_`` and
``full_expectation_`` attributes.  In addition, they carry over the transformed
triangle as ``X_`` along with all of its properties.  Finally, the following
estimators implement the ``predict`` method which allows them to be used on
different Triangles.

.. _chainladder:
.. currentmodule:: chainladder

Basic Chainladder
==================
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

Mack Chainladder
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


.. topic:: References

   .. [M1993] T Mack. Distribution-free calculation of the standard error of chain ladder reserve estimates. Astin Bulletin. Vol. 23. No 2. 1993. pp.213:225
   .. [M1994] T Mack. The standard error of chain ladder reserve estimates: Recursive calculation and inclusion of a tail factor. Astin Bulletin. Vol. 29. No 2. 1999. pp.361:366


.. _bornferg:

Bornhuetter-Ferguson
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


Smoothing chainladder ultimates by using them as apriori figures in the
Bornhuetter Ferguson method.

  >>> raa = cl.load_sample('RAA')
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

.. _benktander:

Benktander
==========

The :class:`Benktander` method, introduced in 1976, is a credibility-weighted
average of the ``BornhuetterFerguson`` technique and the development technique.
The advantage cited by the authors is that this method will prove more
responsive than the Bornhuetter-Ferguson technique and more stable
than the development technique. It is also known as the iterated BF method.
The generalized formula is:

.. math::
   \sum_{k=0}^{n-1}(1-\frac{1}{CDF}) + Apriori\times (1-\frac{1}{CDF})^{n}

``n=0`` yields the expected loss method, ``n=1`` yields the traditional BF method,
and finally when ``n`` is sufficiently large, the method converges to the
traditional CL method.

.. figure:: /auto_examples/images/sphx_glr_plot_benktander_001.png
   :target: ../auto_examples/plot_benktander.html
   :align: center
   :scale: 70%


Mack noted the ``Benktander`` method is found to have almost always a smaller mean
squared error than the other two methods and to be almost as precise as an exact
Bayesian procedure.

.. topic:: References

 .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 9, 2010.


.. _capecod:

Cape Cod
========

The :class:`CapeCod` method, also known as the Stanard-Buhlmann method, is similar to the
Bornhuetter-Ferguson technique. As in the Bornhuetter-Ferguson technique, the Cape Cod
method splits ultimate claims into two components: actual reported (or paid) and expected
unreported (or unpaid). As an accident year (or other time interval) matures, the actual reported
claims replace the expected unreported claims and the initial expected claims assumption
becomes gradually less important. The primary difference between the two methods is the
derivation of the expected claim ratio. In the Cape Cod technique, the expected claim ratio is
obtained from the reported claims experience instead of an independent and often judgmental
selection as in the Bornhuetter-Ferguson technique

.. topic:: References

 .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 10, 2010.
