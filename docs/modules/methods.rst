.. _ibnr:

============
IBNR Models
============

.. _chainladder:
.. currentmodule:: chainladder

Deterministic Chainladder
=========================
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

.. topic:: References

  .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 7, 2010.

.. _mack:

Mack Chainladder
================

The :class:`MackChainladder` model can be regarded as a special form of a
weighted linear regression through the origin for each development period. By using
a regression framework, statistics about the variability of the data and the parameter
estimates allows for the estimation of prediciton errors.  The Mack Chainladder
method is the most basic of stochastic methods.

.. topic:: References

   .. [M1993] T Mack. Distribution-free calculation of the standard error of chain ladder reserve estimates. Astin Bulletin. Vol. 23. No 2. 1993. pp.213:225
   .. [M1994] T Mack. The standard error of chain ladder reserve estimates: Recursive calculation and inclusion of a tail factor. Astin Bulletin. Vol. 29. No 2. 1999. pp.361:366


.. _bornferg:

Deterministic Bornhuetter-Ferguson
==================================
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

.. topic:: References

 .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 9, 2010.

.. _benktander:

Deterministic Benktander
==========================

The :class:`Benktander` method, introduced in 1976, is a credibility-weighted
average of the ``BornhuetterFerguson`` technique and the development technique.
 The advantage cited by the authors is that this method will prove more
responsive than the Bornhuetter-Ferguson technique and more stable
than the development technique. It is also known as the interated BF method.
The generalized formula is:

.. math::
   \sum_{k=0}^{n-1}(1-\frac{1}{CDF}) + Apriori\times (1-\frac{1}{CDF})^{n}

`n=1` yields the traditional BF method, and when `n` is sufficiently large, the
method converges to the traditional CL method.

Mack noted the ``Benktander`` method is found to have almost always a smaller mean
squared error than the other two methods and to be almost as precise as an exact
Bayesian procedure.

.. topic:: References

 .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 9, 2010.


.. _capecod:

Deterministic Cape Cod
==========================

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
