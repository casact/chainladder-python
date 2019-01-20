.. _multiplicative:


===========================
Multiplicative IBNR Methods
===========================

The multiplicative methods are of the most frequently used methodologies for
estimating unpaid claims.


.. _chainladder:
.. currentmodule:: chainladder

Deterministic Chainladder
-------------------------
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
----------------

The :class:`MackChainladder` model can be regarded as a special form of a
weighted linear regression through the origin for each development period. By using
a regression framework, statistics about the variability of the data and the parameter
estimates allows for the estimation of prediciton errors.  The Mack Chainladder
method is the most basic of stochastic methods.

.. topic:: References

   .. [M1993] T Mack. Distribution-free calculation of the standard error of chain ladder reserve estimates. Astin Bulletin. Vol. 23. No 2. 1993. pp.213:225
   .. [M1994] T Mack. The standard error of chain ladder reserve estimates: Recursive calculation and inclusion of a tail factor. Astin Bulletin. Vol. 29. No 2. 1999. pp.361:366
