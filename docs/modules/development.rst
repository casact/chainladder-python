.. _development:

=========================
Loss Development Patterns
=========================


.. currentmodule:: chainladder


.. _dev:
Basic Development
==================

:class:`Development` allows for the selection of loss development patterns.


.. _incremental:
Incremental Additive
====================

The :class:`IncrementalAdditive` method uses both the triangle of incremental losses and the exposure vector
for each accident year as a base. Incremental additive ratios are computed by taking the ratio of
incremental loss to the exposure (which has been adjusted for the measurable effect of inflation), for
each accident year. This gives the amount of incremental loss in each year and at each age expressed as a
percentage of exposure, which we then use to square the triangle.

.. topic:: References

  .. [BS2010] T Boles and A Staudt, "On the Accuracy of Loss Reserving Methodology", CAS E-forum Fall 2010

.. _munich:
Munich Adjustment
==================

:class:`MunichAdjustment` allows for correcting both paid and incurred development
patterns for the ratio of paid-to-incurred.

.. topic:: References

  .. [QM2004] G Quarg, Gerhard, and T Mack, "Munich Chain Ladder: A Reserving Method that Reduces the Gap between IBNR"
