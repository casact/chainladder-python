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

  .. [S2006] K Schmidt, "Methods and Models of Loss Reserving Based on Run–Off Triangles: A Unifying Survey"

.. _munich:

Munich Adjustment
==================

:class:`MunichAdjustment` combines the paid (P) and incurred (I) data types by taking the
(P/I) ratio into account in its projections.

.. topic:: References

  .. [QM2004] G Quarg, Gerhard, and T Mack, "Munich Chain Ladder: A Reserving Method that Reduces the Gap between IBNR"


.. _bootstrap:

Bootstrap Over-dispersed Poisson Sample
========================================

:class:`BootstrapODPSample` simulates new triangles according to the ODP Bootstrap
model. The class only simulates new triangles from which you can generate
statistics about parameter uncertainty. Estimates of ultimate along with process
uncertainty would occur with the various :ref:`IBNR Models<methods_toc>`.

.. topic:: References

  .. [SM2016] M Shapland, "Using the ODP Bootstrap Model: A Practitioner's Guide", CAS Monograph No.4
