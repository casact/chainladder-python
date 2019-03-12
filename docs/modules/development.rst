.. _development:

=========================
Loss Development Patterns
=========================


.. currentmodule:: chainladder


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
   >>> raa = cl.load_dataset('raa')
   >>> cl.Development(average=['volume']+['simple']*8).fit(raa)

This approach works for ``average``, ``n_periods``, ``drop_high`` and ``drop_low``.

Omitting link ratios
--------------------
There are several arguments for dropping individual cells from the triangle as
well as excluding whole valuation periods or highs and lows.  Any combination
of the 'drop' arguments is permissible.

**Example:**
   >>> import chainladder as cl
   >>> raa = cl.load_dataset('raa')
   >>> cl.Development(drop_high=True, drop_low=True).fit(raa)
   >>> cl.Development(drop_valuation='1985').fit(raa)
   >>> cl.Development(drop=[('1985', 12), ('1987', 24)]).fit(raa)
   >>> cl.Development(drop=('1985', 12), drop_valuation='1988').fit(raa)

.. note::
  ``drop_high`` and ``drop_low`` are ignored in cases where the number of link
  ratios available for a given development period is less than 3.

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
