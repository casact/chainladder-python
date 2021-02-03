.. _adjustments:

.. currentmodule:: chainladder

=================
Data Adjustments
=================
``chainladder`` facilitates practical reserving workflows.


.. _bootstrap:

Bootstrap Sampling
==================

:class:`BootstrapODPSample` is a transformer that simulates new triangles
according to the ODP Bootstrap model.  That is both the ``index`` and ``column``
of the Triangle must be of unity length.  Upon fitting the Estimator, the
``index`` will contain the individual simulations.

  >>> import chainladder as cl
  >>> raa = cl.load_sample('raa')
  >>> cl.BootstrapODPSample(n_sims=500).fit_transform(raa)
  Valuation: 1990-12
  Grain:     OYDY
  Shape:     (500, 1, 10, 10)
  Index:      ['Total']
  Columns:    ['values']

.. note::
   The `BootstrapODPSample` can only apply to single triangles as it needs the
   ``index`` axis to be free to hold the different triangle simluations.

The class only simulates new triangles from which you can generate
statistics about parameter and process uncertainty.  This allows for converting
the various deterministic :ref:`IBNR Models<methods_toc>` into stochastic
methods.

An example of using the :class:`BootstrapODPSample` with the :class:`BornhuetterFerguson`
method:

.. figure:: /auto_examples/images/sphx_glr_plot_stochastic_bornferg_001.png
   :target: ../auto_examples/plot_stochastic_bornferg.html
   :align: center
   :scale: 70%

Like the `Development` estimators, The `BootstrapODPSample` allows for ommission
of certain residuals from its sampling algorithm with a suite of "dropping"
parameters.  See :ref:`Omitting Link Ratios<dropping>`.

.. topic:: References

  .. [SM2016] `M Shapland, "Using the ODP Bootstrap Model: A Practitioner's Guide", CAS Monograph No.4 <https://www.casact.org/pubs/monographs/papers/04-shapland.pdf>`__


.. _berqsherm:

Berquist Sherman
================
:class:`BerquistSherman` provides a mechanism of restating the inner diagonals of a
triangle for changes in claims practices.  These adjustments can materialize in
case incurred and paid amounts as well as closed claims count development.

In all cases, the adjustments retain the unadjusted latest diagonal of the
triangle.  For the Incurred adjustment, an assumption of the trend rate in
average open case reserves must be supplied.  For the adjustments to paid
amounts and closed claim counts, an estimator, such as `Chainladder` is needed
to calulate ultimate reported count so that the ``disposal_rate_`` of the
model can be calculated.

`BerquistSherman` is strictly a data adjustment to the `Triangle` and it does
not attempt to estimate development patterns, tails, or ultimate values.

.. figure:: /auto_examples/images/sphx_glr_plot_berqsherm_closure_001.png
   :target: ../auto_examples/plot_berqsherm_closure.html
   :align: center
   :scale: 50%

.. topic:: References

  .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 13, 2010.


.. _parallelogramolf:

ParallelogramOLF
=================

The :class:`ParallelogramOLF` estimator is used to on-level a Triangle using
the parallogram technique.  It requires a "rate history" and supports both
vertical line estimates as well as the more common effective date estimates.
This estimator can be used within other estimators that depend on on-leveling,
such as the :class:`CapeCod` method.

.. figure:: /auto_examples/images/sphx_glr_plot_capecod_onlevel_001.png
   :target: ../auto_examples/plot_capecod_onlevel.html
   :align: center
   :scale: 70%


.. _trend:

Trend
======

The :class:`Trend` estimator is a convenience estimator that allows for compound
trends to be used in other estimators that have a ``trend`` assumption.  This
enables more complex trend assumptions to be used.

**Example:**
  >>> import chainladder as cl
  >>> ppauto_loss = cl.load_sample('clrd').groupby('LOB').sum().loc['ppauto', 'CumPaidLoss']
  >>> ppauto_prem = cl.load_sample('clrd').groupby('LOB').sum() \
  ...                 .loc['ppauto']['EarnedPremDIR'].latest_diagonal
  >>> # Simple trend
  >>> a = cl.CapeCod(trend=0.05).fit(ppauto_loss, sample_weight=ppauto_prem).ultimate_.sum()
  >>> # Equivalent using a Trend Estimator. This allows us to convert to more complex trends
  >>> b = cl.CapeCod().fit(cl.Trend(.05).fit_transform(ppauto_loss), sample_weight=ppauto_prem).ultimate_.sum()
  >>> a == b
  True
