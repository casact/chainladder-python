.. _workflow:

.. currentmodule:: chainladder

==========
Workflow
==========
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


.. _pipeline:

Pipeline
========
The :class:`Pipeline` class implements utilities to build a composite
estimator, as a chain of transforms and estimators.  Said differently, a
`Pipeline` is a way to wrap multiple estimators into a single compact object.
The `Pipeline` is borrowed from scikit-learn.  As an example of compactness,
we can simulate a set of triangles using bootstrap sampling, apply volume-weigted
development, exponential tail curve fitting, and get the 95%-ile IBNR estimate.

  >>> import chainladder as cl
  >>> steps=[
  ...     ('sample', cl.BootstrapODPSample(random_state=42)),
  ...     ('dev', cl.Development(average='volume')),
  ...     ('tail', cl.TailCurve('exponential')),
  ...     ('model', cl.Chainladder())]
  >>> pipe = cl.Pipeline(steps=steps)
  >>> pipe.fit(cl.load_sample('genins'))
  >>> pipe.named_steps.model.ibnr_.sum('origin').quantile(.95)
             values
  NaN  2.606327e+07

Each estimator contained within a pipelines ``steps`` can be accessed by name
using the ``named_steps`` attribute of the `Pipeline`.

Chainladder Persistence
========================

The `Pipeline` along with all estimators can be persisted to disk or database
using ``to_json`` or ``to_pickle``.  Restoring the pipeline is as simple as
``cl.read_json`` or ``cl.read_pickle``.

  >>> # Persisting the pipe to JSON
  >>> pipe_json = pipe.to_json()
  >>> pipe_json
  '[{"name": "sample", "params": {"drop": null, "drop_high": null, "drop_low": null, "drop_valuation": null, "hat_adj": true, "n_periods": -1, "n_sims": 1000, "random_state": 42}, "__class__": "BootstrapODPSample"}, {"name": "dev", "params": {"average": "volume", "drop": null, "drop_high": null, "drop_low": null, "drop_valuation": null, "fillna": null, "n_periods": -1, "sigma_interpolation": "log-linear"}, "__class__": "Development"}, {"name": "tail", "params": {"attachment_age": null, "curve": "exponential", "errors": "ignore", "extrap_periods": 100, "fit_period": [null, null]}, "__class__": "TailCurve"}, {"name": "model", "params": {}, "__class__": "Chainladder"}]'
  >>> # Rehydrating the pipeline from JSON.
  >>> cl.read_json(pipe_json)
  Pipeline(memory=None,
           steps=[('sample',
                   BootstrapODPSample(drop=None, drop_high=None, drop_low=None,
                                      drop_valuation=None, hat_adj=True,
                                      n_periods=-1, n_sims=1000,
                                      random_state=42)),
                  ('dev',
                   Development(average='volume', drop=None, drop_high=None,
                               drop_low=None, drop_valuation=None, fillna=None,
                               n_periods=-1, sigma_interpolation='log-linear')),
                  ('tail',
                   TailCurve(attachment_age=None, curve='exponential',
                             errors='ignore', extrap_periods=100,
                             fit_period=[None, None])),
                  ('model', Chainladder())],
           verbose=False)

The saved Estimator does not retain any fitted attributes, nor does it retain
the data on which it was fit.  It is simply the model definition.  However,
the Triangle itself can also be saved allowing for a full rehydration of the
original model.

  >>> # Dumping triangle to JSON
  >>> triangle_json = cl.load_sample('genins').to_json()
  >>> # Recalling model and Triangle and rehydrating the results
  >>> cl.read_json(pipe_json).fit(cl.read_json(triangle_json)) \
  ...                        .named_steps.model.ibnr_.sum('origin').quantile(.95)
             values
  NaN  2.606327e+07


.. _gridsearch:

GridSearch
==========
The grid search provided by :class:`GridSearch` exhaustively generates
candidates from a grid of parameter values specified with the ``param_grid``
parameter.  Like `Pipeline`, `Gridsearch` borrows from its scikit-learn counterpart
`GridSearchCV`.

Because reserving techniques are different from supervised machine learning,
`Gridsearch` does not try to pick optimal hyperparameters for you. It is more of
a scenario-testing estimator.

`GridSearch` can be applied to all other estimators, including the `Pipeline`
estimator.  To use it, one must specify a ``param_grid`` as well as a ``scoring``
function which defines the estimator property(s) you wish to capture.  If capturing
multiple properties is desired, multiple scoring functions can be created and
stored in a dictionary.

Here we capture multiple properties of the `TailBondy` estimator using the
`GridSearch` routine to test the sensitivity of the model to changing hyperparameters.

.. figure:: /auto_examples/images/sphx_glr_plot_bondy_sensitivity_001.png
   :target: ../auto_examples/plot_bondy_sensitivity.html
   :align: center
   :scale: 70%

Using `GridSearch` for scenario testing is entirely optional.  You can write
your own looping mechanisms to achieve the same result.  For example:

.. figure:: /auto_examples/images/sphx_glr_plot_capecod_001.png
   :target: ../auto_examples/plot_capecod.html
   :align: center
   :scale: 50%
