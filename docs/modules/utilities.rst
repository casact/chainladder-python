.. _utils:

.. currentmodule:: chainladder

=========
Utilities
=========
Utilities contains example datasets and extra functionality to facilitate a
reserving workflow.

.. _samples:

Sample Datasets
===============
A variety of datasets can be loaded using :func:`load_sample()`.  These are
sample datasets that are used in a variety of examples within this
documentation.

========= =======================================================
Dataset   Description
========= =======================================================
abc       ABC Data
auto      Auto Data
berqsherm Data from the Berquist Sherman paper
cc_sample Sample Insurance Data for Cape Cod Method in Struhuss
clrd      CAS Loss Reserving Database
genins    General Insurance Data used in Clark
ia_sample Sample data for Incremental Additive Method in Schmidt
liab      more data
m3ir5     more data
mcl       Sample insurance data for Munich Adjustment in Quarg
mortgage  more data
mw2008    more data
mw2014    more data
quarterly Sample data to demonstrate changing Triangle grain
raa       Sample data used in Mack Chainladder
ukmotor   more data
usaa      more data
usauto    more data
========= =======================================================


Chainladder Persistence
========================

All estimators can be persisted to disk or database
using ``to_json`` or ``to_pickle``.  Restoring the estimator is as simple as
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
