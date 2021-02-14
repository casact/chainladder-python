.. _workflow:

.. currentmodule:: chainladder

==========
Workflow
==========
``chainladder`` facilitates practical reserving workflows.


.. _pipeline_docs:

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


.. _voting:

VotingChainladder
==================
The :class:`VotingChainladder` ensemble method allows the actuary to vote between
different underlying ``ibnr_`` by way of a matrix of weights.

For example, the actuary may choose the :class:`Chainladder` method for the first
4 origin periods, the :class:`BornhuetterFerguson` method for the next 3 origin periods,
and the :class:`CapeCod` method for the final 3.

   >>> raa = cl.load_sample('RAA')
   >>> cl_ult = cl.Chainladder().fit(raa).ultimate_ # Chainladder Ultimate
   >>> apriori = cl_ult*0+(cl_ult.sum()/10) # Mean Chainladder Ultimate
   >>>
   >>> bcl = cl.Chainladder()
   >>> bf = cl.BornhuetterFerguson()
   >>> cc = cl.CapeCod()
   >>>
   >>> estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]
   >>> weights = np.array([[1, 0, 0]] * 4 + [[0, 1, 0]] * 3 + [[0, 0, 1]] * 3)
   >>>
   >>> vot = cl.VotingChainladder(estimators=estimators, weights=weights)
   >>> vot.fit(raa, sample_weight=apriori)
   >>> vot.ultimate_
                 2262
   1981  18834.000000
   1982  16857.953917
   1983  24083.370924
   1984  28703.142163
   1985  28203.700714
   1986  19840.005163
   1987  18840.362337
   1988  23106.943030
   1989  20004.502125
   1990  21605.832631

Alternatively, the actuary may choose to combine all methods using weights. Omitting
the weights parameter results in the average of all predictions assuming a weight of 1.

   >>> raa = cl.load_sample('RAA')
   >>> cl_ult = cl.Chainladder().fit(raa).ultimate_ # Chainladder Ultimate
   >>> apriori = cl_ult * 0 + (float(cl_ult.sum()) / 10) # Mean Chainladder Ultimate
   >>>
   >>> bcl = cl.Chainladder()
   >>> bf = cl.BornhuetterFerguson()
   >>> cc = cl.CapeCod()
   >>>
   >>> estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]
   >>>
   >>> vot = cl.VotingChainladder(estimators=estimators)
   >>> vot.fit(raa, sample_weight=apriori)
   >>> vot.ultimate_
                 2262
   1981  18834.000000
   1982  16887.197765
   1983  24041.977401
   1984  28435.540175
   1985  28466.807537
   1986  19770.579236
   1987  18547.931167
   1988  23305.361472
   1989  18530.213787
   1990  20331.432662


.. _gridsearch_docs:

GridSearch
==========
The grid search provided by :class:`GridSearch` exhaustively generates
candidates from a grid of parameter values specified with the ``param_grid``
parameter.  Like `Pipeline`, `GridSearch` borrows from its scikit-learn counterpart
``GridSearchCV``.

Because reserving techniques are different from supervised machine learning,
`GridSearch` does not try to pick optimal hyperparameters for you. It is more of
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
