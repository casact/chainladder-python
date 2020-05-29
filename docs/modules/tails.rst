.. _tails:

.. currentmodule:: chainladder

===============
Tail Estimators
===============
The Tails module provides a variety of tail transformers that allow for the
extrapolation of development patterns beyond the end of the triangle.  Tail
factors are used extensively in commercial lines of business.

.. _constant:
External Data
=============
:class:`TailConstant` allows you to input a tail factor as a constant.  This is
useful when relying on tail selections from an external source like industry data.

.. warning::
   The chainlader package does not currently support any stochastic
   paramters for :class:`TailConstant`.  This precludes the compatibility of
   this estimator with stochastic methods such as :class:`MackChainladder`.

The tail factor supplied applies to all individual triangles contained within
the Triangle object.  If this is not the desired outcome, slicing individual
triangles and applying :class:`TailConstant` separately to each can be done.

Decay
-----
An exponential ``decay`` parameter is also available to describe the run-off nature of
the tail.  Although tail factors are generally specified as scalar value, the
``chainladder`` package will expand it over a single future calendar year so that
run-off analyses can be performed for the upcoming year.

  >>> import chainladder as cl
  >>> abc = cl.Development().fit_transform(cl.load_sample('abc'))
  >>> abc = cl.TailConstant(tail=1.05, decay=0.95).fit_transform(abc)
  >>> abc.ldf_
            12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132   132-144   144-Ult
  (All)  2.308599  1.421098  1.199934  1.113445  1.072736  1.047559  1.034211  1.026047  1.020188  1.016259  1.002436  1.047448

As we can see in the example, the 5% tail in the example is split between the
amount to run-off over the subsequent calendar period **132-144**, and the
remainder, **144-Ult**.  The split is controlled by the ``decay`` parameter.

.. _curve:
LDF Curve Fitting
=================
:class:`TailCurve` allows for extrapolating a tail factor using curve fitting.
Currently, exponential decay of LDFs and inverse power curve (for slower decay)
are supported.

.. figure:: /auto_examples/images/sphx_glr_plot_tailcurve_compare_001.png
   :target: ../auto_examples/plot_tailcurve_compare.html
   :align: center
   :scale: 50%

In general, ``inverse_power`` fit produce more conservative tail estimates than
the ``exponential`` fit.

Extrapolation Period
---------------------
The ``extrap_periods`` parameter allows for limiting how far beyond the edge of
the triangle the tail should be extrapolated.  Results for the ``inverse_power``
curve are sensitive to this parameter as it tends to converge slowly to its
asymptotic value.

.. figure:: /auto_examples/images/sphx_glr_plot_extrap_period_001.png
   :target: ../auto_examples/plot_extrap_period.html
   :align: center
   :scale: 50%

.. topic:: References

  .. [CAS2013] `CAS Tail Factor Working Party, "The Estimation of Loss Development Tail
      Factors: A Summary Report", Casualty Actuarial Society E-Forum <https://www.casact.org/pubs/forum/13fforum/02-Tail-Factors-Working-Party.pdf>`__

.. _bondy:
The Bondy Tail
==============
:class:`TailBondy` allows for setting the tail factor using the Generalized Bondy
method.  The main estimate of the method is the Bondy exponent.  The tail factor
is can be described as a function of using an ``ldf_`` factor representative of the
last age in the Triangle and the Bondy exponent ``b_``.

More formally, the tail factor is defined as:

.. math::
   F(n)=f(n-1)^{B}f(n-1)^{B^{2}}...=f(n-1)^{\frac{B}{1-B}}

Rather than using the last known development factor explicitely, we estimate the
fitted LDF using the formula above along with the ``earliest_age`` parameter.

  >>> import chainladder as cl
  >>> # Data and initial development patterns
  ... triangle = cl.load_sample('tail_sample')
  >>> dev = cl.Development(average='simple').fit_transform(triangle)
  >>> # Estimate the Bondy Tail
  ... tail = cl.TailBondy(earliest_age=12).fit(dev)
  >>> # Get last fitted LDF of the model
  ... last_fitted_ldf = (tail.earliest_ldf_**(tail.b_**8))
  >>> # Calculate the tail using the Bondy formula above
  ... last_fitted_ldf**(tail.b_/(1-tail.b_))
         incurred     paid
  Total
  Total  1.003982  1.02773
  >>> # Compare to actual tail
  ... tail.tail_
         incurred     paid
  Total
  Total  1.003982  1.02773

.. topic:: References

  .. [CAS2013] `CAS Tail Factor Working Party, "The Estimation of Loss Development Tail
      Factors: A Summary Report", Casualty Actuarial Society E-Forum <https://www.casact.org/pubs/forum/13fforum/02-Tail-Factors-Working-Party.pdf>`__


.. _tailclark:
Growth Curve Extrapolation
==========================
:class:`TailClark` is a continuation of the :class:`ClarkLDF` model.  Familiarity
with :class:`ClarkLDF` will aid in understanding this Estimator.  The growth
curve approach used by Clark produces development patterns for any age including
ages beyond the edge of the Triangle.

An example completing Clark's model:

  >>> import chainladder as cl
  >>> genins = cl.load_sample('genins')
  >>> dev = cl.ClarkLDF()
  >>> tail = cl.TailClark()
  >>> tail.fit(dev.fit_transform(genins)).cdf_
            12-Ult    24-Ult    36-Ult    48-Ult    60-Ult    72-Ult    84-Ult    96-Ult   108-Ult   120-Ult   132-Ult
  (All)  21.086776  5.149415  2.993038  2.229544  1.857163  1.642623  1.505601  1.411711  1.344001  1.293236  1.253992


.. topic:: References

  .. [CD2003] `Clark, David R., "LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood Approach",
               Casualty Actuarial Society Forum, Fall, 2003 <https://www.casact.org/pubs/forum/03fforum/03ff041.pdf>`__
