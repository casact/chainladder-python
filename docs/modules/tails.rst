.. _tails:

.. currentmodule:: chainladder

=====
Tails
=====
The Tails module provides a variety of tail transformers that allow for the
extrapolation of development patterns beyond the end of the triangle.  Tail
factors are used extensively in commercial lines of business.

.. _constant:
External Data
=============
:class:`TailConstant` allows you to input a tail factor as a constant.  This is
useful when relying on tail selections from an external source like industry data.

.. _curve:
LDF Curve Fitting
=================
:class:`TailCurve` allows for extrapolating a tail factor using curve fitting.
Currently, Exponential Decay of ldfs and Inverse Power curve (for slower decay)
are supported.

.. topic:: References

  .. [CAS2013] `CAS Tail Factor Working Party, "The Estimation of Loss Development Tail
      Factors: A Summary Report", Casualty Actuarial Society E-Forum <https://www.casact.org/pubs/forum/13fforum/02-Tail-Factors-Working-Party.pdf>`__

.. _bondy:
The Bondy Tail
==============
:class:`TailBondy` allows for setting the tail factor using the Generalized Bondy
method.

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
