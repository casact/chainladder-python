.. _tails:

.. currentmodule:: chainladder

=====
Tails
=====
The Tails module provides a variety of tail transformers that allow for the
extrapolation of development patterns beyond the end of the triangle.  Tail
factors are used extensively in commercial lines of business.

.. _constant:
Constant
========
:class:`TailConstant` allows you to input a tail factor as a constant.  This is
useful when relying on tail selections from an external source like industry data.

.. _curve:
Curve Fit
=========
:class:`TailCurve` allows for extrapolating a tail factor using curve fitting.
Currently, Exponential Decay of ldfs and Inverse Power curve (for slower decay)
are supported.

.. _bondy:
Bondy
=====
:class:`TailBondy` allows for setting the tail factor using the Generalized Bondy
method.
