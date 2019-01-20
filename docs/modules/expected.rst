.. _expected:

.. currentmodule:: chainladder

===========================
Expected Loss IBNR Methods
===========================

The following are a set of IBNR methods whose structure generally involves
IBNR being a function of an apriori ultimate loss.  In most cases these methods
differ only by how the apriori ultimate loss is selected.

.. _bornferg:

Deterministic Bornhuetter-Ferguson
==================================
The :class:`BornhuetterFerguson` technique is essentially a blend of the
development and expected claims techniques. In the development technique, we multiply actual
claims by a cumulative claim development factor. This technique can lead to erratic, unreliable
projections when the cumulative development factor is large because a relatively small swing in
reported claims or the reporting of an unusually large claim could result in a very large swing in
projected ultimate claims. In the expected claims technique, the unpaid claim estimate is equal to
the difference between a predetermined estimate of expected claims and the actual payments.
This has the advantage of stability, but it completely ignores actual results as reported. The
Bornhuetter-Ferguson technique combines the two techniques by splitting ultimate claims into
two components: actual reported (or paid) claims and expected unreported (or unpaid) claims. As
experience matures, more weight is given to the actual claims and the expected claims become
gradually less important.

.. topic:: References

  .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 9, 2010.

.. _benktander:

Deterministic Benktander
==========================

The :class:`Benktander` method, introduced in 1976, is a credibility-weighted average of the BornhuetterFerguson technique and the development technique. The advantage cited by the authors is that
this method will prove more responsive than the Bornhuetter-Ferguson technique and more stable
than the development technique

.. topic:: References

  .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 9, 2010.


.. _capecod:

Deterministic Cape Cod
==========================

The :class:`CapeCod` method, also known as the Stanard-Buhlmann method, is similar to the
Bornhuetter-Ferguson technique. As in the Bornhuetter-Ferguson technique, the Cape Cod
method splits ultimate claims into two components: actual reported (or paid) and expected
unreported (or unpaid). As an accident year (or other time interval) matures, the actual reported
claims replace the expected unreported claims and the initial expected claims assumption
becomes gradually less important. The primary difference between the two methods is the
derivation of the expected claim ratio. In the Cape Cod technique, the expected claim ratio is
obtained from the reported claims experience instead of an independent and often judgmental
selection as in the Bornhuetter-Ferguson technique

.. topic:: References

  .. [F2010] J.  Friedland, "Estimating Unpaid Claims Using Basic Techniques", Version 3, Ch. 10, 2010.
