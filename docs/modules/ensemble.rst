.. _ensemble:

================
Ensemble Methods
================

Common Properties
=================
Ensemble methods exist to ensemble multiple IBNR estimators into a single method.
Ensemble methods implement the ``ibnr_`` and ``ultimate_`` attributes.

.. _voting:

VotingChainladder
==================
The :class:`VotingChainladder` ensemble method allows the actuary to vote between
different underlying :ref:`_ibnr` by way of a matrix of weights.

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
