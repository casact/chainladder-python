================================================================
Chapter 6 - The Development Triangle as a Diagnostic Tool
================================================================

This chapter dives deeper into understanding the triangle. We will demonstrate how to manipulate a ``Triangle`` in the chainladder package by recreating the various tables in this chapter. 

.. doctest::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import chainladder as cl
    >>> pd.set_option('display.max_columns', None)
    >>> pd.set_option('display.width', 1000)
    >>> tri = cl.load_sample('friedland_xyz_auto_bi')

Table 1 - Summary of Earned Premium and Rate Changes
#######################################################

.. doctest::

    >>> data = [
    ...     [2002, 61183, None ,0 , None],
    ...     [2003, 69175, .05, '5.0%', '7.7%'],
    ...     [2004, 99322, .075, '12.9%', '33.6%'],
    ...     [2005, 138151, .15, '29.8%', '21.0%'],
    ...     [2006, 107578, .1, '42.8%', '-29.2%'],
    ...     [2007, 62438, .2, '14.2%', '-27.5%'],
    ...     [2008, 47797, .2, '-8.6%', '-4.3%']
    ... ]
    >>> columns = [
    ...     'Calendar Year',
    ...     'Earned Premiums',
    ...     'Rate Changes',
    ...     'Cumulative Average Rate Level',
    ...     'Annual Exposure Change'
    ... ]
    >>> df = pd.DataFrame(data, columns=columns)
    >>> with pd.option_context('display.width', 1000):
    ...     print(df)

Table 2 - Reported Claim Development Triangle
##################################################

We don't need the full triangle for this chapter. Here is how to filter a triangle based on accident year and development age. 

.. doctest::

    >>> tri = tri[tri.origin >= '2002'][tri.development <= 84]
    >>> tri['Reported Claims']
               12       24       36       48       60       72       84
    2002  12811.0  20370.0  26656.0  37667.0  44414.0  48701.0  48169.0
    2003   9651.0  16995.0  30354.0  40594.0  44231.0  44373.0      NaN
    2004  16995.0  40180.0  58866.0  71707.0  70288.0      NaN      NaN
    2005  28674.0  47432.0  70340.0  70655.0      NaN      NaN      NaN
    2006  27066.0  46783.0  48804.0      NaN      NaN      NaN      NaN
    2007  19477.0  31732.0      NaN      NaN      NaN      NaN      NaN
    2008  18632.0      NaN      NaN      NaN      NaN      NaN      NaN

Table 3 - Paid Claim Development Triangle
##################################################

.. doctest::

    >>> tri['Paid Claims']
              12       24       36       48       60       72       84
    2002  2318.0   7932.0  13822.0  22095.0  31945.0  40629.0  44437.0
    2003  1743.0   6240.0  12683.0  22892.0  34505.0  39320.0      NaN
    2004  2221.0   9898.0  25950.0  43439.0  52811.0      NaN      NaN
    2005  3043.0  12219.0  27073.0  40026.0      NaN      NaN      NaN
    2006  3531.0  11778.0  22819.0      NaN      NaN      NaN      NaN
    2007  3529.0  11865.0      NaN      NaN      NaN      NaN      NaN
    2008  3409.0      NaN      NaN      NaN      NaN      NaN      NaN

Table 4 - Ratio of Reported Claims to Earned Premium
#######################################################

To divide losses by premium, we need to turn premium from a ``Series`` into a ``Triangle``. Here is a nifty trick to do that. 

.. doctest::

    >>> tri['Reported Claims'] * 0 + df["Earned Premiums"]
               12       24       36        48        60       72       84
    2002  61183.0  69175.0  99322.0  138151.0  107578.0  62438.0  47797.0
    2003  61183.0  69175.0  99322.0  138151.0  107578.0  62438.0      NaN
    2004  61183.0  69175.0  99322.0  138151.0  107578.0      NaN      NaN
    2005  61183.0  69175.0  99322.0  138151.0       NaN      NaN      NaN
    2006  61183.0  69175.0  99322.0       NaN       NaN      NaN      NaN
    2007  61183.0  69175.0      NaN       NaN       NaN      NaN      NaN
    2008  61183.0      NaN      NaN       NaN       NaN      NaN      NaN

That didn't quite work. We want each accident year to have the same premium. The reason why this is happening is that development period is the last dimension (i.e. down a row in a triangle), and origin period (accident year) is the second-last dimension (i.e. rows down a column). Any 1-D collection is automatically assumed to be values down a row, rather than rows down a column (a little unintuitive, as a ``pandas.DataFrame`` is displayed vertically.). So we need a little help in ``numpy`` land to rectify the problem.

    >>> prem_tri = tri['Reported Claims'] * 0 + df["Earned Premiums"].to_numpy().reshape(-1,1) # accident year is the second-order dimension in a Triangle
    >>> prem_tri
                12        24        36        48       60       72       84
    2002   61183.0   61183.0   61183.0   61183.0  61183.0  61183.0  61183.0
    2003   69175.0   69175.0   69175.0   69175.0  69175.0  69175.0      NaN
    2004   99322.0   99322.0   99322.0   99322.0  99322.0      NaN      NaN
    2005  138151.0  138151.0  138151.0  138151.0      NaN      NaN      NaN
    2006  107578.0  107578.0  107578.0       NaN      NaN      NaN      NaN
    2007   62438.0   62438.0       NaN       NaN      NaN      NaN      NaN
    2008   47797.0       NaN       NaN       NaN      NaN      NaN      NaN

Now we can divide seamlessly into our loss triangles

.. doctest::

    >>> (tri['Reported Claims'] / prem_tri).round(decimals=3)
             12     24     36     48     60     72     84
    2002  0.209  0.333  0.436  0.616  0.726  0.796  0.787
    2003  0.140  0.246  0.439  0.587  0.639  0.641    NaN
    2004  0.171  0.405  0.593  0.722  0.708    NaN    NaN
    2005  0.208  0.343  0.509  0.511    NaN    NaN    NaN
    2006  0.252  0.435  0.454    NaN    NaN    NaN    NaN
    2007  0.312  0.508    NaN    NaN    NaN    NaN    NaN
    2008  0.390    NaN    NaN    NaN    NaN    NaN    NaN
