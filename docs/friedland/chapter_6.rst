================================================================
Chapter 6 - The Development Triangle as a Diagnostic Tool
================================================================

This chapter dives deeper into understanding the triangle. We will demonstrate how to manipulate a ``Triangle`` in the chainladder package by recreating the various tables in this chapter. 

.. doctest::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import chainladder as cl
    >>> tri = cl.load_sample('friedland_xyz_auto_bi')

Table 1 - Summary of Earned Premium and Rate Changes
#######################################################

WIP

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
