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

We need to manually load this table of premium and rate change figures. Note that we are not loading the last two columns, as they can be derived based on premium and rate change. 

.. doctest::

    >>> data = [
    ...     [2002, 61183, 0],
    ...     [2003, 69175, .05],
    ...     [2004, 99322, .075],
    ...     [2005, 138151, .15],
    ...     [2006, 107578, .1],
    ...     [2007, 62438, -.2],
    ...     [2008, 47797, -.2]
    ... ]
    >>> columns = [
    ...     'Calendar Year',
    ...     'Earned Premiums',
    ...     'Rate Changes'
    ... ]
    >>> df_prem = pd.DataFrame(data, columns=columns)
    >>> df_prem['Date'] = pd.to_datetime(df_prem['Calendar Year'].astype(int).astype(str) + '-01-01') # see discussion below on why we are doing this
    >>> df_prem['On-level Factor'] = cl.parallelogram_olf(df_prem['Rate Changes'],df_prem['Date'],vertical_line = True).reset_index()['OLF']
    >>> df_prem['Cumulative Average Rate Level'] = ((1 + df_prem['Rate Changes']).product() / df_prem['On-level Factor'] - 1).round(decimals=3)    
    >>> df_prem['Premium Change'] = df_prem['Earned Premiums'].div(df_prem['Earned Premiums'].shift(1)).dropna()
    >>> df_prem['Annual Exposure Change'] = (df_prem['Premium Change'] / (1 + df_prem['Rate Changes']) - 1).round(decimals=3)
    >>> df_prem[['Calendar Year','Earned Premiums','Rate Changes','Cumulative Average Rate Level','Annual Exposure Change']]

    To simplify the analysis in this chapter and in Part 3, assume that the rate changes in the above table represent the average earned rate level for the year

    -- Friedland, p84

What this assumption means in practice, is that the rate change figures are already on an earned basis. We match this assumption through (1) setting the rate change dates to the beginning of the year, and (2) specifying that vertial_line = True to the utility function ``parallelogram_olf`` (related to, but not to be confused with the estimator ``ParallelogramOLF``). 

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

    >>> tri['Reported Claims'] * 0 + df_prem["Earned Premiums"]
               12       24       36        48        60       72       84
    2002  61183.0  69175.0  99322.0  138151.0  107578.0  62438.0  47797.0
    2003  61183.0  69175.0  99322.0  138151.0  107578.0  62438.0      NaN
    2004  61183.0  69175.0  99322.0  138151.0  107578.0      NaN      NaN
    2005  61183.0  69175.0  99322.0  138151.0       NaN      NaN      NaN
    2006  61183.0  69175.0  99322.0       NaN       NaN      NaN      NaN
    2007  61183.0  69175.0      NaN       NaN       NaN      NaN      NaN
    2008  61183.0      NaN      NaN       NaN       NaN      NaN      NaN

That didn't quite work. We want each accident year to have the same premium. The reason why this is happening is that development period is the last dimension (i.e. down a row in a triangle), and origin period (accident year) is the second-last dimension (i.e. rows down a column). Any 1-D collection is automatically assumed to be values down a row, rather than rows down a column (a little unintuitive, as a ``pandas.DataFrame`` is displayed vertically.). So we need a little help in ``numpy`` land to rectify the problem.

.. doctest::

    >>> prem_tri = tri['Reported Claims'] * 0 + df_prem["Earned Premiums"].to_numpy().reshape(-1,1)
    >>> prem_tri
                12        24        36        48       60       72       84
    2002   61183.0   61183.0   61183.0   61183.0  61183.0  61183.0  61183.0
    2003   69175.0   69175.0   69175.0   69175.0  69175.0  69175.0      NaN
    2004   99322.0   99322.0   99322.0   99322.0  99322.0      NaN      NaN
    2005  138151.0  138151.0  138151.0  138151.0      NaN      NaN      NaN
    2006  107578.0  107578.0  107578.0       NaN      NaN      NaN      NaN
    2007   62438.0   62438.0       NaN       NaN      NaN      NaN      NaN
    2008   47797.0       NaN       NaN       NaN      NaN      NaN      NaN

Now we can divide two triangles seamlessly

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

Table 5 - Ratio of Reported Claims to Earned Premium
#######################################################

    We calculate the on-level premium using the average rate level changes by year and restating the earned premium for each year as if it was written at the 2008 rate level.

    -- Friedland, p84

We don't need to follow Friedland's approach here, as we already got on-level factors. 

.. doctest::

    >>> ol_prem_tri = prem_tri * df_prem["On-level Factor"].to_numpy().reshape(-1,1)
    >>> ol_prem_tri

And the actual Table 5 is straight-forward. 

.. doctest::

    >>> (tri['Reported Claims'] / olprem_tri).round(decimals=3)

Table 6 - Ratio of Paid Claims-to-Reported Claims
#######################################################

.. doctest::

    >>> (tri['Paid Claims'] / tri['Reported Claims']).round(decimals=3)

Table 7 - Ratio of Reported Claims to Earned Premium
#######################################################

.. doctest::

    >>> (tri['Paid Claims'] / olprem_tri).round(decimals=3)

Table 7 - Ratio of Reported Claims to Earned Premium
#######################################################

.. doctest::

    >>> (tri['Paid Claims'] / olprem_tri).round(decimals=3)

Table 8 - Reported Claim Count Development Triangle
#######################################################

The count data is stored under a different name. 

.. doctest::

    >>> tri_cnt = cl.load_sample('friedland_xyz_freq_sev')
    >>> tri_cnt = tri_cnt[tri_cnt.origin >= '2002'][tri_cnt.development <= 84]
    >>> tri_cnt["Reported Claim Counts"]

Table 9 - Closed Claim Count Development Triangle
#######################################################

.. doctest::

    >>> tri_cnt["Closed Claim Counts"]

Table 10 - Ratio of Closed-to-Reported Claim Counts
#######################################################

.. doctest::

    >>> (tri_cnt["Reported Claim Counts"] / tri_cnt["Reported Claim Counts"]).round(decimals=3)

Table 12 – Average Reported Claim Development Triangle
#######################################################

The losses are stored in the thousands. When calcualting severity, we need to multiply back the thousand. 

.. doctest::

    >>> (tri["Reported Claims"] / tri_cnt["Reported Claim Counts"] * 1000).round(decimals=0)

Table 13 – Average Paid Claim Development Triangle
#######################################################

.. doctest::

    >>> (tri["Paid Claims"] / tri_cnt["Closed Claim Counts"] * 1000).round(decimals=0)

Table 14 – Average Case Outstanding Development Triangle
#######################################################

.. doctest::

    >>> ((tri["Reported Claims"] - tri["Paid Claims"]) / (tri_cnt["Reported Claim Counts"] - tri_cnt["Closed Claim Counts"]) * 1000).round(decimals=0)
