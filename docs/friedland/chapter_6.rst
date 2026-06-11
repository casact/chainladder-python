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
       Calendar Year  Earned Premiums  Rate Changes  Cumulative Average Rate Level  Annual Exposure Change
    0           2002            61183         0.000                          0.000                     NaN
    1           2003            69175         0.050                          0.050                   0.077
    2           2004            99322         0.075                          0.129                   0.336
    3           2005           138151         0.150                          0.298                   0.210
    4           2006           107578         0.100                          0.428                  -0.292
    5           2007            62438        -0.200                          0.142                  -0.275
    6           2008            47797        -0.200                         -0.086                  -0.043

We take a different approach from Friedland to calculate the on-level factors in order to leverage the functionality available in the chainladder package. This approach is more direct since we are almost always after on-leveled premium. We are calculating Cumulative Average Rate Level merely to demonstrate parity with the text. 

    To simplify the analysis in this chapter and in Part 3, assume that the rate changes in the above table represent the average earned rate level for the year

    -- Friedland, p84

What this assumption means in practice, is that the rate change figures are already on an earned basis. This tells us to do these two things as we call the utility function ``parallelogram_olf``

* setting the rate change dates to the beginning of the year
* specifying that ``vertial_line = True``

Note that this utility function is related to, but not to be confused with, the estimator ``ParallelogramOLF``. 

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

Table 5 - Ratio of Reported Claims to On-Level Earned Premium
################################################################

    We calculate the on-level premium using the average rate level changes by year and restating the earned premium for each year as if it was written at the 2008 rate level.

    -- Friedland, p84

We don't need to follow Friedland's approach here, as we already got on-level factors. 

.. doctest::

    >>> ol_prem_tri = prem_tri * df_prem["On-level Factor"].to_numpy().reshape(-1,1)
    >>> ol_prem_tri
                    12            24            36            48            60            72            84
    2002  55911.227988  55911.227988  55911.227988  55911.227988  55911.227988  55911.227988  55911.227988
    2003  60204.386000  60204.386000  60204.386000  60204.386000  60204.386000  60204.386000           NaN
    2004  80411.091200  80411.091200  80411.091200  80411.091200  80411.091200           NaN           NaN
    2005  97258.304000  97258.304000  97258.304000  97258.304000           NaN           NaN           NaN
    2006  68849.920000  68849.920000  68849.920000           NaN           NaN           NaN           NaN
    2007  49950.400000  49950.400000           NaN           NaN           NaN           NaN           NaN
    2008  47797.000000           NaN           NaN           NaN           NaN           NaN           NaN

And the actual Table 5 is straight-forward. 

.. doctest::

    >>> (tri['Reported Claims'] / ol_prem_tri).round(decimals=3)
             12     24     36     48     60     72     84
    2002  0.229  0.364  0.477  0.674  0.794  0.871  0.862
    2003  0.160  0.282  0.504  0.674  0.735  0.737    NaN
    2004  0.211  0.500  0.732  0.892  0.874    NaN    NaN
    2005  0.295  0.488  0.723  0.726    NaN    NaN    NaN
    2006  0.393  0.679  0.709    NaN    NaN    NaN    NaN
    2007  0.390  0.635    NaN    NaN    NaN    NaN    NaN
    2008  0.390    NaN    NaN    NaN    NaN    NaN    NaN

Table 6 - Ratio of Paid Claims-to-Reported Claims
#######################################################

.. doctest::

    >>> (tri['Paid Claims'] / tri['Reported Claims']).round(decimals=3)
             12     24     36     48     60     72     84
    2002  0.181  0.389  0.519  0.587  0.719  0.834  0.923
    2003  0.181  0.367  0.418  0.564  0.780  0.886    NaN
    2004  0.131  0.246  0.441  0.606  0.751    NaN    NaN
    2005  0.106  0.258  0.385  0.566    NaN    NaN    NaN
    2006  0.130  0.252  0.468    NaN    NaN    NaN    NaN
    2007  0.181  0.374    NaN    NaN    NaN    NaN    NaN
    2008  0.183    NaN    NaN    NaN    NaN    NaN    NaN

Table 7 - Ratio of Paid Claims to Earned Premium
#######################################################

.. doctest::

    >>> (tri['Paid Claims'] / ol_prem_tri).round(decimals=3)
             12     24     36     48     60     72     84
    2002  0.041  0.142  0.247  0.395  0.571  0.727  0.795
    2003  0.029  0.104  0.211  0.380  0.573  0.653    NaN
    2004  0.028  0.123  0.323  0.540  0.657    NaN    NaN
    2005  0.031  0.126  0.278  0.412    NaN    NaN    NaN
    2006  0.051  0.171  0.331    NaN    NaN    NaN    NaN
    2007  0.071  0.238    NaN    NaN    NaN    NaN    NaN
    2008  0.071    NaN    NaN    NaN    NaN    NaN    NaN

Table 8 - Reported Claim Count Development Triangle
#######################################################

.. doctest::

    >>> tri["Reported Claim Counts"]
              12      24      36      48      60      72      84
    2002  1342.0  1514.0  1548.0  1557.0  1549.0  1552.0  1554.0
    2003  1373.0  1616.0  1630.0  1626.0  1629.0  1629.0     NaN
    2004  1932.0  2168.0  2234.0  2249.0  2258.0     NaN     NaN
    2005  2067.0  2293.0  2367.0  2390.0     NaN     NaN     NaN
    2006  1473.0  1645.0  1657.0     NaN     NaN     NaN     NaN
    2007  1192.0  1264.0     NaN     NaN     NaN     NaN     NaN
    2008  1036.0     NaN     NaN     NaN     NaN     NaN     NaN

Table 9 - Closed Claim Count Development Triangle
#######################################################

.. doctest::

    >>> tri["Closed Claim Counts"]
             12      24      36      48      60      72      84
    2002  203.0   607.0   841.0  1089.0  1327.0  1464.0  1523.0
    2003  181.0   614.0   941.0  1263.0  1507.0  1568.0     NaN
    2004  235.0   848.0  1442.0  1852.0  2029.0     NaN     NaN
    2005  295.0  1119.0  1664.0  1946.0     NaN     NaN     NaN
    2006  307.0   906.0  1201.0     NaN     NaN     NaN     NaN
    2007  329.0   791.0     NaN     NaN     NaN     NaN     NaN
    2008  276.0     NaN     NaN     NaN     NaN     NaN     NaN

Table 10 - Ratio of Closed-to-Reported Claim Counts
#######################################################

.. doctest::

    >>> (tri["Closed Claim Counts"] / tri["Reported Claim Counts"]).round(decimals=3)
             12     24     36     48     60     72    84
    2002  0.151  0.401  0.543  0.699  0.857  0.943  0.98
    2003  0.132  0.380  0.577  0.777  0.925  0.963   NaN
    2004  0.122  0.391  0.645  0.823  0.899    NaN   NaN
    2005  0.143  0.488  0.703  0.814    NaN    NaN   NaN
    2006  0.208  0.551  0.725    NaN    NaN    NaN   NaN
    2007  0.276  0.626    NaN    NaN    NaN    NaN   NaN
    2008  0.266    NaN    NaN    NaN    NaN    NaN   NaN

Table 12 – Average Reported Claim Development Triangle
#######################################################

The losses are stored in the thousands. When calcualting severity, we need to multiply back the thousand. 

.. doctest::

    >>> (tri["Reported Claims"] / tri["Reported Claim Counts"] * 1000).round(decimals=0)
               12       24       36       48       60       72       84
    2002   9546.0  13454.0  17220.0  24192.0  28673.0  31380.0  30997.0
    2003   7029.0  10517.0  18622.0  24966.0  27152.0  27239.0      NaN
    2004   8797.0  18533.0  26350.0  31884.0  31128.0      NaN      NaN
    2005  13872.0  20686.0  29717.0  29563.0      NaN      NaN      NaN
    2006  18375.0  28440.0  29453.0      NaN      NaN      NaN      NaN
    2007  16340.0  25104.0      NaN      NaN      NaN      NaN      NaN
    2008  17985.0      NaN      NaN      NaN      NaN      NaN      NaN

We see some very slight differences with the table in the text, likely due to rounding (the losses were rounded to the nearest thousand). 

Table 13 – Average Paid Claim Development Triangle
#######################################################

.. doctest::

    >>> (tri["Paid Claims"] / tri["Closed Claim Counts"] * 1000).round(decimals=0)
               12       24       36       48       60       72       84
    2002  11419.0  13068.0  16435.0  20289.0  24073.0  27752.0  29177.0
    2003   9630.0  10163.0  13478.0  18125.0  22896.0  25077.0      NaN
    2004   9451.0  11672.0  17996.0  23455.0  26028.0      NaN      NaN
    2005  10315.0  10920.0  16270.0  20568.0      NaN      NaN      NaN
    2006  11502.0  13000.0  19000.0      NaN      NaN      NaN      NaN
    2007  10726.0  15000.0      NaN      NaN      NaN      NaN      NaN
    2008  12351.0      NaN      NaN      NaN      NaN      NaN      NaN

Table 14 – Average Case Outstanding Development Triangle
#########################################################

.. doctest::

    >>> ((tri["Reported Claims"] - tri["Paid Claims"]) / (tri["Reported Claim Counts"] - tri["Closed Claim Counts"]) * 1000).round(decimals=0)
               12       24       36       48       60       72        84
    2002   9212.0  13713.0  18153.0  33274.0  56167.0  91727.0  120387.0
    2003   6634.0  10734.0  25647.0  48766.0  79721.0  82836.0       NaN
    2004   8706.0  22941.0  41561.0  71204.0  76319.0      NaN       NaN
    2005  14464.0  29994.0  61546.0  68984.0      NaN      NaN       NaN
    2006  20184.0  47368.0  56985.0      NaN      NaN      NaN       NaN
    2007  18480.0  42002.0      NaN      NaN      NaN      NaN       NaN
    2008  20030.0      NaN      NaN      NaN      NaN      NaN       NaN
