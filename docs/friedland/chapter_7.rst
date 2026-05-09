================================================================
Chapter 7
================================================================

Development Technique
##########################

    The development technique, also known as the chain ladder technique, is one of the most frequently used methodologies for estimating unpaid claims.

    -- Friedland, p84

This chapter covers the foundational development/chainladder method. In the chainladder package, this is implemented in the ``Development`` class. 

.. doctest::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import chainladder as cl

p106
==================

Diving straight into Exhibit 1. We will begin by importing packages and loading the triangle at the top of p106.

PART 1 - Data Triangle
-----------------------

.. doctest::

    >>> tri = cl.load_sample('friedland_us_industry_auto')
    >>> tri['Reported Claims'].round(decimals = 0)
                 12          24          36          48          60          72          84          96          108         120
    1998  37017487.0  43169009.0  45568919.0  46784558.0  47337318.0  47533264.0  47634419.0  47689655.0  47724678.0  47742304.0
    1999  38954484.0  46045718.0  48882924.0  50219672.0  50729292.0  50926779.0  51069285.0  51163540.0  51185767.0         NaN
    2000  41155776.0  49371478.0  52358476.0  53780322.0  54303086.0  54582950.0  54742188.0  54837929.0         NaN         NaN
    2001  42394069.0  50584112.0  53704296.0  55150118.0  55895583.0  56156727.0  56299562.0         NaN         NaN         NaN
    2002  44755243.0  52971643.0  56102312.0  57703851.0  58363564.0  58592712.0         NaN         NaN         NaN         NaN
    2003  45163102.0  52497731.0  55468551.0  57015411.0  57565344.0         NaN         NaN         NaN         NaN         NaN
    2004  45417309.0  52640322.0  55553673.0  56976657.0         NaN         NaN         NaN         NaN         NaN         NaN
    2005  46360869.0  53790061.0  56786410.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN
    2006  46582684.0  54641339.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN
    2007  48853563.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN

To calculate the triangle of age-to-age factors, use the age-to-age property of the triangle. 

PART 2 - Age-to-Age Factors
----------------------------

.. doctest::
    
    >>> tri['Reported Claims'].age_to_age.round(decimals = 3) 
          12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    1998  1.166  1.056  1.027  1.012  1.004  1.002  1.001   1.001      1.0
    1999  1.182  1.062  1.027  1.010  1.004  1.003  1.002   1.000      NaN
    2000  1.200  1.061  1.027  1.010  1.005  1.003  1.002     NaN      NaN
    2001  1.193  1.062  1.027  1.014  1.005  1.003    NaN     NaN      NaN
    2002  1.184  1.059  1.029  1.011  1.004    NaN    NaN     NaN      NaN
    2003  1.162  1.057  1.028  1.010    NaN    NaN    NaN     NaN      NaN
    2004  1.159  1.055  1.026    NaN    NaN    NaN    NaN     NaN      NaN
    2005  1.160  1.056    NaN    NaN    NaN    NaN    NaN     NaN      NaN
    2006  1.173    NaN    NaN    NaN    NaN    NaN    NaN     NaN      NaN

To calculate the average age-to-age factors, we will use the ``Development`` class to ``fit_transform`` the original triangle. The specific choice of average paramters (n_period, etc.) are provided to the ``Development`` class. The property that holds the calculated average age-to-age factors is the ``ldf_``. 

PART 3 - Average Age-to-Age Factors
------------------------------------

.. doctest::

    # Simple Average
    # Latest 5
    >>> simple_5 = cl.Development(n_periods=5, average='simple').fit_transform(tri['Reported Claims'])
    >>> simple_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> simple_3 = cl.Development(n_periods=3, average='simple').fit_transform(tri['Reported Claims'])
    >>> simple_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

    # Medial Average
    # Latest 5x1
    >>> medial_5x1 = cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri['Reported Claims'])
    >>> medial_5x1.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.165  1.057  1.027   1.01  1.004  1.003  1.002   1.001      1.0

    # Volume-weighted Average
    # Latest 5
    >>> volume_5 = cl.Development(n_periods=5, average='volume').fit_transform(tri['Reported Claims'])
    >>> volume_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> volume_3 = cl.Development(n_periods=3, average='volume').fit_transform(tri['Reported Claims'])
    >>> volume_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

We will also fake some geometric averages.

.. code-block:: python

    # Geometric Average
    # Latest 4
    >>> geometric_4 = cl.Development(n_periods=3, average='geometric').fit_transform(tri['Reported Claims'])
    >>> geometric_4.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.057  1.027  1.011  1.004  1.003  1.002   1.001      1.0

Now we can select some factors. 

PART 4 - Selected Age-to-Age Factors
--------------------------------------

To create a hard-coded pattern, we can use the ``DevelopmentConstant`` class. In an actual workflow, you can save the development pattern from the prior analysis and load for reference in the following analysis. 

We will also be using the ``TailConstant`` class to add a tail factor to each development pattern

.. doctest::

    # Prior Selected
    >>>  prior_selected = cl.TailConstant(
    ...     tail = 1,
    ...     attachment_age = 120,
    ...     projection_period = 0
    ... ).fit_transform(
    ...     cl.DevelopmentConstant(
    ...         patterns = {
    ...             12:1.16, 
    ...             24:1.057, 
    ...             36:1.028, 
    ...             48:1.012, 
    ...             60:1.005, 
    ...             72:1.003, 
    ...             84:1.001, 
    ...             96:1.001, 
    ...             108:1.000
    ...         }, 
    ...         style='ldf'
    ...     ).fit_transform(tri['Reported Claims'])
    ... )
    >>> prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)   1.16  1.057  1.028  1.012  1.005  1.003  1.001   1.001      1.0      1.0

    # Selected
    >>> selected_pattern = cl.TailConstant(
    ...     tail = 1,
    ...     attachment_age = 120,
    ...     projection_period = 0
    ... ).fit_transform(simple_3)
    >>> selected_pattern = .ldf_.round(decimals=3)
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0      1.0

The Development class has a ``cdf_`` property that will automatically multiples age-to-age factors cumulatively into age-to-ultimate factors. The Friedland text uses the rounded LDF to calculate CDF. Therefore we will manually calculate the rounded age-to-ultimate factors using the ``incr_to_cum()`` method. 

.. doctest::

    # CDF to Ultimate
    >>> selected_cdf = selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3)
    too lazy to type

The triangle manipulation that we used in Chapter 5 can also be used on development patterns. 

.. doctest::

    # Percent Reported
    >>> (1 / selecte_cdf).round(decimals = 3)
    too lazy to type again
