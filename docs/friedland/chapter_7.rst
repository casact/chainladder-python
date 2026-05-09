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

To create a hard-coded pattern, we can use the ``DevelopmentConstant`` class. In a production workflow, you can save the development pattern from the prior analysis and load for reference in the following analysis. 

We will also be using the ``TailConstant`` class to add a tail factor to each development pattern. 

.. doctest::

    # Prior Selected
    >>> prior_method =  cl.DevelopmentConstant(
    ...      patterns = {
    ...         12:1.16, 
    ...         24:1.057, 
    ...         36:1.028, 
    ...         48:1.012, 
    ...         60:1.005, 
    ...         72:1.003, 
    ...         84:1.001, 
    ...         96:1.001, 
    ...         108:1.000
    ...     }, 
    ...     style='ldf'
    ... )
    >>> prior_ft = prior_method.fit_transform(tri['Reported Claims'])
    >>> tail_method = cl.TailConstant(
    ...     tail = 1,
    ...     attachment_age = 120,
    ...     projection_period = 0
    ... )
    >>> prior_selected = tail_method.fit_transform(prior_ft)
    >>> prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)   1.16  1.057  1.028  1.012  1.005  1.003  1.001   1.001      1.0      1.0

    # Selected
    >>> selected_pattern = tail_method.fit_transform(simple_3)
    >>> selected_pattern.ldf_.round(decimals=3)
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
    >>> (1 / selected_cdf).round(decimals = 3)
    too lazy to type again

p107
==================

Moving onto the next page, all the calculations are identical to the previous page. In a production workflow, commonly repeated methods and selections can be made into pipelines for repetition. 

PART 1 - Data Triangle
-----------------------

.. doctest::

    >>> tri['Paid Claims'].round(decimals = 0)
                  12          24          36          48          60          72          84          96         108         120
    1998  18539254.0  33231039.0  40062008.0  43892039.0  45896535.0  46765422.0  47221322.0  47446877.0  47555456.0  47644187.0
    1999  20410193.0  36090684.0  43259402.0  47159241.0  49208532.0  50162043.0  50625757.0  50878808.0  51000534.0         NaN
    2000  22120843.0  38976014.0  46389282.0  50562385.0  52735280.0  53740101.0  54284334.0  54533225.0         NaN         NaN
    2001  22992259.0  40096198.0  47767835.0  52093916.0  54363436.0  55378801.0  55878421.0         NaN         NaN         NaN
    2002  24092782.0  41795313.0  49903803.0  54352884.0  56754376.0  57807215.0         NaN         NaN         NaN         NaN
    2003  24084451.0  41399612.0  49070332.0  53584201.0  55930654.0         NaN         NaN         NaN         NaN         NaN
    2004  24369770.0  41489863.0  49236678.0  53774672.0         NaN         NaN         NaN         NaN         NaN         NaN
    2005  25100697.0  42702229.0  50644994.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN
    2006  25608776.0  43606497.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN
    2007  27229969.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN

PART 2 - Age-to-Age Factors
----------------------------

.. doctest::
    
    >>> tri['Paid Claims'].age_to_age.round(decimals = 3) 
             12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120
    1998  1.792469  1.205560  1.095603  1.045669  1.018931  1.009749  1.004777  1.002288  1.001866
    1999  1.768268  1.198631  1.090150  1.043455  1.019377  1.009244  1.004998  1.002392       NaN
    2000  1.761959  1.190201  1.089958  1.042975  1.019054  1.010127  1.004585       NaN       NaN
    2001  1.743900  1.191331  1.090565  1.043566  1.018677  1.009022       NaN       NaN       NaN
    2002  1.734765  1.194005  1.089153  1.044183  1.018551       NaN       NaN       NaN       NaN
    2003  1.718935  1.185285  1.091988  1.043790       NaN       NaN       NaN       NaN       NaN
    2004  1.702514  1.186716  1.092167       NaN       NaN       NaN       NaN       NaN       NaN
    2005  1.701237  1.186004       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    2006  1.702795       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

PART 3 - Average Age-to-Age Factors
------------------------------------

.. doctest::

    # Simple Average
    # Latest 5
    >>> simple_5 = cl.Development(n_periods=5, average='simple').fit_transform(tri['Paid Claims'])
    >>> simple_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> simple_3 = cl.Development(n_periods=3, average='simple').fit_transform(tri['Paid Claims'])
    >>> simple_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

    # Medial Average
    # Latest 5x1
    >>> medial_5x1 = cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri['Paid Claims'])
    >>> medial_5x1.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.165  1.057  1.027   1.01  1.004  1.003  1.002   1.001      1.0

    # Volume-weighted Average
    # Latest 5
    >>> volume_5 = cl.Development(n_periods=5, average='volume').fit_transform(tri['Paid Claims'])
    >>> volume_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> volume_3 = cl.Development(n_periods=3, average='volume').fit_transform(tri['Paid Claims'])
    >>> volume_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

.. code-block:: python

    # Geometric Average
    # Latest 4
    >>> geometric_4 = cl.Development(n_periods=3, average='geometric').fit_transform(tri['Reported Claims'])
    >>> geometric_4.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.057  1.027  1.011  1.004  1.003  1.002   1.001      1.0

PART 4 - Selected Age-to-Age Factors
--------------------------------------

.. doctest::

    # Prior Selected
    >>> prior_method =  cl.DevelopmentConstant(
    ...      patterns = {
    ...         12:1.16, 
    ...         24:1.057, 
    ...         36:1.028, 
    ...         48:1.012, 
    ...         60:1.005, 
    ...         72:1.003, 
    ...         84:1.001, 
    ...         96:1.001, 
    ...         108:1.000
    ...     }, 
    ...     style='ldf'
    ... )
    >>> prior_ft = prior_method.fit_transform(tri['Reported Claims'])
    >>> tail_method = cl.TailConstant(
    ...     tail = 1,
    ...     attachment_age = 120,
    ...     projection_period = 0
    ... )
    >>> prior_selected = tail_method.fit_transform(prior_ft)
    >>> prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)   1.16  1.057  1.028  1.012  1.005  1.003  1.001   1.001      1.0      1.0

    # Selected
    >>> selected_pattern = tail_method.fit_transform(simple_3)
    >>> selected_pattern = selected_pattern.ldf_.round(decimals=3)
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0      1.0

    # CDF to Ultimate
    >>> selected_cdf = selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3)
    too lazy to type

    # Percent Reported
    >>> (1 / selected_cdf).round(decimals = 3)
    too lazy to type again
