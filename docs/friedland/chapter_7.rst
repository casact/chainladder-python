================================================================
Chapter 7
================================================================

Development Technique
-------------

    The development technique, also known as the chain ladder technique, is one of the most frequently used methodologies for estimating unpaid claims.

    -- Friedland, p84

This chapter covers the foundational development/chainladder method. In the chainladder package, this is implemented in the Development class. 

.. doctest::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import chainladder as cl

Diving straight into Exhibit 1. We will begin by loading the triangle at the top of p106.

.. doctest::

    >>> tri = cl.load_sample('friedland_us_industry_auto')
    # PART 1 - Data Triangle
    >>> tri['Reported Claims'].round(decimals = 0)

To calculate the triangle of age-to-age factors, use the age-to-age property of the triangle. 

.. doctest::

    # PART 2 - Age-to-Age Factors
    >>> tri['Reported Claims'].age_to_age.round(decimals = 3) 

To calculate the average age-to-age factors, we will use the Development class to fit_transform the original triangle. The specific choice of average paramters (n_period, etc.) are provided to the Development class. 

.. doctest::

    # PART 3 - Average Age-to-Age Factors
    # Simple Average
    # Latest 5
    >>> cl.Development(n_periods=5, average='simple').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> cl.Development(n_periods=3, average='simple').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

    # Medial Average
    # Latest 5x1
    >>> cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.165  1.057  1.027   1.01  1.004  1.003  1.002   1.001      1.0

    # Volume-weighted Average
    # Latest 5
    >>> cl.Development(n_periods=5, average='volume').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> cl.Development(n_periods=3, average='volume').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

Now we can select a set of averages. 
