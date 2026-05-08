================================================================
Estimating Unpaid Claims Using Basic Techniques
================================================================

Chapter 7
-------------
We will begin by importing the necessary packages

.. ipython:: python

    import numpy as np
    import pandas as pd
    import chainladder as cl

We will load the underlying dataset for Exhibit I

.. ipython:: python

    tri = cl.load_sample('friedland_us_industry_auto')
    print(tri)

Here is the exhibit from page 106. 

.. ipython::
    :doctest:
    In [1]: 1 + 1
    Out[2]: 3

.. ipython::
    :okwarning:

    # PART 1 - Data Triangle
    In [1]: tri['Reported Claims'].round(decimals = 0)

    # PART 2 - Age-to-Age Factors
    In [2]: tri['Reported Claims'].age_to_age.round(decimals = 3) 

    # PART 3 - Average Age-to-Age Factors
    # Simple Average
    # Latest 5
    In [3]: cl.Development(n_periods=5, average='simple').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
    Out[3]: 12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    In [3]: cl.Development(n_periods=3, average='simple').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
    Out[3]: 12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

    # Medial Average
    # Latest 5x1
    In [4]: cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
    Out[4]: 12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.165  1.057  1.027   1.01  1.004  1.003  1.002   1.001      1.0

    # Volume-weighted Average
    # Latest 5
    In [3]: cl.Development(n_periods=5, average='volume').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
    Out[3]: 12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    In [3]: cl.Development(n_periods=3, average='volume').fit_transform(tri['Reported Claims']).ldf_.round(decimals = 3) 
    Out[3]: 12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

Next, we'll move on to Exhibit II. 
