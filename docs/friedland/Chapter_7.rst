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

P106

We will load the underlying dataset for Exhibit I

.. ipython:: python

    url = 'https://raw.githubusercontent.com/casact/chainladder-python/refs/heads/main/chainladder/utils/data/friedland_us_industry_auto.csv'
    c = pd.read_csv(url)
    tri = cl.Triangle(
        data = c,
        origin = 'Accident Year',
        development = 'Calendar Year',
        columns = ['Reported Claims','Paid Claims'], 
        cumulative = True
    )
    print(tri)

We will define a couple of reusable functions for multiple exhibits in this chapter

.. ipython::

    # PART 1 - Data Triangle
    In [1]: tri.round(decimals = 0) 

    # PART 2 - Age-to-Age Factors
    In [2]: tri.age_to_age.round(decimals = 3) 

    # PART 3 - Average Age-to-Age Factors
    # Simple Average
    # Latest 5
    In [3]: cl.Development(n_periods=5, average='simple').fit_transform(tri).ldf_.round(decimals = 3) 

    # Latest 3
    In [3]: cl.Development(n_periods=3, average='simple').fit_transform(tri).ldf_.round(decimals = 3) 

    # Medial Average
    # Latest 5x1
    In [4]: cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri)

    # Volume-weighted Average
    # Latest 5
    In [3]: cl.Development(n_periods=5, average='volume').fit_transform(tri).ldf_.round(decimals = 3) 

    # Latest 5
    In [3]: cl.Development(n_periods=5, average='volume').fit_transform(tri).ldf_.round(decimals = 3) 

Now we use the functions we just defined to actually perform the analysis
