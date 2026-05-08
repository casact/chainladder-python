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

    url = 'https://raw.githubusercontent.com/casact/chainladder-python/refs/heads/master/chainladder/utils/data/friedland_us_industry_auto.csv'
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

.. ipython:: python

    from typing import List

    def standard_analysis(tri: cl.Triangle()) -> List[cl.Triangle()]:
        simple_5 = cl.Development(n_periods=5, average='simple').fit_transform(tri)
        simple_3 = cl.Development(n_periods=3, average='simple').fit_transform(tri)
        medial_5x1 = cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri)
        volume_5 = cl.Development(n_periods=5, average='volume').fit_transform(tri)
        volume_3 = cl.Development(n_periods=3, average='volume').fit_transform(tri)
        devs = {
        "triangle" : tri.copy(),
        "simple_5": simple_5,
        "simple_3": simple_3,
        "medial_5x1": medial_5x1,
        "volume_5": volume_5,
        "volume_3": volume_3,
        }
        return devs

    def display_standard_analysis(devs: List[cl.Triangle()]) -> None:
        print('PART 1 - Data Triangle')
        print(devs['triangle'])
        print('PART 2 - Age-to-Age Factors')
        print(devs['triangle'].age_to_age)
        print('PART 3 - Average Age-to-Age Factors')
        print('Simple Average')
        print('Latest 5')
        print(devs['simple_5'].ldf_)
        print('Latest 3')
        print(devs['simple_3'].ldf_)
        print('Medial Average')
        print('Latest 5x1')
        print(devs['medial_5x1'].ldf_)
        print('Volume-weighted Average')
        print('Latest 5')
        print(devs['volume_5'].ldf_)
        print('Latest 3')
        print(devs['volume_3'].ldf_)
        print('Geometric Average')
        print('Latest 4')
        print('Coming soon')
        return None

Now we use the functions we just defined to actually perform the analysis

.. ipython:: python

    reported_devs = standard_analysis(tri['Reported Claims'])
    display_standard_analysis(reported_devs)
