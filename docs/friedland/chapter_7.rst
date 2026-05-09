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

Exhibit I Sheet 1 p106
========================

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
    >>> reported_simple_5 = cl.Development(n_periods=5, average='simple').fit_transform(tri['Reported Claims'])
    >>> reported_simple_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> reported_simple_3 = cl.Development(n_periods=3, average='simple').fit_transform(tri['Reported Claims'])
    >>> reported_simple_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

    # Medial Average
    # Latest 5x1
    >>> reported_medial_5x1 = cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri['Reported Claims'])
    >>> reported_medial_5x1.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.165  1.057  1.027   1.01  1.004  1.003  1.002   1.001      1.0

    # Volume-weighted Average
    # Latest 5
    >>> reported_volume_5 = cl.Development(n_periods=5, average='volume').fit_transform(tri['Reported Claims'])
    >>> reported_volume_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.168  1.058  1.027  1.011  1.004  1.003  1.002   1.001      1.0

    # Latest 3
    >>> reported_volume_3 = cl.Development(n_periods=3, average='volume').fit_transform(tri['Reported Claims'])
    >>> reported_volume_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0

We will also fake some geometric averages.

.. code-block:: python

    # Geometric Average
    # Latest 4
    >>> reported_geometric_4 = cl.Development(n_periods=3, average='geometric').fit_transform(tri['Reported Claims'])
    >>> reported_geometric_4.ldf_.round(decimals = 3) 
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
    >>> reported_prior_selected = tail_method.fit_transform(prior_ft)
    >>> prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)   1.16  1.057  1.028  1.012  1.005  1.003  1.001   1.001      1.0      1.0

    # Selected
    >>> reported_selected_pattern = tail_method.fit_transform(reported_simple_3)
    >>> selected_pattern.ldf_.round(decimals=3)
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0      1.0

The Development class has a ``cdf_`` property that will automatically multiples age-to-age factors cumulatively into age-to-ultimate factors. The Friedland text uses the rounded LDF to calculate CDF. Therefore we will manually calculate the rounded age-to-ultimate factors using the ``incr_to_cum()`` method. 

.. doctest::

    # CDF to Ultimate
    >>> reported_selected_cdf = reported_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3)
    >>> reported_selected_cdf
    too lazy to type

The triangle manipulation that we used in Chapter 5 can also be used on development patterns. 

.. doctest::

    # Percent Reported
    >>> (1 / reported_selected_cdf).round(decimals = 3)
           12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
    (All)   0.774   0.901   0.951   0.978   0.989   0.994   0.997   0.999      1.0      1.0

Exhibit I Sheet 2 p107
========================

Moving onto the next page, all the calculations are identical to the previous page. In a production workflow, commonly repeated methods and selections can be made into pipelines for repetition. 

PART 1 - Data Triangle
-----------------------

.. doctest::

    >>> tri['Paid Claims'].round(decimals = 0)
                 12          24          36          48          60          72          84          96          108         120
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
          12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    1998  1.792  1.206  1.096  1.046  1.019  1.010  1.005   1.002    1.002
    1999  1.768  1.199  1.090  1.043  1.019  1.009  1.005   1.002      NaN
    2000  1.762  1.190  1.090  1.043  1.019  1.010  1.005     NaN      NaN
    2001  1.744  1.191  1.091  1.044  1.019  1.009    NaN     NaN      NaN
    2002  1.735  1.194  1.089  1.044  1.019    NaN    NaN     NaN      NaN
    2003  1.719  1.185  1.092  1.044    NaN    NaN    NaN     NaN      NaN
    2004  1.703  1.187  1.092    NaN    NaN    NaN    NaN     NaN      NaN
    2005  1.701  1.186    NaN    NaN    NaN    NaN    NaN     NaN      NaN
    2006  1.703    NaN    NaN    NaN    NaN    NaN    NaN     NaN      NaN

PART 3 - Average Age-to-Age Factors
------------------------------------

.. doctest::

    # Simple Average
    # Latest 5
    >>> paid_simple_5 = cl.Development(n_periods=5, average='simple').fit_transform(tri['Paid Claims'])
    >>> paid_simple_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.712  1.189  1.091  1.044  1.019   1.01  1.005   1.002    1.002

    # Latest 3
    >>> paid_simple_3 = cl.Development(n_periods=3, average='simple').fit_transform(tri['Paid Claims'])
    >>> paid_simple_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.702  1.186  1.091  1.044  1.019  1.009  1.005   1.002    1.002

    # Medial Average
    # Latest 5x1
    >>> paid_medial_5x1 = cl.Development(n_periods=5, average='simple',drop_high = 1, drop_low = 1).fit_transform(tri['Paid Claims'])
    >>> paid_medial_5x1.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.708  1.188  1.091  1.044  1.019  1.009  1.005   1.002    1.002

    # Volume-weighted Average
    # Latest 5
    >>> paid_volume_5 = cl.Development(n_periods=5, average='volume').fit_transform(tri['Paid Claims'])
    >>> paid_volume_5.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.712  1.189  1.091  1.044  1.019   1.01  1.005   1.002    1.002

    # Latest 3
    >>> paid_volume_3 = cl.Development(n_periods=3, average='volume').fit_transform(tri['Paid Claims'])
    >>> paid_volume_3.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.702  1.186  1.091  1.044  1.019  1.009  1.005   1.002    1.002

.. code-block:: python

    # Geometric Average
    # Latest 4
    >>> paid_geometric_4 = cl.Development(n_periods=3, average='geometric').fit_transform(tri['Paid Claims'])
    >>> paid_geometric_4.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.704  1.188  1.091  1.044  1.019  1.010  1.005   1.002    1.002 

PART 4 - Selected Age-to-Age Factors
--------------------------------------

.. doctest::

    # Prior Selected
    >>> paid_prior_method =  cl.DevelopmentConstant(
    ...      patterns = {
    ...         12:1.707, 
    ...         24:1.189, 
    ...         36:1.091, 
    ...         48:1.044, 
    ...         60:1.019, 
    ...         72:1.01, 
    ...         84:1.005, 
    ...         96:1.003, 
    ...         108:1.001
    ...     }, 
    ...     style='ldf'
    ... )
    >>> paid_prior_ft = paid_prior_method.fit_transform(tri['Paid Claims'])
    >>> tail_method = cl.TailConstant(
    ...     tail = 1.002,
    ...     attachment_age = 120,
    ...     projection_period = 0
    ... )
    >>> paid_prior_selected = tail_method.fit_transform(paid_prior_ft)
    >>> paid_prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.707  1.189  1.091  1.044  1.019   1.01  1.005   1.003    1.001    1.002

    # Selected
    >>> paid_selected_pattern = paid_tail_method.fit_transform(paid_simple_3)
    >>> paid_selected_pattern.ldf_.round(decimals=3)
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0      1.0

    # CDF to Ultimate
    >>> paid_selected_cdf = paid_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3)
    too lazy to type

    # Percent Reported
    >>> (1 / paid_selected_cdf).round(decimals = 3)
           12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
    (All)   0.774   0.901   0.951   0.978   0.989   0.994   0.997   0.999      1.0      1.0

Exhibit I Sheet 3 p108
========================

This is a common report layout for reserving analyses. Some Pandas manipulation is needed to achieve the tabular look.  

.. doctest::

    >>> exhibit = pd.DataFrame() # initializing a DataFrame
    >>> exhibit["Reported Claims"] = reported_selected_pattern.latest_diagonal.to_frame(origin_as_datetime=False).iloc[:,0].fillna(0) # using a vector of losses to anchor the exhibit index
    >>> age = reported_selected_pattern.development.iloc[::-1] # flipping the age order
    >>> age.index = exhibit.index # forcing the index to match
    >>> exhibit['Age'] = age
    >>> exhibit = exhibit[['Age','Reported Claims']] # reordering the columns
    >>> exhibit ['Paid Claims'] = paid_selected_pattern.latest_diagonal.to_frame(origin_as_datetime=False).iloc[:,0].fillna(0) # adding in paid losses
    >>> reported_cdf = reported_reported_pattern.cdf_.T # transposing the CDF
    >>> reported_cdf.index = exhibit.index[::-1] # forcing the index to match
    >>> exhibit["Reported CDF"] = reported_cdf 
    >>> paid_cdf = paid_selected_pattern.cdf_.T
    >>> paid_cdf.index = exhibit.index[::-1]
    >>> exhibit["Paid CDF"] = paid_cdf
    >>> exhibit["Reported Ultimate"] = cl.Chainladder().fit(reported_selected_pattern).ultimate_.to_frame(origin_as_datetime=False).iloc[:,0].fillna(0)
    >>> exhibit["Paid Ultimate"] = cl.Chainladder().fit(paid_selected_pattern).ultimate_.to_frame(origin_as_datetime=False).iloc[:,0].fillna(0)
    >>> exhibit
    too lazy to type

Unfortunately this does not match the table from the text, due to rounding. We will construct a separate, rounded exhibit to demonstrate parity. 

.. doctest::

    >>> rounded_exhibit = pd.DataFrame() # initializing a DataFrame
    >>> rounded_exhibit['Age'] = exhibit['Age']
    >>> rounded_exhibit['Reported Claims'] = exhibit['Reported Claims'].round(decimals=0)
    >>> rounded_exhibit['Paid Claims'] = exhibit['Paid Claims'].round(decimals=0)
    >>> rounded_reported_cdf = reported_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3).T
    >>> rounded_reported_cdf.index = rounded_exhibit.index[::-1]
    >>> rounded_exhibit["Reported CDF"] = rounded_reported_cdf
    >>> rounded_paid_cdf = paid_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3).T
    >>> rounded_paid_cdf.index = rounded_exhibit.index[::-1]
    >>> rounded_exhibit["Paid CDF"] = rounded_paid_cdf
    >>> rounded_exhibit["Reported Ultimate"] = rounded_exhibit['Reported Claims'] * rounded_exhibit["Reported CDF"]
    >>> rounded_exhibit["Paid Ultimate"] = rounded_exhibit['Paid Claims'] * rounded_exhibit["Paid CDF"]
    >>> rounded_exhibit
    too lazy to type

