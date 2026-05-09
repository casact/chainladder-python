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

For the prior selected, we need to create a hard-coded pattern, using the ``DevelopmentConstant`` class. In a production workflow, you can save the development pattern from the prior analysis and load for reference in the subsequent analysis. 

We will also be using the ``TailConstant`` class to add a tail factor to each development pattern. 

.. doctest::

    # Prior Selected
    >>> reported_prior_method =  cl.DevelopmentConstant(
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
    >>> reported_prior_ft = reported_prior_method.fit_transform(tri['Reported Claims'])
    >>> reported_tail_method = cl.TailConstant(
    ...     tail = 1,
    ...     attachment_age = 120,
    ...     projection_period = 0
    ... )
    >>> reported_prior_selected = reported_tail_method.fit_transform(reported_prior_ft)
    >>> reported_prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)   1.16  1.057  1.028  1.012  1.005  1.003  1.001   1.001      1.0      1.0

    # Selected
    >>> reported_selected_pattern = reported_tail_method.fit_transform(reported_simple_3)
    >>> reported_selected_pattern.ldf_.round(decimals=3)
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0      1.0

The Development class has a ``cdf_`` property that will automatically multiply age-to-age factors cumulatively into age-to-ultimate factors. The Friedland text uses the rounded LDF to calculate CDF. Therefore we will manually calculate the rounded age-to-ultimate factors using the ``incr_to_cum()`` method. 

.. doctest::

    # CDF to Ultimate
    >>> reported_selected_cdf = reported_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3)
    >>> reported_selected_cdf
           12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
    (All)   1.292    1.11   1.051   1.023   1.011   1.006   1.003   1.001      1.0      1.0

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
    >>> paid_tail_method = cl.TailConstant(
    ...     tail = 1.002,
    ...     attachment_age = 120,
    ...     projection_period = 0
    ... )
    >>> paid_prior_selected = paid_tail_method.fit_transform(paid_prior_ft)
    >>> paid_prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.707  1.189  1.091  1.044  1.019   1.01  1.005   1.003    1.001    1.002

    # Selected
    >>> paid_selected_pattern = paid_tail_method.fit_transform(paid_simple_3)
    >>> paid_selected_pattern.ldf_.round(decimals=3)
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.702  1.186  1.091  1.044  1.019  1.009  1.005   1.002    1.002    1.002

    # CDF to Ultimate
    >>> paid_selected_cdf = paid_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3)
    >>> paid_selected_cdf
           12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
    (All)    2.39   1.404   1.184   1.085    1.04    1.02   1.011   1.006    1.004    1.002

    # Percent Reported
    >>> (1 / paid_selected_cdf).round(decimals = 3)
           12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
    (All)   0.418   0.712   0.845   0.922   0.962    0.98   0.989   0.994    0.996    0.998

Exhibit I Sheet 3 p108
========================

This is a common report layout for reserving analyses. Some Pandas manipulation is needed to achieve the tabular look.  

.. doctest::

    >>> exhibit = pd.DataFrame() # initializing a DataFrame
    >>> exhibit["Reported Claims"] = reported_selected_pattern.latest_diagonal.to_frame(origin_as_datetime=False) # using a vector of losses to anchor the exhibit index
    >>> age = reported_selected_pattern.development.iloc[::-1] # flipping the age order
    >>> age.index = exhibit.index # forcing the index to match
    >>> exhibit['Age'] = age
    >>> exhibit = exhibit[['Age','Reported Claims']] # reordering the columns
    >>> exhibit ['Paid Claims'] = paid_selected_pattern.latest_diagonal.to_frame(origin_as_datetime=False) # adding in paid losses
    >>> reported_cdf = reported_selected_pattern.cdf_.T # transposing the CDF
    >>> reported_cdf.index = exhibit.index[::-1] # forcing the index to match
    >>> exhibit["Reported CDF"] = reported_cdf 
    >>> paid_cdf = paid_selected_pattern.cdf_.T
    >>> paid_cdf.index = exhibit.index[::-1]
    >>> exhibit["Paid CDF"] = paid_cdf
    >>> exhibit["Reported Ultimate"] = cl.Chainladder().fit(reported_selected_pattern).ultimate_.to_frame(origin_as_datetime=False) # using the chainladder predictor to return the ultimate
    >>> exhibit["Paid Ultimate"] = cl.Chainladder().fit(paid_selected_pattern).ultimate_.to_frame(origin_as_datetime=False)
    >>> exhibit[['Age','Reported Claims','Paid Claims']] # splitting the display due to the number of columns
          Age  Reported Claims  Paid Claims
    1998  120       47742304.0   47644187.0
    1999  108       51185767.0   51000534.0
    2000   96       54837929.0   54533225.0
    2001   84       56299562.0   55878421.0
    2002   72       58592712.0   57807215.0
    2003   60       57565344.0   55930654.0
    2004   48       56976657.0   53774672.0
    2005   36       56786410.0   50644994.0
    2006   24       54641339.0   43606497.0
    2007   12       48853563.0   27229969.0

    >>> exhibit[['Reported CDF','Paid CDF','Reported Ultimate','Paid Ultimate']]
          Reported CDF  Paid CDF  Reported Ultimate  Paid Ultimate
    1998      1.000000  1.002000       4.774230e+07   4.773948e+07
    1999      1.000369  1.003870       5.120467e+07   5.119788e+07
    2000      1.000954  1.006219       5.489024e+07   5.487237e+07
    2001      1.002540  1.011036       5.644257e+07   5.649507e+07
    2002      1.005300  1.020604       5.890327e+07   5.899830e+07
    2003      1.009908  1.039752       5.813573e+07   5.815399e+07
    2004      1.021554  1.085341       5.820476e+07   5.836386e+07
    2005      1.049493  1.184218       5.959697e+07   5.997474e+07
    2006      1.108139  1.404485       6.055018e+07   6.124466e+07
    2007      1.289977  2.390688       6.301997e+07   6.509837e+07

Unfortunately this does not match the table from the text, due to rounding. We will construct a separate, rounded exhibit to demonstrate parity. 

.. doctest::

    >>> rounded_exhibit = pd.DataFrame() # initializing a DataFrame
    >>> rounded_exhibit['Age'] = exhibit['Age']
    >>> rounded_exhibit['Reported Claims'] = exhibit['Reported Claims']
    >>> rounded_exhibit['Paid Claims'] = exhibit['Paid Claims']
    >>> rounded_reported_cdf = reported_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3).T
    >>> rounded_reported_cdf.index = rounded_exhibit.index[::-1]
    >>> rounded_exhibit["Reported CDF"] = rounded_reported_cdf
    >>> rounded_paid_cdf = paid_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3).T
    >>> rounded_paid_cdf.index = rounded_exhibit.index[::-1]
    >>> rounded_exhibit["Paid CDF"] = rounded_paid_cdf
    >>> rounded_exhibit["Reported Ultimate"] = (rounded_exhibit['Reported Claims'] * rounded_exhibit["Reported CDF"])
    >>> rounded_exhibit["Paid Ultimate"] = (rounded_exhibit['Paid Claims'] * rounded_exhibit["Paid CDF"])
    >>> rounded_exhibit[['Reported CDF','Paid CDF','Reported Ultimate','Paid Ultimate']] # only displaying the rounded columns
          Reported CDF  Paid CDF  Reported Ultimate  Paid Ultimate
    1998         1.000     1.002       4.774230e+07   4.773948e+07
    1999         1.000     1.004       5.118577e+07   5.120454e+07
    2000         1.001     1.006       5.489277e+07   5.486042e+07
    2001         1.003     1.011       5.646846e+07   5.649308e+07
    2002         1.006     1.020       5.894427e+07   5.896336e+07
    2003         1.011     1.040       5.819856e+07   5.816788e+07
    2004         1.023     1.085       5.828712e+07   5.834552e+07
    2005         1.051     1.184       5.968252e+07   5.996367e+07
    2006         1.110     1.404       6.065189e+07   6.122352e+07
    2007         1.292     2.390       6.311880e+07   6.507963e+07

Exhibit I Sheet 4 p109
========================

This is another common report layout for reserving analyses. The manipulation here are more straight-forward.  

.. doctest::

    >>> unpaid_exhibit = rounded_exhibit[['Reported Claims','Paid Claims','Reported Ultimate','Paid Ultimate']]
    >>> unpaid_exhibit['Case Outstanding'] = unpaid_exhibit['Reported Claims'] - unpaid_exhibit['Paid Claims']
    >>> unpaid_exhibit['Reported IBNR'] = unpaid_exhibit['Reported Ultimate'] - unpaid_exhibit['Reported Claims']
    >>> unpaid_exhibit['Paid IBNR'] = unpaid_exhibit['Paid Ultimate'] - unpaid_exhibit['Paid Claims']
    >>> unpaid_exhibit['Reported Unpaid'] = unpaid_exhibit['Reported IBNR'] + unpaid_exhibit['Case Outstanding']
    >>> unpaid_exhibit['Paid Unpaid'] = unpaid_exhibit['Paid IBNR'] + unpaid_exhibit['Case Outstanding']
    >>> unpaid_exhibit[['Case Outstanding','Reported IBNR']]
          Case Outstanding  Reported IBNR
    1998           98117.0   0.000000e+00
    1999          185233.0   0.000000e+00
    2000          304704.0   5.483793e+04
    2001          421141.0   1.688987e+05
    2002          785497.0   3.515563e+05
    2003         1634690.0   6.332188e+05
    2004         3201985.0   1.310463e+06
    2005         6141416.0   2.896107e+06
    2006        11034842.0   6.010547e+06
    2007        21623594.0   1.426524e+07

    >>> unpaid_exhibit[['Paid IBNR','Reported Unpaid','Paid Unpaid']]
             Paid IBNR  Reported Unpaid   Paid Unpaid
    1998  9.528837e+04     9.811700e+04  1.934054e+05
    1999  2.040021e+05     1.852330e+05  3.892351e+05
    2000  3.271994e+05     3.595419e+05  6.319034e+05
    2001  6.146626e+05     5.900397e+05  1.035804e+06
    2002  1.156144e+06     1.137053e+06  1.941641e+06
    2003  2.237226e+06     2.267909e+06  3.871916e+06
    2004  4.570847e+06     4.512448e+06  7.772832e+06
    2005  9.318679e+06     9.037523e+06  1.546009e+07
    2006  1.761702e+07     1.704539e+07  2.865187e+07
    2007  3.784966e+07     3.588883e+07  5.947325e+07

Exhibit II Sheet 1 p110
========================

Now that we have walked through an analysis step by step, let's introduce some scaling by streamlining the creation of individual ``Development`` objects, 

.. doctest::

    >>> import re
    >>> tri = cl.load_sample('friedland_xyz_auto_bi')
    >>> assumptions_list = ['simple_5','simple_3','simple_2','volume_4','volume_3','volume_2']
    >>> assumptions = {x:{'n_periods':int(re.match(r'.+_(.+)', x).group(1)),'average':re.match(r'(.+)_', x).group(1)} for x in assumptions_list}
    >>> assumptions['medial 5x1'] = {'n_periods':5, 'average':'simple','drop_high':1, 'drop_low':1}
    >>> devs = {}
    >>> tails = {'Reported Claims':1,'Paid Claims':1.01}
    >>> selections = {'Reported Claims':'volume_2','Paid Claims':'volume_2'}
    >>> for x in tri.columns:
    ...     print(tri[x])
    ...     print(tri[x].age_to_age)
    ...     devs[x] = {}
    ...     for k,v in assumptions.items():
    ...         devs[x][k] = cl.Development(**v).fit_transform(tri[x])
    ...         print(devs[x][k].ldf_.round(decimals=3))
    ...     devs[x]["selected"] = cl.TailConstant(tail = tails[x], attachment_age = 132, projection_period = 0).fit_transform(devs[x][selections[x]])
    ...     print(devs[x]["selected"].ldf_.round(decimals=3))
    ...     sel_cdf = devs[x]["selected"].ldf_.round(decimals=3).incr_to_cum()
    ...     print(sel_cdf)
    ...     print((1/sel_cdf).round(decimals=3))
    way too lazy to type

let's see what happens
