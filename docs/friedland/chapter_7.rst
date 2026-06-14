================================================================
Chapter 7 - Development Technique
================================================================

    The development technique, also known as the chain ladder technique, is one of the most frequently used methodologies for estimating unpaid claims.

    -- Friedland, p84

This chapter covers the foundational development/chainladder method. In the chainladder package, this is implemented in the ``Development`` estimator. 

.. doctest::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import chainladder as cl
    >>> pd.set_option('display.max_columns', None)
    >>> pd.set_option('display.width', 1000)

Exhibit I Sheet 1 p106
##########################

Diving straight into Exhibit 1. 

PART 1 - Data Triangle
-----------------------

We have already imported the necessary packages loading the ``Triangle`` at the top of p106. Let's take a look at the ``Triangle`` we just loaded. 

.. doctest::

    >>> tri = cl.load_sample('friedland_us_industry_auto')
    >>> tri['Reported Claims']
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

PART 2 - Age-to-Age Factors
----------------------------

To calculate age-to-age factors, use the ``age-to-age`` attribute of the ``Triangle``. 

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

PART 3 - Average Age-to-Age Factors
------------------------------------

To calculate the average age-to-age factors, we will use the ``Development`` estimator to ``fit_transform`` the original ``Triangle``. This calculates the averages but also preserves the ability to apply other estimators later. The specific choices of average paramters (n_period, etc.) are provided to ``Development``. The attribute for the calculated average age-to-age factors is the ``ldf_``. 

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

    # Geometric Average
    # Latest 4
    >>> reported_geometric_4 = cl.Development(n_periods=4, average='geometric').fit_transform(tri['Reported Claims'])
    >>> reported_geometric_4.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.164  1.057  1.027  1.011  1.004  1.003  1.002   1.001      1.0

PART 4 - Selected Age-to-Age Factors
--------------------------------------

For the prior selected, we need to create a hard-coded pattern, using the ``DevelopmentConstant`` estimator. In a production workflow, you can save the development pattern from the prior analysis and load for reference in a subsequent analysis. 

We will also be using the ``TailConstant`` estimator to add a tail factor to selected development patterns. We add the tail factor to a transformed ``Triangle`` (i.e. applying ``fit_transforme`` of the ``Development`` estimator to a ``Triangle``) by using ``fit_transform`` once again. 

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
    ...     projection_period = 0
    ... )
    >>> reported_prior_selected = reported_tail_method.fit_transform(reported_prior_ft)
    >>> reported_prior_selected.ldf_
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)   1.16  1.057  1.028  1.012  1.005  1.003  1.001   1.001      1.0      1.0

Next we can select some factors. We can also reuse the `TailConstant` from the previous step. This is fairly common in practice, as tail factors are selected less frequently than the development pattern itself, so need to be carried from analysis to analysis. 

.. doctest::

    # Selected
    >>> reported_selected_pattern = reported_tail_method.fit_transform(reported_simple_3)
    >>> reported_selected_pattern.ldf_.round(decimals=3)
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    (All)  1.164  1.056  1.027  1.012  1.005  1.003  1.002   1.001      1.0      1.0

The Development estimator has a ``cdf_`` attribute that will automatically multiply age-to-age factors cumulatively into age-to-ultimate factors. The Friedland text uses the rounded LDF to calculate CDF. This can be achieved in this package by using the ``incr_to_cum()`` method of the rounded age-to-age factors. 

.. doctest::

    # CDF to Ultimate
    # First without rounding
    >>> reported_selected_pattern.cdf_
             12-Ult    24-Ult    36-Ult    48-Ult    60-Ult  72-Ult   84-Ult    96-Ult   108-Ult  120-Ult
    (All)  1.289977  1.108139  1.049493  1.021554  1.009908  1.0053  1.00254  1.000954  1.000369      1.0

    # Then with rounding
    >>> reported_selected_cdf = reported_selected_pattern.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3)
    >>> reported_selected_cdf
           12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
    (All)   1.292    1.11   1.051   1.023   1.011   1.006   1.003   1.001      1.0      1.0

To calculate % reported, we will use ``Triangle`` manipulation from Chapter 5 directly on the development pattern (which is also a ``Triangle``). 

.. doctest::

    # Percent Reported
    >>> (1 / reported_selected_cdf).round(decimals = 3)
           12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
    (All)   0.774   0.901   0.951   0.978   0.989   0.994   0.997   0.999      1.0      1.0

Exhibit I Sheet 2 p107
##########################

Moving onto the next page, all the calculations are identical to the previous page. We will manually repeat the same code. In a production workflow, commonly repeated methods and selections can be streamlined, which we will demonstrate in Exhibit II. 

PART 1 - Data Triangle
-----------------------

.. doctest::

    >>> tri['Paid Claims']
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

    # Geometric Average
    # Latest 4
    >>> paid_geometric_4 = cl.Development(n_periods=4, average='geometric').fit_transform(tri['Paid Claims'])
    >>> paid_geometric_4.ldf_.round(decimals = 3) 
           12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120
    (All)  1.706  1.188  1.091  1.044  1.019   1.01  1.005   1.002    1.002 

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
##########################

This is a common report layout for reserving analyses. Some ``Pandas`` manipulation is needed to retrieve all the figures from the transformed ``Triangle`` objects and achieve the tabular look. We will create a function to reuse the manipulation throughout this chapter. 

.. doctest::

    >>> def development_summary(reported: cl.Triangle(), paid: cl.Triangle()) -> pd.DataFrame():
    ...     output = pd.DataFrame() # initializing a DataFrame
    ...     output["Reported Claims"] = reported.latest_diagonal.to_frame(origin_as_datetime=False) # using a vector of losses to anchor the exhibit index
    ...     age = reported.development.iloc[::-1] # flipping the age order
    ...     age.index = output.index # forcing the index to match
    ...     output['Age'] = age
    ...     output = output[['Age','Reported Claims']] # reordering the columns
    ...     output ['Paid Claims'] = paid.latest_diagonal.to_frame(origin_as_datetime=False) # adding in paid losses
    ...     reported_cdf = reported.cdf_.T # transposing the CDF
    ...     reported_cdf.index = output.index[::-1] # forcing the index to match
    ...     output["Reported CDF"] = reported_cdf 
    ...     paid_cdf = paid.cdf_.T
    ...     paid_cdf.index = output.index[::-1]
    ...     output["Paid CDF"] = paid_cdf
    ...     output["Reported Ultimate"] = cl.Chainladder().fit(reported).ultimate_.to_frame(origin_as_datetime=False) # using the Chainladder estimator to return the ultimate
    ...     output["Paid Ultimate"] = cl.Chainladder().fit(paid).ultimate_.to_frame(origin_as_datetime=False)
    ...     return output
    >>> exhibit = development_summary(reported_selected_pattern,paid_selected_pattern)
    >>> exhibit
          Age  Reported Claims  Paid Claims  Reported CDF  Paid CDF  Reported Ultimate  Paid Ultimate
    1998  120       47742304.0   47644187.0      1.000000  1.002000       4.774230e+07   4.773948e+07
    1999  108       51185767.0   51000534.0      1.000369  1.003870       5.120467e+07   5.119788e+07
    2000   96       54837929.0   54533225.0      1.000954  1.006219       5.489024e+07   5.487237e+07
    2001   84       56299562.0   55878421.0      1.002540  1.011036       5.644257e+07   5.649507e+07
    2002   72       58592712.0   57807215.0      1.005300  1.020604       5.890327e+07   5.899830e+07
    2003   60       57565344.0   55930654.0      1.009908  1.039752       5.813573e+07   5.815399e+07
    2004   48       56976657.0   53774672.0      1.021554  1.085341       5.820476e+07   5.836386e+07
    2005   36       56786410.0   50644994.0      1.049493  1.184218       5.959697e+07   5.997474e+07
    2006   24       54641339.0   43606497.0      1.108139  1.404485       6.055018e+07   6.124466e+07
    2007   12       48853563.0   27229969.0      1.289977  2.390688       6.301997e+07   6.509837e+07

Unfortunately this does not match the table from the text, due to rounding. We will construct a separate, rounded exhibit to reconcile to the text. 

.. doctest::

    >>> def rounded_development_summary(reported: cl.Triangle(), paid: cl.Triangle()) -> pd.DataFrame():
    ...     output = pd.DataFrame() # initializing a DataFrame
    ...     output["Reported Claims"] = reported.latest_diagonal.to_frame(origin_as_datetime=False) # using a vector of losses to anchor the exhibit index
    ...     age = reported.development.iloc[::-1] # flipping the age order
    ...     age.index = output.index # forcing the index to match
    ...     output['Age'] = age
    ...     output = output[['Age','Reported Claims']] # reordering the columns
    ...     output ['Paid Claims'] = paid.latest_diagonal.to_frame(origin_as_datetime=False) # adding in paid losses
    ...     reported_cdf = reported.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3).T
    ...     reported_cdf.index = output.index[::-1]
    ...     output["Reported CDF"] = reported_cdf
    ...     paid_cdf = paid.ldf_.round(decimals = 3).incr_to_cum().round(decimals = 3).T
    ...     paid_cdf.index = output.index[::-1]
    ...     output["Paid CDF"] = paid_cdf
    ...     output["Reported Ultimate"] = (output['Reported Claims'] * output["Reported CDF"]).round(decimals = 0) # taking a short cut to calculate the ultimate without using Chainladder 
    ...     output["Paid Ultimate"] = (output['Paid Claims'] * output["Paid CDF"]).round(decimals = 0)
    ...     return output
    >>> rounded_exhibit = rounded_development_summary(reported_selected_pattern,paid_selected_pattern)
    >>> rounded_exhibit[['Reported CDF','Paid CDF','Reported Ultimate','Paid Ultimate']] # only displaying the rounded columns
          Reported CDF  Paid CDF  Reported Ultimate  Paid Ultimate
    1998         1.000     1.002         47742304.0     47739475.0
    1999         1.000     1.004         51185767.0     51204536.0
    2000         1.001     1.006         54892767.0     54860424.0
    2001         1.003     1.011         56468461.0     56493084.0
    2002         1.006     1.020         58944268.0     58963359.0
    2003         1.011     1.040         58198563.0     58167880.0
    2004         1.023     1.085         58287120.0     58345519.0
    2005         1.051     1.184         59682517.0     59963673.0
    2006         1.110     1.404         60651886.0     61223522.0
    2007         1.292     2.390         63118803.0     65079626.0

Exhibit I Sheet 4 p109
##########################

This is another common report layout for reserving analyses. The manipulation here are more straight-forward that the previous exhibit.  

.. doctest::

    >>> def unpaid_summary(dev_sum: pd.DataFrame()) -> pd.DataFrame():
    ...     output = dev_sum.loc[:,['Reported Claims','Paid Claims','Reported Ultimate','Paid Ultimate']]
    ...     output['Case Outstanding'] = output['Reported Claims'] - output['Paid Claims']
    ...     output['Reported Method IBNR'] = output['Reported Ultimate'] - output['Reported Claims']
    ...     output['Paid Method IBNR'] = output['Paid Ultimate'] - output['Reported Claims']
    ...     output['Reported Method Unpaid'] = output['Reported Method IBNR'] + output['Case Outstanding']
    ...     output['Paid Method Unpaid'] = output['Paid Method IBNR'] + output['Case Outstanding']
    ...     return output
    >>> unpaid_exhibit = unpaid_summary(rounded_exhibit)
    >>> unpaid_exhibit[['Case Outstanding','Reported Method IBNR','Paid Method IBNR','Reported Method Unpaid','Paid Method Unpaid']]
          Case Outstanding  Reported Method IBNR  Paid Method IBNR  Reported Method Unpaid  Paid Method Unpaid
    1998           98117.0                   0.0           -2829.0                 98117.0             95288.0
    1999          185233.0                   0.0           18769.0                185233.0            204002.0
    2000          304704.0               54838.0           22495.0                359542.0            327199.0
    2001          421141.0              168899.0          193522.0                590040.0            614663.0
    2002          785497.0              351556.0          370647.0               1137053.0           1156144.0
    2003         1634690.0              633219.0          602536.0               2267909.0           2237226.0
    2004         3201985.0             1310463.0         1368862.0               4512448.0           4570847.0
    2005         6141416.0             2896107.0         3177263.0               9037523.0           9318679.0
    2006        11034842.0             6010547.0         6582183.0              17045389.0          17617025.0
    2007        21623594.0            14265240.0        16226063.0              35888834.0          37849657.0

Exhibit II Sheet 1 p110
################################

Now that we have walked through an analysis step by step, let's introduce some scaling by streamlining the entire exhibit into single function. 

.. doctest::

    >>> def dev_exhibit(tri: cl.Triangle, avg_params: dict[str,int], selected_avg: str, tail: float) -> dict[cl.Triangle()]:
    ...     print('PART 1 - Data Triangle')
    ...     print(tri)
    ...     print('PART 2 - Age-to-Age Factors')
    ...     print(tri.age_to_age)
    ...     devs = {}
    ...     print('PART 3 - Average Age-to-Age Factor')
    ...     for k,v in avg_params.items():
    ...         devs[k] = cl.Development(**v).fit_transform(tri)
    ...     def print_ldfs(ldf_dict:dict[cl.Triangle()]):
    ...         print(pd.concat([v.to_frame().rename(index={'(All)':k}) for k,v in ldf_dict.items()]))
    ...         return None
    ...     print_ldfs({k:v.ldf_.round(decimals=3) for k,v in devs.items()})
    ...     devs["Selected"] = cl.TailConstant(tail = tail, projection_period = 0).fit_transform(devs[selected_avg])
    ...     selected = {}
    ...     selected['CDF to Ultimate'] = devs["Selected"].ldf_.round(decimals=3).incr_to_cum().round(decimals=3)
    ...     selected['Percent Reported'] = (1/selected['CDF to Ultimate']).round(decimals=3)
    ...     print('PART 4 - Selected Age-to-Age Factor')
    ...     print_ldfs({'Selected':devs['Selected'].ldf_.round(decimals=3)})
    ...     print_ldfs(selected)
    ...     return devs
    >>> import re
    >>> tri = cl.load_sample('friedland_xyz_auto_bi')
    >>> assumptions_list = ['simple_5','simple_3','simple_2','volume_4','volume_3','volume_2','geometric_3']
    >>> assumptions = {x:{'n_periods':int(re.match(r'.+_(.+)', x).group(1)),'average':re.match(r'(.+)_', x).group(1)} for x in assumptions_list}
    >>> assumptions['medial 5x1'] = {'n_periods':5, 'average':'simple','drop_high':1, 'drop_low':1}
    >>> reported_devs = dev_exhibit(tri['Reported Claims'],assumptions,'volume_2',1)
    PART 1 - Data Triangle
              12       24       36       48       60       72       84       96       108      120      132
    1998      NaN      NaN  11171.0  12380.0  13216.0  14067.0  14688.0  16366.0  16163.0  15835.0  15822.0
    1999      NaN  13255.0  16405.0  19639.0  22473.0  23764.0  25094.0  24795.0  25071.0  25107.0      NaN
    2000  15676.0  18749.0  21900.0  27144.0  29488.0  34458.0  36949.0  37505.0  37246.0      NaN      NaN
    2001  11827.0  16004.0  21022.0  26578.0  34205.0  37136.0  38541.0  38798.0      NaN      NaN      NaN
    2002  12811.0  20370.0  26656.0  37667.0  44414.0  48701.0  48169.0      NaN      NaN      NaN      NaN
    2003   9651.0  16995.0  30354.0  40594.0  44231.0  44373.0      NaN      NaN      NaN      NaN      NaN
    2004  16995.0  40180.0  58866.0  71707.0  70288.0      NaN      NaN      NaN      NaN      NaN      NaN
    2005  28674.0  47432.0  70340.0  70655.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    2006  27066.0  46783.0  48804.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    2007  19477.0  31732.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    2008  18632.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    PART 2 - Age-to-Age Factors
             12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
    1998       NaN       NaN  1.108227  1.067528  1.064392  1.044146  1.114243  0.987596  0.979707  0.999179
    1999       NaN  1.237646  1.197135  1.144305  1.057447  1.055967  0.988085  1.011131  1.001436       NaN
    2000  1.196032  1.168062  1.239452  1.086354  1.168543  1.072291  1.015048  0.993094       NaN       NaN
    2001  1.353175  1.313547  1.264295  1.286967  1.085689  1.037834  1.006668       NaN       NaN       NaN
    2002  1.590040  1.308591  1.413078  1.179122  1.096524  0.989076       NaN       NaN       NaN       NaN
    2003  1.760957  1.786055  1.337353  1.089595  1.003210       NaN       NaN       NaN       NaN       NaN
    2004  2.364225  1.465057  1.218140  0.980211       NaN       NaN       NaN       NaN       NaN       NaN
    2005  1.654181  1.482965  1.004478       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    2006  1.728479  1.043199       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    2007  1.629204       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    PART 3 - Average Age-to-Age Factor
                 12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    simple_5     1.827  1.417  1.247  1.124  1.082  1.040  1.031   0.997    0.991    0.999
    simple_3     1.671  1.330  1.187  1.083  1.062  1.033  1.003   0.997    0.991    0.999
    simple_2     1.679  1.263  1.111  1.035  1.050  1.013  1.011   1.002    0.991    0.999
    volume_4     1.802  1.376  1.185  1.094  1.081  1.033  1.019   0.998    0.993    0.999
    volume_3     1.674  1.325  1.147  1.060  1.060  1.028  1.005   0.998    0.993    0.999
    volume_2     1.687  1.265  1.102  1.020  1.050  1.010  1.011   1.000    0.993    0.999
    geometric_3  1.670  1.314  1.178  1.080  1.061  1.033  1.003   0.997    0.991    0.999
    medial 5x1   1.715  1.419  1.273  1.118  1.080  1.046  1.011   0.993    0.991    0.999
    PART 4 - Selected Age-to-Age Factor
              12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132  132-144
    Selected  1.687  1.265  1.102   1.02   1.05   1.01  1.011     1.0    0.993    0.999      1.0
                      12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult  132-Ult
    CDF to Ultimate    2.551   1.512   1.196   1.085   1.064   1.013   1.003   0.992    0.992    0.999      1.0
    Percent Reported   0.392   0.661   0.836   0.922   0.940   0.987   0.997   1.008    1.008    1.001      1.0

Exhibit II Sheet 2 p111
##########################

.. doctest::

    >>> paid_devs = dev_exhibit(tri['Paid Claims'],assumptions,'volume_2',1.01)
    PART 1 - Data Triangle
             12       24       36       48       60       72       84       96       108      120      132
    1998     NaN      NaN   6309.0   8521.0  10082.0  11620.0  13242.0  14419.0  15311.0  15764.0  15822.0
    1999     NaN   4666.0   9861.0  13971.0  18127.0  22032.0  23511.0  24146.0  24592.0  24817.0      NaN
    2000  1302.0   6513.0  12139.0  17828.0  24030.0  28853.0  33222.0  35902.0  36782.0      NaN      NaN
    2001  1539.0   5952.0  12319.0  18609.0  24387.0  31090.0  37070.0  38519.0      NaN      NaN      NaN
    2002  2318.0   7932.0  13822.0  22095.0  31945.0  40629.0  44437.0      NaN      NaN      NaN      NaN
    2003  1743.0   6240.0  12683.0  22892.0  34505.0  39320.0      NaN      NaN      NaN      NaN      NaN
    2004  2221.0   9898.0  25950.0  43439.0  52811.0      NaN      NaN      NaN      NaN      NaN      NaN
    2005  3043.0  12219.0  27073.0  40026.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    2006  3531.0  11778.0  22819.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    2007  3529.0  11865.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    2008  3409.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
    PART 2 - Age-to-Age Factors
             12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
    1998       NaN       NaN  1.350610  1.183194  1.152549  1.139587  1.088884  1.061863  1.029587  1.003679
    1999       NaN  2.113373  1.416793  1.297473  1.215425  1.067130  1.027009  1.018471  1.009149       NaN
    2000  5.002304  1.863811  1.468655  1.347880  1.200707  1.151423  1.080669  1.024511       NaN       NaN
    2001  3.867446  2.069724  1.510593  1.310495  1.274860  1.192345  1.039088       NaN       NaN       NaN
    2002  3.421915  1.742562  1.598539  1.445802  1.271842  1.093726       NaN       NaN       NaN       NaN
    2003  3.580034  2.032532  1.804936  1.507295  1.139545       NaN       NaN       NaN       NaN       NaN
    2004  4.456551  2.621742  1.673950  1.215751       NaN       NaN       NaN       NaN       NaN       NaN
    2005  4.015445  2.215648  1.478447       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    2006  3.335599  1.937426       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    2007  3.362142       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
    PART 3 - Average Age-to-Age Factor
                 12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132
    simple_5     3.750  2.110  1.613  1.365  1.220  1.129  1.059   1.035    1.019    1.004
    simple_3     3.571  2.258  1.652  1.390  1.229  1.146  1.049   1.035    1.019    1.004
    simple_2     3.349  2.077  1.576  1.362  1.206  1.143  1.060   1.021    1.019    1.004
    volume_4     3.713  2.206  1.615  1.342  1.218  1.128  1.056   1.030    1.017    1.004
    volume_3     3.550  2.238  1.619  1.349  1.222  1.141  1.051   1.030    1.017    1.004
    volume_2     3.349  2.079  1.574  1.316  1.203  1.136  1.059   1.022    1.017    1.004
    geometric_3  3.558  2.241  1.647  1.384  1.227  1.145  1.049   1.035    1.019    1.004
    medial 5x1   3.653  2.062  1.594  1.368  1.229  1.128  1.060   1.025    1.019    1.004
    PART 4 - Selected Age-to-Age Factor
              12-24  24-36  36-48  48-60  60-72  72-84  84-96  96-108  108-120  120-132  132-144
    Selected  3.349  2.079  1.574  1.316  1.203  1.136  1.059   1.022    1.017    1.004     1.01
                      12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult  132-Ult
    CDF to Ultimate   21.999   6.569   3.160   2.007   1.525   1.268   1.116   1.054    1.031    1.014     1.01
    Percent Reported   0.045   0.152   0.316   0.498   0.656   0.789   0.896   0.949    0.970    0.986     0.99

Exhibit II Sheet 3 p112
##########################

.. doctest::

    >>> exhibit = rounded_development_summary(reported_devs["Selected"],paid_devs["Selected"])
    >>> exhibit
          Age  Reported Claims  Paid Claims  Reported CDF  Paid CDF  Reported Ultimate  Paid Ultimate
    1998  132          15822.0      15822.0         1.000     1.010            15822.0        15980.0
    1999  120          25107.0      24817.0         0.999     1.014            25082.0        25164.0
    2000  108          37246.0      36782.0         0.992     1.031            36948.0        37922.0
    2001   96          38798.0      38519.0         0.992     1.054            38488.0        40599.0
    2002   84          48169.0      44437.0         1.003     1.116            48314.0        49592.0
    2003   72          44373.0      39320.0         1.013     1.268            44950.0        49858.0
    2004   60          70288.0      52811.0         1.064     1.525            74786.0        80537.0
    2005   48          70655.0      40026.0         1.085     2.007            76661.0        80332.0
    2006   36          48804.0      22819.0         1.196     3.160            58370.0        72108.0
    2007   24          31732.0      11865.0         1.512     6.569            47979.0        77941.0
    2008   12          18632.0       3409.0         2.551    21.999            47530.0        74995.0

Exhibit II Sheet 4 p113
##########################

.. doctest::

    >>> unpaid_exhibit = unpaid_summary(exhibit)
    >>> unpaid_exhibit[['Case Outstanding','Reported Method IBNR','Paid Method IBNR','Reported Method Unpaid','Paid Method Unpaid']]
          Case Outstanding  Reported Method IBNR  Paid Method IBNR  Reported Method Unpaid  Paid Method Unpaid
    1998               0.0                   0.0             158.0                     0.0               158.0
    1999             290.0                 -25.0              57.0                   265.0               347.0
    2000             464.0                -298.0             676.0                   166.0              1140.0
    2001             279.0                -310.0            1801.0                   -31.0              2080.0
    2002            3732.0                 145.0            1423.0                  3877.0              5155.0
    2003            5053.0                 577.0            5485.0                  5630.0             10538.0
    2004           17477.0                4498.0           10249.0                 21975.0             27726.0
    2005           30629.0                6006.0            9677.0                 36635.0             40306.0
    2006           25985.0                9566.0           23304.0                 35551.0             49289.0
    2007           19867.0               16247.0           46209.0                 36114.0             66076.0
    2008           15223.0               28898.0           56363.0                 44121.0             71586.0

Exhibit III Sheet 1 p114
##########################

WIP
