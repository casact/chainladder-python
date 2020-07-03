"""
====================================
Picking Bornhuetter-Ferguson Apriori
====================================

This example demonstrates how you can can use the output of one method as the
apriori selection for the Bornhuetter-Ferguson Method.
"""
import chainladder as cl
import pandas as pd

# Create Aprioris as the mean AY chainladder ultimate
raa = cl.load_sample('RAA')
cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
apriori = cl_ult*0+(cl_ult.sum()/10)  # Mean Chainladder Ultimate
bf_ult = cl.BornhuetterFerguson(apriori=1).fit(raa, sample_weight=apriori).ultimate_

# Plot of Ultimates
pd.concat(
    (cl_ult.to_frame().rename({9999: 'Chainladder'}, axis=1),
     bf_ult.to_frame().rename({9999: 'BornhuetterFerguson'}, axis=1)),
    axis=1).plot(grid=True, marker='o').set(
    xlabel='Accident Year', ylabel='Ultimate');
