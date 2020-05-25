"""
================================
Pandas-style slicing of Triangle
================================

This example demonstrates the familiarity of the pandas API applied to a
:class:`Triangle` instance.

"""
import chainladder as cl

# The base Triangle Class:
cl.Triangle

# Load data
clrd = cl.load_sample('clrd')
# pandas-style Aggregations
clrd = clrd.groupby('LOB').sum()
# pandas-style value/column slicing
clrd = clrd['CumPaidLoss']
# pandas loc-style index slicing
clrd = clrd.loc['medmal']

# Plot
g = clrd.link_ratio.plot(
    marker='o', grid=True,
    title='Medical Malpractice Link Ratios').set(
    ylabel='Link Ratio', xlabel='Accident Year');
