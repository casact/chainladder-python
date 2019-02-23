"""
================================
Pandas-style slicing of Triangle
================================

This example demonstrates the familiarity of the pandas API applied to a
:class:`Triangle` instance.

"""
import chainladder as cl
import seaborn as sns
sns.set_style('whitegrid')

# The base Triangle Class:
cl.Triangle

# Load data
clrd = cl.load_dataset('clrd')
# pandas-style Aggregations
clrd = clrd.groupby('LOB').sum()
# pandas-style value/column slicing
clrd = clrd['CumPaidLoss']
# pandas loc-style index slicing
clrd = clrd.loc['medmal']

# Plot
g = clrd.link_ratio.plot(marker='o') \
        .set(title='Medical Malpractice Link Ratios',
             ylabel='Link Ratio', xlabel='Accident Year')
