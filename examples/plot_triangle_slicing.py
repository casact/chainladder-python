"""
================================
Pandas-style slicing of Triangle
================================

This example demonstrates the familiarity of the pandas API applied to a
:class:`Triangle` instance.

"""
import chainladder as cl
import seaborn as sns
import matplotlib.pyplot as plt
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

# Convert link ratios to dataframe
link_ratios = clrd.link_ratio.to_frame().unstack().reset_index()
link_ratios.columns = ['Age', 'Accident Year', 'Link Ratio']

# Plot
sns.pointplot(hue='Age', y='Link Ratio', x='Accident Year',
              data=link_ratios, markers='.')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
g = plt.title('Medical Malpractice Link Ratios')
