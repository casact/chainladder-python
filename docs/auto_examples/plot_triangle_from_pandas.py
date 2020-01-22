"""
=======================
Basic Triangle Creation
=======================

This example demonstrates the typical way you'd ingest data into a Triangle.
Data in tabular form in a pandas DataFrame is required.  At a minimum, columns
specifying origin and development, and a value must be present.  Note, you can
include more than one column as a list as well as any number of indices for
creating triangle subgroups.

In this example, we create a triangle object with triangles for each company
in the CAS Loss Reserve Database for Workers' Compensation.
"""

import chainladder as cl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the data
lobs = 'wkcomp'
data = pd.read_csv(r'https://www.casact.org/research/reserve_data/wkcomp_pos.csv')
data = data[data['DevelopmentYear']<=1997]

# Create a triangle
triangle = cl.Triangle(
    data, origin='AccidentYear', development='DevelopmentYear',
    index=['GRNAME'], columns=['IncurLoss_D','CumPaidLoss_D','EarnedPremDIR_D'])

# Output
print('Raw data:')
print(data.head())
print()
print('Triangle summary:')
print(triangle)
print()
print('Aggregate Paid Triangle:')
print(triangle['CumPaidLoss_D'].sum())

# Plot data
ax = triangle['CumPaidLoss_D'].sum().T.plot(
    marker='.', title='CAS Loss Reserve Database: Workers Compensation');
ax.set(xlabel='Development Period', ylabel='Cumulative Paid Loss')

plt.show()
