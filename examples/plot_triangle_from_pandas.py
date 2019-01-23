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
data = pd.read_csv(f'https://www.casact.org/research/reserve_data/wkcomp_pos.csv')
data = data[data['DevelopmentYear']<=1997]

# Create a triangle
triangle = cl.Triangle(data, origin='AccidentYear',
                       development='DevelopmentYear',
                       index=['GRNAME'],
                       columns=['IncurLoss_D','CumPaidLoss_D','EarnedPremDIR_D'])

# Output
print('Raw data:')
print(data.head())
print()
print('Triangle summary:')
print(triangle)
print()
print('Aggregate Paid Triangle:')
print(triangle['CumPaidLoss_D'].sum())


plot_data = triangle['CumPaidLoss_D'].sum().to_frame().unstack().reset_index()
plot_data.columns = ['Development Period', 'Accident Year', 'Cumulative Paid Loss']

sns.set_style('whitegrid')
plt.title('CAS Loss Reserve Database: Workers'' Compensation')
g = sns.pointplot(x='Development Period', y='Cumulative Paid Loss',
                  hue='Accident Year', data=plot_data, markers='.')
