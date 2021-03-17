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

# Read in the data
data = pd.read_csv(r'https://raw.githubusercontent.com/casact/chainladder-python/master/chainladder/utils/data/clrd.csv')

# Create a triangle
triangle = cl.Triangle(
    data, origin='AccidentYear', development='DevelopmentYear',
    index=['GRNAME'], columns=['IncurLoss','CumPaidLoss','EarnedPremDIR'])

# Output
print('Raw data:')
print(data.head())
print()
print('Triangle summary:')
print(triangle)
print()
print('Aggregate Paid Triangle:')
print(triangle['CumPaidLoss'].sum())

# Plot data
triangle['CumPaidLoss'].sum().T.plot(
    marker='.', grid=True,
    title='CAS Loss Reserve Database: Workers Compensation').set(
    xlabel='Development Period', ylabel='Cumulative Paid Loss');
