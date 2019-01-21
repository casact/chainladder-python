"""
====================================================
Basic Assumption Tuning with Pipeline and Gridsearch
====================================================

This example demonstrates testing multiple number of periods in the development
transformer to see its influence on the overall ultimate estimate.
"""

import seaborn as sns
sns.set_style('whitegrid')

import chainladder as cl

tri = cl.load_dataset('abc')

# Set up Pipeline
steps = [('dev',cl.Development()),
         ('chainladder',cl.Chainladder())]
params = dict(dev__n_periods=[item for item in range(2,11)])
pipe = cl.Pipeline(steps=steps)

# Develop scoring function that returns an Ultimate/Incurred Ratio
scoring = lambda x: x.named_steps.chainladder.ultimate_.sum()[0] / tri.latest_diagonal.sum()[0]

# Run GridSearch
grid = cl.GridSearch(pipe, params, scoring).fit(tri)

# Plot Results
g = sns.pointplot(data=grid.results_,x='dev__n_periods',y='score')
