"""
====================================
Munich Chainladder Correlation Plots
====================================

This example demonstrates how to recreate the the residual correlation plots
of the Munich Chainladder paper.
"""
import chainladder as cl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fit Munich Model
mcl = cl.load_sample('mcl')
model = cl.MunichAdjustment([('paid', 'incurred')]).fit(mcl)

# Plot Data
fig, ((ax0, ax1)) = plt.subplots(ncols=2, figsize=(15,5))

# Paid lambda line
pd.DataFrame(
    {'(P/I)': np.linspace(-2,2,2),
     'P': np.linspace(-2,2,2)*model.lambda_.loc['paid']}).plot(
    x='(P/I)', y='P', legend=False, ax=ax0)

# Paid scatter
paid_plot = pd.concat(
    (model.resids_['paid'].melt(value_name='P')['P'],
     model.q_resids_['paid'].melt(value_name='(P/I)')['(P/I)']),
    axis=1).plot(
        kind='scatter', y='P', x='(P/I)', ax=ax0,
        xlim=(-2,2), ylim=(-2,2), grid=True, title='Paid')

# Incurred lambda line
inc_lambda = pd.DataFrame(
    {'(I/P)': np.linspace(-2,2,2),
     'I': np.linspace(-2,2,2)*model.lambda_.loc['incurred']})
inc_lambda.plot(x='(I/P)', y='I', ax=ax1, legend=False);

# Incurred scatter
incurred_plot = pd.concat(
    (model.resids_['incurred'].melt(value_name='I')['I'],
     model.q_resids_['incurred'].melt(value_name='(I/P)')['(I/P)']),
    axis=1).plot(
        kind='scatter', y='I', x='(I/P)', ax=ax1,
        xlim=(-2,2), ylim=(-2,2), grid=True, title='Incurred');
fig.suptitle("Munich Chainladder Residual Correlations");
