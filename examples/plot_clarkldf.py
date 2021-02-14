"""
====================
Clark Growth Curves
====================

This example demonstrates one of the attributes of the :class:`ClarkLDF`. We can
use the growth curve ``G_`` to estimate the percent of ultimate at any given
age.
"""
import chainladder as cl
import numpy as np

# Grab Industry triangles
clrd = cl.load_sample('clrd').groupby('LOB').sum()

# Fit Clark Cape Cod method
model = cl.ClarkLDF(growth='loglogistic').fit(
    clrd['CumPaidLoss'],
    sample_weight=clrd['EarnedPremDIR'].latest_diagonal)

# sample ages
ages = np.linspace(1, 300, 30)

# Plot results
model.G_(ages).T.plot(
    title='Loglogistic Growth Curves', grid=True).set(
    xlabel='Age', ylabel='% of Ultimate');
