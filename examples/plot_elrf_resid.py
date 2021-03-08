"""
==========================================
Extended Link Ratio Family Residuals
==========================================

This example replicates the diagnostic residuals from Barnett and Zehnwirth's
"Best Estimates for Reserves" paper in which they describe the Extended Link
Ratio Family (ELRF) model.  This `Development` estimator is based on the ELRF
model.

The weighted standardized residuals are contained in the ``std_residuals_``
property of the fitted estimator.  Using these, we can replicate Figure 2.6
from the paper.

"""
import chainladder as cl
import pandas as pd
import matplotlib.pyplot as plt

raa = cl.load_sample('raa')
model = cl.Development().fit(raa)

fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(ncols=2, nrows=2, figsize=(13,10))
model.std_residuals_.T.plot(
    style='.', color='gray', legend=False, grid=True, ax=ax00,
    xlabel='Development Month', ylabel='Weighted Standardized Residuals')
model.std_residuals_.mean('origin').T.plot(
    color='red', legend=False, grid=True, ax=ax00)
model.std_residuals_.plot(
    style='.', color='gray', legend=False, grid=True, ax=ax01, xlabel='Origin Period')
model.std_residuals_.mean('development').plot(
    color='red', legend=False, grid=True, ax=ax01)
model.std_residuals_.dev_to_val().T.plot(
    style='.', color='gray', legend=False, grid=True, ax=ax10,
    xlabel='Valuation Date', ylabel='Weighted Standardized Residuals')
model.std_residuals_.dev_to_val().mean('origin').T.plot(color='red', legend=False, grid=True, ax=ax10)
pd.concat((
    (raa[raa.valuation<raa.valuation_date]*model.ldf_.values).unstack().rename('Fitted Values'),
    model.std_residuals_.unstack().rename('Residual')), 1).dropna().plot(
    kind='scatter', marker='o', color='gray', x='Fitted Values', y='Residual', ax=ax11, grid=True, sharey=True)
fig.suptitle("Barnett Zehnwirth\nStandardized residuals of the Extended Link Ratio Family (ELRF)\n(Fig 2.6)");
