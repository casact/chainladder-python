"""
==========================================
Probabilistic Trend Family Residuals
==========================================

This example replicates the diagnostic residuals from Barnett and Zehnwirth's
"Best Estimates for Reserves" paper in which they describe the Probabilistic
Trend Family (PTF) model.  With the "ABC" triangle, they show that the basic
chainladder, which ignores trend along the valuation axis, fails to have iid
weighted standardized residuals along the valuation of the Triangle.

We fit a "diagnostic" model that deliberately ignores modeling the valuation
vector.  This is done by specifying the patsy formula ``C(origin)+C(development)``
which fits origin and development as categorical features.
"""
import chainladder as cl
import pandas as pd
import matplotlib.pyplot as plt

abc = cl.load_sample('abc')
model = cl.BarnettZehnwirth(formula='C(origin)+C(development)').fit(abc)

fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(ncols=2, nrows=2, figsize=(13,10))
model.std_residuals_.T.plot(
    style='.', color='gray', legend=False, grid=True, ax=ax00,
    xlabel='Development Month', ylabel='Weighted Standardized Residuals')
model.std_residuals_.plot(
    style='.', color='gray', legend=False, grid=True, ax=ax01, xlabel='Origin Period')
model.std_residuals_.dev_to_val().T.plot(
    style='.', color='gray', legend=False, grid=True, ax=ax10,
    xlabel='Valuation Date', ylabel='Weighted Standardized Residuals')
model.std_residuals_.dev_to_val().mean('origin').T.plot(
    color='red', legend=False, grid=True, ax=ax10)
pd.concat((
    model.triangle_ml_[model.triangle_ml_.valuation<=abc.valuation_date].log().unstack().rename('Fitted Values'),
    model.std_residuals_.unstack().rename('Residual')), 1).dropna().plot(
    kind='scatter', marker='o', color='gray', x='Fitted Values', y='Residual', ax=ax11, grid=True, sharey=True)
fig.suptitle("Barnett Zehnwirth\nStandardized residuals of the statistical chainladder model\n(Fig 3.11)");
