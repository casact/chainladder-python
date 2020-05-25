"""
=========================
Munich Adjustment Example
=========================

This example demonstrates how to adjust LDFs by the relationship between Paid
and Incurred using the MunichAdjustment.
.
"""

import chainladder as cl
import pandas as pd
import matplotlib.pyplot as plt

# Load data
mcl = cl.load_sample('mcl')
# Volume weighted (default) LDFs
dev = cl.Development().fit_transform(mcl)
# Traditional Chainladder
cl_traditional = cl.Chainladder().fit(dev).ultimate_
# Munich Adjustment
dev_munich = cl.MunichAdjustment(paid_to_incurred={'paid':'incurred'}).fit_transform(dev)
cl_munich = cl.Chainladder().fit(dev_munich).ultimate_

# Plot data
fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, figsize=(10,5))
plot_kw = dict(kind='bar', grid=True, color=('blue', 'green'), alpha=0.7)

plot1_data = cl_munich.to_frame().T.rename(
    {'incurred':'Ultimate Incurred', 'paid': 'Ultimate Paid'}, axis=1)

plot2_data = pd.concat(
    ((cl_munich['paid'] / cl_munich['incurred']).rename(
        'columns', ['Munich']).to_frame(),
     (cl_traditional['paid'] / cl_traditional['incurred']).rename(
         'columns', ['Traditional']).to_frame()), axis=1)

plot1_data.plot(
    title='Munich Chainladder', ax=ax0, **plot_kw).set(
    ylabel='Ultimate', xlabel='Accident Year')
plot2_data.plot(
    title='P/I Ratio Comparison', ax=ax1, ylim=(0,1.25), **plot_kw).set(
    ylabel='Paid Ultimate / Incurred Ultimate', xlabel='Accident Year');
