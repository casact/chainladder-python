"""
======================================
Analyzing Cape Cod Apriori Sensitivity
======================================

This example demonstrates the usage of the deterministic CapeCod method and
shows the sensitivity of the apriori expectation to various choices of ``trend``
and ``decay``.
"""

import chainladder as cl
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

# Grab data
ppauto_loss = cl.load_dataset('clrd').groupby('LOB').sum().loc['ppauto']['CumPaidLoss']
ppauto_prem = cl.load_dataset('clrd').groupby('LOB').sum() \
                .loc['ppauto']['EarnedPremDIR'].latest_diagonal.rename('development',['Premium'])

def get_apriori(decay, trend):
    """ Function to grab apriori array from cape cod method """
    cc = cl.CapeCod(decay=decay, trend=trend)
    cc.fit(ppauto_loss, sample_weight=ppauto_prem)
    return cc.detrended_apriori_.to_frame()

def get_plot_data(trend):
    """ Function to grab plot data """
    # Initial apriori DataFrame
    detrended_aprioris = get_apriori(0,trend)
    detrended_aprioris.columns=['decay: 0%']

    # Add columns to apriori DataFrame
    for item in [25, 50, 75, 100]:
        detrended_aprioris[f'decay: {item}%'] = get_apriori(item/100, trend)
    return detrended_aprioris


# Plot Data
fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(10,10))
get_plot_data(-0.05).plot(ax=ax00)
get_plot_data(-.025).plot(ax=ax01)
get_plot_data(0).plot(ax=ax10)
get_plot_data(0.025).plot(ax=ax11)
ax00.set_title('Trend: -5%')
ax01.set_title('Trend: -2.5%')
ax10.set_title('Trend: 0%')
ax11.set_title('Trend: 2.5%')
g = fig.suptitle("Private Passenger Auto Cape Cod Detrended Aprioris")
