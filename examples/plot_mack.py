"""
========================
Mack Chainladder Example
========================

This example demonstrates how you can can use the Mack Chainladder method.
"""
import pandas as pd
import chainladder as cl

# Load the data
data = cl.load_sample('raa')

# Compute Mack Chainladder ultimates and Std Err using 'simple' average
mack = cl.MackChainladder()
dev = cl.Development(average='volume')
mack.fit(dev.fit_transform(data))

# Plotting
plot_data = mack.summary_.to_frame()
g = plot_data[['Latest', 'IBNR']].plot(
    kind='bar', stacked=True, ylim=(0, None), grid=True,
    yerr=pd.DataFrame({'latest': plot_data['Mack Std Err']*0,
                       'IBNR': plot_data['Mack Std Err']}),
    title='Mack Chainladder Ultimate').set(
    xlabel='Accident Year', ylabel='Loss');
