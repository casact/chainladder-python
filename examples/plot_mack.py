"""
========================
Mack Chainladder Example
========================

This example demonstrates how you can can use the Mack Chainladder method.
"""
import pandas as pd
import chainladder as cl
import seaborn as sns
sns.set_style('whitegrid')

# Load the data
data = cl.load_dataset('raa')

# Compute Mack Chainladder ultimates and Std Err using 'simple' average
mack = cl.MackChainladder()
dev = cl.Development(average='simple')
mack.fit(dev.fit_transform(data))

# Plotting
plot_data = mack.summary_.to_frame()
g = plot_data[['Latest', 'IBNR']] \
    .plot(kind='bar', stacked=True,
          yerr=pd.DataFrame({'latest': plot_data['Mack Std Err']*0,
                             'IBNR': plot_data['Mack Std Err']}),
          ylim=(0, None), title='Mack Chainladder Ultimate')
g.set_xlabel('Accident Year')
_ = g.set_ylabel('Loss')
