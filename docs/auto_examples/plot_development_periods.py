"""
===========================
Development scenario tuning
===========================

This example demonstrates testing multiple number of periods in the development
transformer to see its influence on the overall ultimate estimate.
"""

import chainladder as cl
import seaborn as sns
import pandas as pd
sns.set_style('whitegrid')

# Loop through 2 through 10 year weighted average development
ult_ratio = {}
for n_periods in range(2, 11):
    abc = cl.load_dataset('abc')
    dev = cl.Development(n_periods=n_periods).fit_transform(abc)
    ult = cl.Chainladder().fit(abc)
    ult_ratio[n_periods] = ult.ultimate_.sum()[0]/dev.latest_diagonal.sum()[0]

# Plot the data
plot_data = pd.DataFrame([ult_ratio.keys(), ult_ratio.values()],
                         index=['n_period', 'Ultimate to Latest Factor']).T
g = sns.pointplot(data=plot_data, x='n_period', y='Ultimate to Latest Factor')
