"""
=========================
Munich Adjustment Example
=========================

This example demonstrates how to adjust LDFs by the relationship between Paid
and Incurred using the MunichAdjustment.
.
"""

import chainladder as cl
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_palette('muted')

# Load data
mcl = cl.load_dataset('mcl')
# Volume weighted (default) LDFs
dev = cl.Development().fit_transform(mcl)
# Traditional Chainladder
cl_traditional = cl.Chainladder().fit(dev)
# Munich Adjustment
dev_munich = cl.MunichAdjustment(paid_to_incurred={'paid':'incurred'}).fit_transform(dev)
cl_munich = cl.Chainladder().fit(dev_munich)

# Plot data
fig, (ax, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(10,5))
plot1_data = cl_munich.ultimate_['paid'].to_frame()
plot1_data.columns = ['Paid Ultimate']
plot1_data['Incurred Ultimate'] = cl_munich.ultimate_['incurred'].to_frame()
plot2_data = (cl_munich.ultimate_['paid']/cl_munich.ultimate_['incurred']).to_frame()
plot2_data.columns = ['Munich']
plot2_data['Traditional'] = (cl_traditional.ultimate_['paid']/cl_traditional.ultimate_['incurred']).to_frame()
plot1_data.plot(kind='bar', ax=ax)
ax.set_ylabel('Ultimate')
ax.set_xlabel('Accident Year')
ax.set_title('Munich Chainladder')
plot2_data.plot(kind='bar', ax=ax2, ylim=(0,1.25))
ax2.set_title('P/I Ratio Comparison')
ax2.set_xlabel('Accident Year')
g = plt.ylabel('Paid Ultimate / Incurred Ultimate')
