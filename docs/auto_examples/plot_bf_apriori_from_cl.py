"""
====================================
Picking Bornhuetter-Ferguson Apriori
====================================

This example demonstrates how you can can use the output of one method as the
apriori selection for the Bornhuetter-Ferguson Method.
"""
import chainladder as cl
import seaborn as sns
sns.set_style('whitegrid')

# Create Aprioris as the mean AY chainladder ultimate
raa = cl.load_dataset('RAA')
cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
apriori = cl_ult*0+(cl_ult.sum()/10)[0]  # Mean Chainladder Ultimate
bf_ult = cl.BornhuetterFerguson(apriori=1).fit(raa, sample_weight=apriori).ultimate_

# Plot of Ultimates
plot_data = cl_ult.to_frame().rename({'Ultimate': 'Chainladder'}, axis=1)
plot_data['BornhuetterFerguson'] = bf_ult.to_frame()
plot_data = plot_data.stack().reset_index()
plot_data.columns = ['Accident Year', 'Method', 'Ultimate']
plot_data['Accident Year'] = plot_data['Accident Year'].dt.year

g = sns.lineplot(data=plot_data, x='Accident Year', y='Ultimate', hue='Method')
