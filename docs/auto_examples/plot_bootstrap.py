"""
======================
ODP Bootstrap Example
======================

This example demonstrates how you can can use the Overdispersed Poisson
Bootstrap sampler and get various properties about parameter uncertainty.
"""
import chainladder as cl

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


#  Grab a Triangle
tri = cl.load_dataset('bs_sample')
# Generate bootstrap samples
sims = cl.BootstrapODPSample().fit_transform(tri)
# Calculate LDF for each simulation
sim_ldf = cl.Development().fit(sims).ldf_
sim_ldf = sim_ldf[sim_ldf.origin==sim_ldf.origin.max()]

# Plot the Data
fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
tri.T.plot(ax=ax00).set(title='Raw Data', xlabel='Development', ylabel='Incurred')
sims.mean().T.plot(ax=ax01).set(title='Mean Simulation', xlabel='Development', ylabel='Incurred')
sim_ldf.T.plot(legend=False, color='lightgray', ax=ax10) \
       .set(title='Simulated LDF', xlabel='Development', ylabel='LDF')
cl.Development().fit(tri).ldf_.drop_duplicates().T \
                .plot(legend=False, color='red', ax=ax10)
_ = sim_ldf.T.loc['12-24'].plot(kind='hist', bins=50, alpha=0.5, ax=ax11) \
           .set(title='Age 12-24 LDF Distribution', xlabel='LDF')
