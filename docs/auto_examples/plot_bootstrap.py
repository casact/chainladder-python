"""
======================
ODP Bootstrap Example
======================

This example demonstrates how you can can use the Overdispersed Poisson
Bootstrap sampler and get various properties about parameter uncertainty.
"""
import chainladder as cl
import matplotlib.pyplot as plt

#  Grab a Triangle
tri = cl.load_sample('genins')
# Generate bootstrap samples
sims = cl.BootstrapODPSample().fit_transform(tri)
# Calculate LDF for each simulation
sim_ldf = cl.Development().fit(sims).ldf_

# Plot the Data
fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(ncols=2, nrows=2, figsize=(10,10))
# Plot 1
tri.T.plot(ax=ax00, grid=True).set(title='Raw Data', xlabel='Development', ylabel='Incurred')
# Plot 2
sims.mean().T.plot(ax=ax01, grid=True).set(title='Mean Simulation', xlabel='Development')
# Plot 3
sim_ldf.T.plot(legend=False, color='lightgray', ax=ax10, grid=True).set(
    title='Simulated LDF', xlabel='Development', ylabel='LDF')
cl.Development().fit(tri).ldf_.drop_duplicates().T.plot(
    legend=False, color='red', ax=ax10, grid=True)
# Plot 4
sim_ldf.T.loc['12-24'].plot(
    kind='hist', bins=50, alpha=0.5, ax=ax11 , grid=True).set(
    title='Age 12-24 LDF Distribution', xlabel='LDF');
