"""
====================================================================
Benktander: Relationship between Chainladder and BornhuetterFerguson
====================================================================

This example demonstrates the relationship between the Chainladder and
BornhuetterFerguson methods by way fo the Benktander model. Each is a
special case of the Benktander model where ``n_iters = 1`` for BornhuetterFerguson
and as ``n_iters`` approaches infinity yields the chainladder.  As ``n_iters``
increases the apriori selection becomes less relevant regardless of initial
choice.
"""
import chainladder as cl

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

# Load Data
clrd = cl.load_dataset('clrd')
medmal_paid = clrd.groupby('LOB').sum().loc['medmal']['CumPaidLoss']
medmal_prem = clrd.groupby('LOB').sum().loc['medmal']['EarnedPremDIR'].latest_diagonal
medmal_prem.rename(development='premium')

# Generate LDFs and Tail Factor
medmal_paid = cl.Development().fit_transform(medmal_paid)
medmal_paid = cl.TailCurve().fit_transform(medmal_paid)

# Benktander Model
benk = cl.Benktander()

# Prep Benktander Grid Search with various assumptions, and a scoring function
param_grid = dict(n_iters=list(range(1,100,2)),
                  apriori=[0.50, 0.75, 1.00])
scoring = {'IBNR':lambda x: x.ibnr_.sum()[0]}
grid = cl.GridSearch(benk, param_grid, scoring=scoring)
# Perform Grid Search
grid.fit(medmal_paid, sample_weight=medmal_prem)

# Plot data
grid.results_.pivot(index='n_iters', columns='apriori', values='IBNR').plot()
plt.title('Benktander convergence to Chainladder')
g = plt.ylabel('IBNR')
