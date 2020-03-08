"""
===============================================
Testing Sensitivity of Bondy Tail Assumptions
===============================================

This example demonstrates the usage of the `TailBondy` estimator as well as
passing multiple scoring functions to `GridSearch`.
"""

import seaborn as sns
sns.set_style('whitegrid')

import chainladder as cl

# Fit basic development to a triangle
tri = cl.load_dataset('tail_sample')['paid']
dev = cl.Development(average='simple').fit_transform(tri)


# Return both the tail factor and the Bondy exponent in the scoring function
scoring = {
    'tail_factor': lambda x: x.cdf_[x.cdf_.development=='120-9999'].to_frame().values[0,0],
    'bondy_exponent': lambda x : x.b_[0,0]}

# Vary the 'earliest_age' assumption in GridSearch
param_grid=dict(earliest_age=list(range(12, 120, 12)))
grid = cl.GridSearch(cl.TailBondy(), param_grid, scoring)
results = grid.fit(dev).results_

ax = results.plot(x='earliest_age', y='bondy_exponent', title='Bondy Assumption Sensitivity')
results.plot(x='earliest_age', y='tail_factor', secondary_y=True, ax=ax);
