"""
============
IBNR Runoff
============

All IBNR models spin off several results triangles including `inbr_`,
`ultimate_`, `full_expectation`, and `full_triangle_`.  These can be
manipulated into a variety of formats. This example demonstrates how to
create a calendar year runoff of IBNR.
"""

import chainladder as cl
import seaborn as sns
sns.set_style('whitegrid')

# Create a triangle
triangle = cl.load_dataset('GenIns')

# Fit a model
model = cl.Chainladder().fit(triangle)

# Develop IBNR runoff triangle
runoff = (model.full_triangle_.cum_to_incr() - triangle.cum_to_incr())

# Convert to calendar period and aggregate across all accident years
cal_yr_runoff = runoff.dev_to_val().dropna().sum(axis='origin')

# Plot results
cal_yr_runoff.T.plot(kind='bar', legend=False, color='red',
                     title='GenIns: IBNR Run-off', alpha=0.7) \
               .set(xlabel='Calendar Year', ylabel='IBNR');
