"""
============
IBNR Runoff
============

All IBNR models spin off several results triangles including ``inbr_``,
``ultimate_``, ``full_expectation``, and ``full_triangle_``.  These can be
manipulated into a variety of formats. This example demonstrates how to
create a calendar year runoff of IBNR.
"""

import chainladder as cl

# Create a triangle
triangle = cl.load_sample('GenIns')

# Fit a model
model = cl.Chainladder().fit(triangle)

# Develop IBNR runoff triangle
runoff = (model.full_triangle_.cum_to_incr() - triangle.cum_to_incr())

# Convert to calendar period and aggregate across all accident years
cal_yr_runoff = runoff[runoff.valuation>triangle.valuation_date].dev_to_val().sum(axis='origin')

# Plot results
cal_yr_runoff.dropna().T.plot(
    kind='bar', legend=False, color='red', grid=True,
    title='GenIns: IBNR Run-off', alpha=0.7).set(
    xlabel='Calendar Year', ylabel='IBNR');
