"""
=================
Exposure Triangle
=================

Although triangles have both origin and development attributes, it is often
convenient to create premium or exposure vectors that can work with loss
triangles.  The `Triangle` class treats the development parameter as
optional. This example instantiates a 'premium' triangle as a single vector.
"""

import chainladder as cl
import pandas as pd

import chainladder as cl

# Raw premium data in pandas
premium_df = pd.DataFrame(
    {'AccYear':[item for item in range(1977, 1988)],
     'premium': [3000000]*11})

# Create a premium 'triangle' with no development
premium = cl.Triangle(premium_df, origin='AccYear', columns='premium')

# Create some loss triangle
loss = cl.load_sample('abc')
ultimate = cl.Chainladder().fit(loss).ultimate_

# Plot
(ultimate / premium).plot(
    kind='area', title='Loss Ratio by Accident Year',
    alpha=0.7, color='darkgreen', legend=False, grid=True).set(
    xlabel='Accident Year', ylabel='Loss Ratio');
