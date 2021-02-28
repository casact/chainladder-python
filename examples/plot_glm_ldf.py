"""
===========================
GLM Reserving
===========================

This example demonstrates how you can use the `TweedieGLM` estimator to incorporate
the GLM framework into a ``chainladder`` workflow.  The specific case of the Over-dispersed
poisson GLM fit to incremental paids.  It is further shown to match the basic
chainladder `Development` estimator.

"""

import chainladder as cl
import pandas as pd

genins = cl.load_sample('genins')

# Fit an ODP GLM
dev = cl.TweedieGLM(
    design_matrix='C(development) + C(origin)',
    link='log', power=1).fit(genins)

# Grab LDFs vs traditional approach
glm = dev.ldf_.iloc[..., 0, :].T.iloc[:, 0].rename('GLM')
traditional = cl.Development().fit(genins).ldf_.T.iloc[:, 0].rename('Traditional')

# Plot data
pd.concat((glm,traditional), 1).plot(kind='bar', title='LDF: Poisson GLM vs Traditional');
