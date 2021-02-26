"""
===========================
GLM Reserving
===========================

This example demonstrates how you can use the `DevelopmentML` estimator to incorporate
a Tweedie GLM into ``chainladder``.  The specific case of the Over-dispersed
poisson GLM fit to incremental paids.  It is further shown to match the basic
chainladder `Development` estimator as described by England and Verall.

"""

import chainladder as cl
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor

genins = cl.load_sample('genins')

# Create a sklearn transformer that preps the X Matrix
prep_X = ColumnTransformer(transformers=[
    ('dummy', OneHotEncoder(drop='first'), ['origin', 'development']),
])

# Create a sklearn Pipeline for a Model fit
estimator_ml=Pipeline(steps=[
        ('prep X', prep_X),
        ('model', TweedieRegressor(link='log', power=1))],)

dev = cl.DevelopmentML(
    estimator_ml=estimator_ml, y_ml='values',
    fit_incrementals=True).fit(genins)

glm = dev.ldf_.iloc[..., 0, :].T.iloc[:, 0].rename('GLM')
traditional = cl.Development().fit(genins).ldf_.T.iloc[:, 0].rename('Traditional')

pd.concat((glm,traditional), 1).plot(kind='bar', title='LDF: Poisson GLM vs Traditional');
