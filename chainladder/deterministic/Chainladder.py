import numpy as np
import copy
from sklearn.base import BaseEstimator


class Chainladder(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        latest, cdf = X[0].latest_diagonal, X[1].latest_diagonal
        self.ultimates_ = latest * cdf
        self.ultimates_.vdims = latest.vdims
        self.ultimates_.ddims = ['Ultimate']
        return self
