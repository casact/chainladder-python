import numpy as np
import copy
from sklearn.base import BaseEstimator


class Chainladder(BaseEstimator):
    def fit(self, X, y=None):
        latest, cdf = X.triangle[:, :, :, 0], X.triangle[:, :, :, 1]
        obj = copy.deepcopy(X)
        obj.triangle = np.expand_dims(latest * cdf, axis=3)
        obj.ddims = ['Ultimate']
        self.ultimates_ = obj
        return self
