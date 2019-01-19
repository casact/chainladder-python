import copy
import numpy as np
from sklearn.base import BaseEstimator


class TailBase(BaseEstimator):
    ''' Base class for all tail methods.  Tail objects are equivalent
        to development objects with an additional set of tail statistics'''

    def __init__(self):
        pass

    def fit(self, X, y=None, sample_weight=None):
        if X.__dict__.get('ldf_', None) is None:
            raise ValueError('Triangle must have LDFs.')
        self.ldf_ = copy.deepcopy(X.ldf_)
        tail = np.ones(self.ldf_.shape)[..., -1:]
        self.ldf_.triangle = np.concatenate((self.ldf_.triangle, tail), -1)
        self.sigma_ = copy.deepcopy(X.sigma_)
        self.std_err_ = copy.deepcopy(X.std_err_)
        zeros = tail*0
        self.sigma_.triangle = np.concatenate((self.sigma_.triangle, zeros), -1)
        self.std_err_.triangle = np.concatenate((self.std_err_.triangle, zeros), -1)
        ddims = np.append(self.ldf_.ddims, [f'{int(X.ddims[-1])}-Ult'])
        self.ldf_.ddims = self.sigma_.ddims = self.std_err_.ddims = ddims
        return self

    def transform(self, X):
        X.std_err_.triangle = np.concatenate((X.std_err_.triangle, self.std_err_.triangle[..., -1:]), -1)
        X.cdf_.triangle = np.concatenate((X.cdf_.triangle, self.cdf_.triangle[..., -1:]*0+1), -1)
        X.cdf_.triangle = X.cdf_.triangle*self.cdf_.triangle[..., -1:]
        X.ldf_.triangle = np.concatenate((X.ldf_.triangle, self.ldf_.triangle[..., -1:]), -1)
        X.sigma_.triangle = np.concatenate((X.sigma_.triangle, self.sigma_.triangle[..., -1:]), -1)
        X.cdf_.ddims = X.ldf_.ddims = X.sigma_.ddims = X.std_err_.ddims = self.ldf_.ddims
        return X

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X)
        return self.transform(X)

    @property
    def cdf_(self):
        if self.__dict__.get('ldf_', None) is None:
            return
        else:
            obj = copy.deepcopy(self.ldf_)
            cdf_ = np.flip(np.cumprod(np.flip(obj.triangle, -1), -1), -1)
            obj.triangle = cdf_
            return obj
