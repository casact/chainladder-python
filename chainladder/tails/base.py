import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class TailBase(BaseEstimator):
    ''' Base class for all tail methods.  Tail objects are equivalent
        to development objects with an additional set of tail statistics'''

    def __init__(self):
        pass

    def fit(self, X, y=None, sample_weight=None):
        if X.__dict__.get('ldf_', None) is None:
            raise ValueError('Triangle must have LDFs.')
        self._ave_period = {'Y': (1, 12), 'Q': (4, 3), 'M': (12, 1)}[X.development_grain]
        ddims = np.concatenate(
            (X.ddims, [(item+1)*self._ave_period[1] + X.ddims[-1]
                       for item in range(self._ave_period[0])], [9999]), 0)
        self.ldf_ = copy.deepcopy(X.ldf_)
        tail = np.ones(self.ldf_.shape)[..., -1:]
        tail = np.repeat(tail, self._ave_period[0]+1, -1)
        self.ldf_.triangle = np.concatenate((self.ldf_.triangle, tail), -1)
        self.ldf_.valuation = self.ldf_._valuation_triangle(ddims[1:])
        self.ldf_.ddims = np.array([f'{ddims[i]}-{ddims[i+1]}'
                                    for i in range(len(ddims)-1)])
        self.sigma_ = copy.deepcopy(X.sigma_)
        self.std_err_ = copy.deepcopy(X.std_err_)
        zeros = tail[..., -1:]*0
        self.sigma_.triangle = np.concatenate(
            (self.sigma_.triangle, zeros), -1)
        self.std_err_.triangle = np.concatenate(
            (self.std_err_.triangle, zeros), -1)
        self.sigma_.ddims = self.std_err_.ddims = \
            np.append(X.ldf_.ddims, [f'{int(X.ddims[-1])}-Ult'])
        val_array = pd.DataFrame(X.ldf_.valuation.values.reshape(X.ldf_.shape[-2:],order='f'))
        val_array[9999] = pd.to_datetime('2262-04-11')
        val_array = pd.DatetimeIndex(pd.DataFrame(val_array).unstack().values)
        self.sigma_.valuation = self.std_err_.valuation = val_array
        return self

    def transform(self, X):
        X_new = copy.deepcopy(X)
        X_new.std_err_.triangle = np.concatenate((X_new.std_err_.triangle, self.std_err_.triangle[..., -1:]), -1)
        X_new.cdf_.triangle = np.concatenate((X_new.cdf_.triangle, self.cdf_.triangle[..., -self._ave_period[0]-1:]*0+1), -1)
        X_new.cdf_.triangle = X_new.cdf_.triangle*self.cdf_.triangle[..., -self._ave_period[0]-1:-self._ave_period[0]]
        X_new.cdf_.triangle[..., -1] = self.cdf_.triangle[..., -1]
        X_new.ldf_.triangle = np.concatenate((X_new.ldf_.triangle, self.ldf_.triangle[..., -self._ave_period[0]-1:]), -1)
        X_new.sigma_.triangle = np.concatenate((X_new.sigma_.triangle, self.sigma_.triangle[..., -1:]), -1)
        X_new.cdf_.ddims = X_new.ldf_.ddims = self.ldf_.ddims
        X_new.sigma_.ddims = X_new.std_err_.ddims = self.sigma_.ddims
        X_new.cdf_.valuation = X_new.ldf_.valuation = self.ldf_.valuation
        X_new.sigma_.valuation = X_new.std_err_.valuation = self.sigma_.valuation
        return X_new

    def fit_transform(self, X, y=None, sample_weight=None):
        """ Equivalent to fit(X).transform(X)

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs based on the model.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
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
