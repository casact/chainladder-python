import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development


class MethodBase(BaseEstimator):
    def __init__(self):
        pass

    def validate_X(self, X):
        obj = copy.deepcopy(X)
        if obj.__dict__.get('ldf_', None) is None:
            obj = Development().fit_transform(obj)
        if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
            obj = TailConstant().fit_transform(obj)
        self.cdf_ = obj.cdf_
        self.ldf_ = obj.ldf_
        self.average_ = obj.average_
        return obj

    def fit(self, X, y=None, sample_weight=None):
        """Applies the chainladder technique to triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = self.validate_X(X)

        return self

    def predict(self, X):
        """Predicts the chainladder ultimate on a new triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        Returns
        -------
        X_new: Triangle

        """
        obj = copy.deepcopy(self)
        obj.X_ = copy.deepcopy(X)
        return obj

    @property
    def full_expectation_(self):
        obj = copy.deepcopy(self.X_)
        obj.valuation = self._set_valuation(obj)
        obj.triangle = self.ultimate_.triangle / self.cdf_.triangle
        obj.triangle = \
            np.concatenate((obj.triangle, self.ultimate_.triangle), -1)
        obj.ddims = np.array([item for item in obj.ddims]+[9999])
        obj.nan_override = True
        return obj

    @property
    def ibnr_(self):
        obj = copy.deepcopy(self.ultimate_)
        obj.triangle = self.ultimate_.triangle-self.X_.latest_diagonal.triangle
        obj.ddims = ['IBNR']
        return obj

    def _get_full_triangle_(self):
        obj = copy.deepcopy(self.X_)
        w = 1-np.nan_to_num(obj.nan_triangle())
        extend = len(self.ldf_.ddims) - len(self.X_.ddims)
        ones = np.ones((w.shape[-2], extend))
        w = np.concatenate((w, ones), -1)
        obj.nan_override = True
        e_tri = np.repeat(self.ultimate_.triangle,
                          self.cdf_.triangle.shape[3], 3)/self.cdf_.triangle
        e_tri = e_tri * w
        zeros = obj.expand_dims(ones - ones)
        obj.valuation = self._set_valuation(obj)
        obj.triangle = \
            np.concatenate((np.nan_to_num(obj.triangle), zeros), -1) + e_tri
        obj.triangle = np.concatenate((obj.triangle,
                                       self.ultimate_.triangle), 3)
        obj.ddims = np.array([item for item in obj.ddims]+[9999])
        return obj

    def _set_valuation(self, obj):
        ldf_val = pd.DataFrame(self.ldf_.valuation.values.reshape(self.ldf_.shape[-2:], order='f'))
        val_array = pd.DataFrame(obj.valuation.values.reshape(obj.shape[-2:], order='f')).iloc[:,0]
        val_array = pd.concat((val_array, ldf_val), axis=1, ignore_index=True)
        val_array = pd.DatetimeIndex(pd.DataFrame(val_array).unstack().values)
        return val_array
