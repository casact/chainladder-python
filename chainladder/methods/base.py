import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development
from chainladder.core import IO


class MethodBase(BaseEstimator, IO):
    def __init__(self):
        pass

    def validate_X(self, X):
        obj = copy.deepcopy(X)
        if obj.__dict__.get('ldf_', None) is None:
            obj = Development().fit_transform(obj)
        if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
            obj = TailConstant().fit_transform(obj)
        self.cdf_ = obj.__dict__.get('cdf_', None)
        self.ldf_ = obj.__dict__.get('ldf_', None)
        self.average_ = obj.__dict__.get('average_', None)
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
        obj.values = self.ultimate_.values / self.cdf_.values
        obj.values = \
            np.concatenate((obj.values, self.ultimate_.values), -1)
        ddims = [int(item[item.find('-')+1:]) for item in self.cdf_.ddims]
        obj.ddims = np.array([obj.ddims[0]]+ddims)
        obj.valuation = obj._valuation_triangle(obj.ddims)
        obj.nan_override = True
        return obj

    @property
    def ibnr_(self):
        obj = copy.deepcopy(self.ultimate_)
        obj.values = self.ultimate_.values-self.X_.latest_diagonal.values
        obj.ddims = ['IBNR']
        return obj

    def _get_full_triangle_(self):
        obj = copy.deepcopy(self.X_)
        w = 1-np.nan_to_num(obj.nan_triangle())
        extend = len(self.ldf_.ddims) - len(self.X_.ddims)
        ones = np.ones((w.shape[-2], extend))
        w = np.concatenate((w, ones), -1)
        obj.nan_override = True
        e_tri = np.repeat(self.ultimate_.values,
                          self.cdf_.values.shape[3], 3)/self.cdf_.values
        e_tri = e_tri * w
        zeros = obj.expand_dims(ones - ones)
        properties = self.full_expectation_
        obj.valuation = properties.valuation
        obj.ddims = properties.ddims
        obj.values = \
            np.concatenate((np.nan_to_num(obj.values), zeros), -1) + e_tri
        obj.values = np.concatenate((obj.values,
                                     self.ultimate_.values), 3)
        return obj
