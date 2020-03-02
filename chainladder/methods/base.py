# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder.utils.cupy import cp
import copy
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development
from chainladder.core import EstimatorIO


class MethodBase(BaseEstimator, EstimatorIO):
    def __init__(self):
        pass

    def validate_X(self, X):
        obj = copy.copy(X)
        if 'ldf_' not in obj:
            obj = Development().fit_transform(obj)
        if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
            obj = TailConstant().fit_transform(obj)
        for item in ['cdf_', 'ldf_', 'average_']:
            setattr(self, item, getattr(obj, item, None))
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

    def predict(self, X, sample_weight=None):
        """Predicts the chainladder ultimate on a new triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        sample_weight : Triangle
            For exposure-based methods, the exposure to be used for predictions

        Returns
        -------
        X_new: Triangle

        """
        obj = copy.copy(self)
        xp = cp.get_array_module(X.values)
        obj.X_ = copy.copy(X)
        obj.sample_weight = sample_weight
        if xp.unique(self.cdf_.values, axis=-2).shape[-2] == 1:
            obj.cdf_.values = xp.repeat(
                xp.unique(self.cdf_.values, axis=-2),
                len(X.odims), -2)
            obj.ldf_.values = xp.repeat(
                xp.unique(self.ldf_.values, axis=-2),
                len(X.odims), -2)
            obj.cdf_.odims = obj.ldf_.odims = obj.X_.odims
            obj.cdf_.valuation = obj.ldf_.valuation = \
                Development().fit(X).cdf_.valuation
        obj.cdf_._set_slicers()
        obj.ldf_._set_slicers()
        return obj

    @property
    def full_expectation_(self):
        obj = copy.copy(self.X_)
        xp = cp.get_array_module(obj.values)
        obj.values = (self.ultimate_.values /
                      xp.unique(self.cdf_.values, axis=-2))
        obj.values = xp.concatenate((obj.values,
                                    self.ultimate_.values), -1)
        ddims = [int(item[item.find('-')+1:]) for item in self.ldf_.ddims]
        obj.ddims = np.array([obj.ddims[0]]+ddims)
        obj.valuation = obj._valuation_triangle(obj.ddims)
        obj.valuation_date = max(obj.valuation)
        obj.nan_override = True
        obj.values[obj.values == 0] = xp.nan
        obj._set_slicers()
        return obj

    @property
    def ibnr_(self):
        obj = copy.copy(self.ultimate_)
        obj.values = self.ultimate_.values-self.X_.latest_diagonal.values
        obj.ddims = [None]
        obj._set_slicers()
        return obj

    def _get_full_triangle_(self):
        obj = copy.copy(self.X_)
        xp = cp.get_array_module(obj.values)
        w = 1-xp.nan_to_num(obj._nan_triangle())
        extend = len(self.ldf_.ddims) - len(self.X_.ddims)
        ones = xp.ones((w.shape[-2], extend))
        w = xp.concatenate((w, ones), -1)
        obj.nan_override = True
        e_tri = \
            xp.repeat(self.ultimate_.values, self.cdf_.values.shape[3], 3) / \
            xp.unique(self.cdf_.values, axis=-2)
        e_tri = e_tri * w
        zeros = obj._expand_dims(ones - ones)
        properties = self.full_expectation_
        obj.valuation = properties.valuation
        obj.valuation_date = properties.valuation_date
        obj.ddims = properties.ddims
        obj.values = \
            xp.concatenate((xp.nan_to_num(obj.values), zeros), -1) + e_tri
        obj.values = xp.concatenate((obj.values,
                                     self.ultimate_.values), 3)
        obj.values[obj.values==0] = xp.nan
        obj._set_slicers()
        if hasattr(self.X_, '_get_process_variance'):
            obj = self.X_._get_process_variance(obj)
            self.ultimate_.values = obj.values[..., -1:]
        return obj
