# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
from chainladder.utils.cupy import cp
import copy
from chainladder.methods import MethodBase


class Chainladder(MethodBase):
    """
    The basic determinsitic chainladder method.

    Parameters
    ----------
    None

    Attributes
    ----------
    X_
        returns **X** used to fit the triangle
    ultimate_
        The ultimate losses per the method
    ibnr_
        The IBNR per the method
    full_expectation_
        The ultimates back-filled to each development period in **X** replacing
        the known data
    full_triangle_
        The ultimates back-filled to each development period in **X** retaining
        the known data
    """

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Data to which the model will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, sample_weight)
        self.ultimate_ = self._get_ultimate(self.X_)
        self.full_triangle_ = self._get_full_triangle(self)
        return self

    def predict(self, X, sample_weight=None):
        obj = copy.deepcopy(X)
        obj.ultimate_ = self._get_ultimate(obj)
        obj.full_triangle_ = self._get_full_triangle(obj)
        return obj

    def _get_ultimate(self, X):
        """ Private method that uses CDFs to obtain an ultimate vector """
        xp = cp.get_array_module(X.values)
        o, d = X.shape[-2:]
        cdf = xp.repeat(self.cdf_.values[..., 0:1, :d], o, axis=2)
        ultimate_ = (X * cdf).latest_diagonal
        ultimate_.ddims = np.array(['Ult'])
        ultimate_.valuation = pd.DatetimeIndex(
            [pd.to_datetime('2262-04-11')]*o)
        ultimate_._set_slicers()
        ultimate_.valuation_date = ultimate_.valuation.max()
        return ultimate_

    def _get_full_triangle(self, X):
        """ Private method that builds full triangle from ultimates"""
        xp = cp.get_array_module(X.ultimate_.values)
        o, d = X.ultimate_.shape[-2:]
        cdf = copy.deepcopy(self.cdf_)
        cdf.values = xp.repeat(cdf.values[..., 0:1, :], o, axis=2)
        cdf.odims = X.ultimate_.odims
        cdf.valuation_date = X.ultimate_.valuation_date
        full = X.ultimate_ / cdf
        full.values = xp.concatenate((full.values, X.ultimate_.values), -1)
        full.ddims = xp.append(full.ddims, '9999-Ult')
        full.ddims = xp.array([item.split('-')[0] for item in full.ddims])
        full.valuation = full._valuation_triangle()
        return full
