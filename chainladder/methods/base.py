# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
from chainladder.utils.cupy import cp
import copy
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development
from chainladder.core import EstimatorIO
from chainladder.core.common import Common


class MethodBase(BaseEstimator, EstimatorIO, Common):
    ULT_VAL = '2262-03-31 23:59:59.999999999'
    def __init__(self):
        pass

    def validate_X(self, X):
        obj = copy.deepcopy(X)
        if 'ldf_' not in obj:
            obj = Development().fit_transform(obj)
        if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
            obj = TailConstant().fit_transform(obj)
        return obj

    def _align_cdf(self, ultimate):
        """ Vertically align CDF to ultimate vector """
        xp = cp.get_array_module(ultimate.values)
        o, d = ultimate.shape[-2:]
        #cdf = xp.repeat(self.cdf_.values[..., 0:1, :d], o, axis=2)
        ultimate.values = self.cdf_.values[..., :d]*(ultimate.values*0+1)
        cdf = ultimate.latest_diagonal.values
        return cdf

    def _set_ult_attr(self, ultimate):
        """ Ultimate scaffolding """
        xp = cp.get_array_module(ultimate.values)
        ultimate.values[~xp.isfinite(ultimate.values)] = xp.nan
        ultimate.ddims = np.array([9999])
        ultimate.valuation = pd.DatetimeIndex(
            [pd.to_datetime(self.ULT_VAL)]*len(ultimate.odims))
        ultimate._set_slicers()
        ultimate.valuation_date = ultimate.valuation.max()
        return ultimate

    @property
    def ldf_(self):
        return self.X_.ldf_

    @property
    def latest_diagonal(self):
        return self.X_.latest_diagonal

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
        self.sample_weight_ = sample_weight
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
        obj = copy.deepcopy(X)
        obj.ldf_ = self.ldf_
        obj.ultimate_ = self._get_ultimate(obj, sample_weight)
        return obj

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.predict(X, sample_weight)

    def _include_process_variance(self):
        if hasattr(self.X_, '_get_process_variance'):
            full = self.full_triangle_
            obj = self.X_._get_process_variance(full)
            self.ultimate_.values = obj.values[..., -1:]
            process_var = obj - full
        else:
            process_var = None
        return process_var
