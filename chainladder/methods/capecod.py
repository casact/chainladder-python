# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder.utils.cupy import cp
import pandas as pd
import copy
from chainladder.methods import MethodBase


class CapeCod(MethodBase):
    """Applies the CapeCod technique to triangle **X**

    Parameters
    ----------
    trend : float (default=0.0)
        The cape cod trend assumption
    decay : float (defaut=1.0)
        The cape cod decay assumption

    Attributes
    ----------
    triangle :
        returns **X**
    ultimate_ :
        The ultimate losses per the method
    ibnr_ :
        The IBNR per the method
    apriori_ :
        The trended apriori vector developed by the Cape Cod Method
    detrended_apriori_ :
        The detrended apriori vector developed by the Cape Cod Method
    """

    def __init__(self, trend=0, decay=1):
        self.trend = trend
        self.decay = decay

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Loss data to which the model will be applied.
        y : Ignored
        sample_weight : Triangle-like
            The exposure to be used in the method.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if sample_weight is None:
            raise ValueError('sample_weight is required.')
        super().fit(X, y, sample_weight)
        obj = copy.deepcopy(self)
        self.sample_weight_ = sample_weight
        self.ultimate_, self.apriori_, self.detrended_apriori_ = \
            self._get_ultimate_(X, sample_weight, obj)
        self.full_triangle_ = self._get_full_triangle_()
        return self

    def _get_ultimate_(self, X, sample_weight, obj):
        origin, development, len_orig = -2, -1, sample_weight.shape[-2]
        ult = obj.X_
        xp = cp.get_array_module(X.values)
        latest = X.latest_diagonal.values
        ult.values = \
            obj.cdf_.values[..., :ult.shape[development]]*(ult.values*0+1)
        cdf = ult.latest_diagonal.values
        exposure = sample_weight.values
        reported_exposure = exposure/cdf
        trend_exponent = len_orig-xp.arange(len_orig)-1
        trend_array = (1+self.trend)**(trend_exponent)
        trend_array = X._expand_dims(trend_array[..., xp.newaxis])
        decay_matrix = self.decay ** xp.abs(
            xp.arange(len_orig)[xp.newaxis].T -
            xp.arange(len_orig)[xp.newaxis])
        decay_matrix = X._expand_dims(decay_matrix)
        weighted_exposure = \
            xp.swapaxes(reported_exposure, development, origin)*decay_matrix
        trended_ultimate = (latest*trend_array)/reported_exposure
        trended_ultimate = xp.swapaxes(trended_ultimate, development, origin)
        apriori = xp.sum(weighted_exposure*trended_ultimate, development) / \
            xp.sum(weighted_exposure, development)
        ult.values = apriori[..., xp.newaxis]
        ult.ddims = np.array([None])
        apriori_ = copy.copy(ult)
        detrended_ultimate = apriori_.values/trend_array
        detrended_apriori_ = copy.copy(ult)
        detrended_apriori_.values = detrended_ultimate
        ibnr = detrended_ultimate*(1-1/cdf)*exposure
        ult.values = latest + ibnr
        ult.ddims = np.array([None])
        ult.valuation = pd.DatetimeIndex([pd.to_datetime('2262-04-11')] *
                                         len_orig)
        apriori_._set_slicers()
        ult._set_slicers()
        detrended_apriori_._set_slicers()
        return ult, apriori_, detrended_apriori_

    def predict(self, X, sample_weight):
        obj = super().predict(X, sample_weight)
        obj.ultimate_, obj.apriori_, obj.detrended_apriori_ = \
            obj._get_ultimate_(X, sample_weight, obj)
        obj.full_triangle_ = obj._get_full_triangle_()
        return obj
