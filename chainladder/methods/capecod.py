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
        self.sample_weight_ = sample_weight
        self.ultimate_, self.apriori_, self.detrended_apriori_ = \
            self._get_ultimate(X, sample_weight)
        self.process_variance_ = self._include_process_variance()
        return self

    def _get_ultimate(self, X, sample_weight):
        xp = cp.get_array_module(X.values)
        ult = copy.deepcopy(X)
        latest = X.latest_diagonal.values
        len_orig = sample_weight.shape[-2]
        cdf = self._align_cdf(ult)
        exposure = sample_weight.values
        reported_exposure = exposure / cdf
        trend_exponent = len_orig - xp.arange(len_orig) - 1
        trend_array = (1 + self.trend)**(trend_exponent)
        trend_array = X._expand_dims(trend_array[..., None])
        decay_matrix = self.decay ** xp.abs(
            xp.arange(len_orig)[None].T - xp.arange(len_orig)[None])
        decay_matrix = X._expand_dims(decay_matrix)
        weighted_exposure = xp.swapaxes(reported_exposure, -1, -2) * decay_matrix
        trended_ultimate = (latest*trend_array)/reported_exposure
        trended_ultimate = xp.swapaxes(trended_ultimate, -1, -2)
        apriori = (xp.sum(weighted_exposure*trended_ultimate, -1) /
                   xp.sum(weighted_exposure, -1))
        ult.values = apriori[..., None]
        apriori_ = copy.copy(ult)
        detrended_ultimate = apriori_.values/trend_array
        detrended_apriori_ = copy.copy(ult)
        detrended_apriori_.values = detrended_ultimate
        ult.values = latest + detrended_ultimate*(1-1/cdf)*exposure
        ult = self._set_ult_attr(ult)
        apriori_ = self._set_ult_attr(apriori_)
        detrended_apriori_ = self._set_ult_attr(detrended_apriori_)
        return ult, apriori_, detrended_apriori_

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
        obj.ultimate_, obj.apriori_, obj.detrended_apriori_ = \
            self._get_ultimate(obj, sample_weight)
        return obj
