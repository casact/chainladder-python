"""
:ref:`chainladder.methods<methods>`.CapeCod
===============================================

:ref:`CapeCod<capecod>` is a cool method
"""
import numpy as np
import copy
from chainladder.methods import MethodBase


class CapeCod(MethodBase):
    def __init__(self, trend=0, decay=1):
        self.trend = trend
        self.decay = decay

    def fit(self, X, y=None, sample_weight=None):
        """Applies the CapeCod technique to triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored
        sample_weight : Triangle
            Required exposure to be used in the calculation.

        Attributes
        ----------
        triangle :
            returns **X**
        ultimate_ :
            The ultimate losses per the method
        ibnr_ :
            The IBNR per the method
        apriori_ :
            The apriori vector developed by the Cape Cod Method
        """
        super().fit(X, y, sample_weight)
        self.sample_weight_ = sample_weight
        latest = self.X_.latest_diagonal.triangle
        obj = copy.deepcopy(self.X_)
        obj.triangle = self.X_.cdf_.triangle * (obj.triangle*0+1)
        cdf = obj.latest_diagonal.triangle
        exposure = sample_weight.triangle
        reported_exposure = exposure/cdf
        trend_exponent = exposure.shape[-2]-np.arange(exposure.shape[-2])-1
        trend_array = (1+self.trend)**(trend_exponent)
        trend_array = self.X_.expand_dims(np.expand_dims(trend_array, -1))
        decay_matrix = self.decay * \
            (np.abs(np.expand_dims(np.arange(exposure.shape[-2]), 0).T -
             np.expand_dims(np.arange(exposure.shape[-2]), 0)))
        decay_matrix = self.X_.expand_dims(decay_matrix)
        weighted_exposure = reported_exposure * decay_matrix
        trended_ultimate = (latest*trend_array)/(reported_exposure)
        apriori = np.sum(weighted_exposure*trended_ultimate, -1) / \
            np.sum(weighted_exposure, -1)

        obj.triangle = np.expand_dims(apriori, -1)
        obj.ddims = ['Apriori']
        self.apriori_ = copy.deepcopy(obj)
        detrended_ultimate = self.apriori_.triangle/trend_array
        ibnr = detrended_ultimate*(1-1/cdf)*exposure
        obj.triangle = latest + ibnr
        obj.ddims = ['Ultimate']
        self.ultimate_ = obj
        return self
