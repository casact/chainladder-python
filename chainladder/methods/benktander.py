# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import copy
from chainladder.methods import MethodBase


class Benktander(MethodBase):
    """ The Benktander (or iterated Bornhuetter-Ferguson) IBNR model

    Parameters
    ----------
    apriori : float, optional (default=1.0)
        Multiplier for the sample_weight used in the Benktander method
        method. If sample_weight is already an apriori measure of ultimate,
        then use 1.0
    n_iters : int, optional (default=1)
        Multiplier for the sample_weight used in the Bornhuetter Ferguson
        method. If sample_weight is already an apriori measure of ultimate,
        then use 1.0

    Attributes
    ----------
    ultimate_ : Triangle
        The ultimate losses per the method
    ibnr_ : Triangle
        The IBNR per the method

    References
    ----------
    .. [2] Benktander, G. (1976) An Approach to Credibility in Calculating IBNR
           for Casualty Excess Reinsurance. In The Actuarial Review, April 1976, p.7
    """
    def __init__(self, apriori=1.0, n_iters=1):
        self.apriori = apriori
        self.n_iters = n_iters

    def fit(self, X, y=None, sample_weight=None):
        """Applies the Benktander technique to triangle **X**

        Parameters
        ----------
        X : Triangle
            Loss data to which the model will be applied.
        y : None
            Ignored
        sample_weight : Triangle
            Required exposure to be used in the calculation.

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
        self.ultimate_ = self._get_ultimate_(X, sample_weight, obj)
        self.full_triangle_ = self._get_full_triangle_()
        return self

    def _get_ultimate_(self, X, sample_weight, obj):
        ult = obj.X_
        origin, development = -2, -1  # Set axes by name
        latest = X.latest_diagonal.values
        apriori = sample_weight.values*self.apriori
        ult.values = \
            obj.cdf_.values[..., :ult.shape[development]]*(ult.values*0+1)
        cdf = ult.latest_diagonal.values
        cdf = (1-1/cdf)[np.newaxis]
        exponents = np.arange(self.n_iters+1)
        exponents = np.reshape(exponents, tuple([len(exponents)]+[1]*4))
        cdf = cdf**exponents
        ult.values = np.sum(cdf[:-1, ...], 0)*latest+cdf[-1, ...]*apriori
        ult.values[~np.isfinite(ult.values)] = np.nan
        ult.ddims = np.array([None])
        ult.valuation = pd.DatetimeIndex([pd.to_datetime('2262-04-11')] *
                                         ult.shape[origin])
        return ult

    def predict(self, X, sample_weight):
        obj = super().predict(X, sample_weight)
        obj.ultimate_ = self._get_ultimate_(X, sample_weight, obj)
        obj.full_triangle_ = obj._get_full_triangle_()
        return obj
