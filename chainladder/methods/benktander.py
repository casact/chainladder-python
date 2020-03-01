# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from chainladder.utils.cupy import cp
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
    apriori_sigma : float, optional (default=0.0)
        Standard deviation of the apriori.  When used in conjunction with the
        bootstrap model, the model samples aprioris from a lognormal distribution
        using this argument as a standard deviation.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

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
    def __init__(self, apriori=1.0, n_iters=1, apriori_sigma=0, random_state=None):
        self.apriori = apriori
        self.n_iters = n_iters
        self.apriori_sigma = apriori_sigma
        self.random_state = random_state

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
        if hasattr(X, '_get_process_variance'):
            if X.shape[0] != sample_weight.shape[0]:
                sample_weight = sample_weight.broadcast_axis('index', X.index)
        obj = copy.deepcopy(self)
        self.sample_weight_ = sample_weight
        self.ultimate_ = self._get_ultimate_(X, sample_weight, obj)
        self.full_triangle_ = self._get_full_triangle_()
        return self

    def _get_ultimate_(self, X, sample_weight, obj):
        ult = copy.copy(obj.X_)
        xp = cp.get_array_module(ult.values)
        origin, development = -2, -1  # Set axes by name
        latest = X.latest_diagonal.values
        if self.apriori_sigma != 0:
            random_state = xp.random.RandomState(self.random_state)
            apriori = random_state.normal(
                self.apriori, self.apriori_sigma, X.shape[0])
            apriori = apriori.reshape(X.shape[0],-1)[..., np.newaxis, np.newaxis]
            apriori = sample_weight.values * apriori
        else:
            apriori = sample_weight.values*self.apriori
        ult.values = \
            obj.cdf_.values[..., :ult.shape[development]]*(ult.values*0+1)
        cdf = ult.latest_diagonal.values
        cdf = (1-1/cdf)[xp.newaxis]
        exponents = xp.arange(self.n_iters+1)
        exponents = xp.reshape(exponents, tuple([len(exponents)]+[1]*4))
        cdf = cdf**exponents
        ult.values = xp.sum(cdf[:-1, ...], 0)*latest+cdf[-1, ...]*apriori
        ult.values[~xp.isfinite(ult.values)] = xp.nan
        ult.ddims = np.array([None])
        ult.valuation = pd.DatetimeIndex([pd.to_datetime('2262-04-11')] *
                                         ult.shape[origin])
        ult._set_slicers()
        return ult

    def predict(self, X, sample_weight):
        obj = super().predict(X, sample_weight)
        obj.ultimate_ = self._get_ultimate_(X, sample_weight, obj)
        obj.full_triangle_ = obj._get_full_triangle_()
        return obj
