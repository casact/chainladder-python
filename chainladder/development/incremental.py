# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development.base import Development, DevelopmentBase
import numpy as np
from chainladder.utils.cupy import cp
import copy


class IncrementalAdditive(DevelopmentBase):
    """ The Incremental Additive Method.

    Parameters
    ----------
    trend : float (default=0.0)
        A multiplicative trend amount used to trend each development period to
        a common level.
    n_periods : integer, optional (default=-1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods=-1
    average: str optional (default='volume')
        type of averaging to use for ldf average calculation.  Options include
        'volume' and 'simple'.

    Attributes
    ----------
    ldf_ : Triangle
        The estimated loss development patterns
    cdf_ : Triangle
        The estimated cumulative development patterns
    incremental_ : Triangle
        A triangle of full incremental values.


    """
    def __init__(self, trend=0.0, n_periods=-1, average='volume'):
        self.trend = trend
        self.n_periods = n_periods
        self.average = average

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Triangle to which the incremental method is applied.  Triangle must
            be cumulative.
        y : Ignored
        sample_weight : Exposure used in the method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if (type(X.ddims) != np.ndarray):
            raise ValueError('Triangle must be expressed with development lags')
        obj = X.cum_to_incr()/sample_weight
        xp = cp.get_array_module(obj.values)
        x = obj.trend(self.trend)
        w_ = Development(n_periods=self.n_periods-1).fit(x).w_
        w_[w_ == 0] = xp.nan
        w_ = xp.concatenate((w_, (w_[..., -1:]*x._nan_triangle())[..., -1:]),
                            axis=-1)
        if self.average == 'simple':
            y_ = xp.nanmean(w_*x.values, axis=-2)
        if self.average == 'volume':
            y_ = xp.nansum(w_*x.values*sample_weight.values, axis=-2)
            y_ = y_ / xp.nansum(w_*sample_weight.values, axis=-2)
        y_ = xp.repeat(xp.expand_dims(y_, -2), len(x.odims), -2)
        obj = copy.copy(x)
        keeps = 1-xp.nan_to_num(x._nan_triangle()) + \
            xp.nan_to_num(
                x._get_latest_diagonal(compress=False).values[0, 0, ...]*0+1)
        obj.values = (1+self.trend) ** \
            xp.flip((xp.abs(xp.arange(obj.shape[-2])[xp.newaxis].T -
                     xp.arange(obj.shape[-2])[xp.newaxis])), 0)*y_*keeps
        obj.values = obj.values*(X._expand_dims(1-xp.nan_to_num(x._nan_triangle()))) + \
            xp.nan_to_num((X.cum_to_incr()/sample_weight).values)
        obj.values[obj.values == 0] = xp.nan
        obj.nan_override = True
        obj._set_slicers()
        self.incremental_ = obj*sample_weight
        self.ldf_ = obj.incr_to_cum().link_ratio
        self.cdf_ = DevelopmentBase._get_cdf(self.ldf_)
        self.sigma_ = self.std_err_ = 0*self.ldf_
        return self

    def transform(self, X):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = copy.copy(X)
        for item in ['incremental_', 'cdf_', 'ldf_', 'sigma_', 'std_err_']:
            X_new.__dict__[item] = self.__dict__[item]
        return X_new
