# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development import Development, DevelopmentBase
import numpy as np
import pandas as pd
import warnings


class IncrementalAdditive(DevelopmentBase):
    """ The Incremental Additive Method.

    Parameters
    ----------
    trend : float (default=0.0)
        A multiplicative trend amount used to trend each incremental development
        period the valuation_date of the Triangle.
    future_trend : float (default=None)
        The trend to apply to the incremental development periods in the lower
        half of the completed Triangle. If None, then will be set to the value of
        the trend parameter.
    n_periods : integer, optional (default=-1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods=-1
    average: str optional (default='volume')
        type of averaging to use for ldf average calculation.  Options include
        'volume' and 'simple'.
    drop : tuple or list of tuples
        Drops specific origin/development combination(s)
    drop_high : bool or list of bool (default=None)
        Drops highest link ratio(s) from LDF calculation
    drop_low : bool or list of bool (default=None)
        Drops lowest link ratio(s) from LDF calculation
    drop_valuation : str or list of str (default=None)
        Drops specific valuation periods. str must be date convertible.

    Attributes
    ----------
    ldf_ : Triangle
        The estimated loss development patterns
    cdf_ : Triangle
        The estimated cumulative development patterns
    zeta : Triangle
        The fitted incrementals as a percent of exposure trended to the valuation
        date of the Triangle.
    incremental_ : Triangle
        A triangle of full incremental values.


    """

    def __init__(
        self, trend=0.0, n_periods=-1, average="volume", future_trend=0,
        drop=None, drop_high=None, drop_low=None, drop_valuation=None):
        self.trend = trend
        self.n_periods = n_periods
        self.average = average
        self.future_trend = future_trend
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_valuation = drop_valuation
        self.drop = drop


    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Triangle to which the incremental method is applied.  Triangle must
            be cumulative.
        y : None
            Ignored
        sample_weight :
            Exposure used in the method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        from chainladder import ULT_VAL
        from chainladder.utils.utility_functions import num_to_nan

        if type(X.ddims) != np.ndarray:
            raise ValueError("Triangle must be expressed with development lags")
        if X.array_backend == "sparse":
            X = X.set_backend("numpy")
        else:
            X = X.copy()
        if sample_weight.array_backend == "sparse":
            sample_weight = sample_weight.set_backend("numpy")
        else:
            sample_weight = sample_weight.copy()
        xp = X.get_array_module()
        sample_weight.is_cumulative = False
        obj = X.cum_to_incr() / sample_weight.values
        if hasattr(X, "trend_"):
            if self.trend != 0:
                warnings.warn(
                    "IncrementalAdditive Trend assumption is ignored when X has a trend_ property."
                )
            x = obj * obj.trend_.values
        else:
            x = obj.trend(self.trend, axis='valuation')

        w_ = Development(
            n_periods=self.n_periods - 1, drop=self.drop,
            drop_high=self.drop_high, drop_low=self.drop_low,
            drop_valuation=self.drop_valuation).fit(x).w_
        # This will miss drops on the latest diagonal
        w_ = num_to_nan(w_)
        w_ = xp.concatenate((w_, (w_[..., -1:] * x.nan_triangle)[..., -1:]), axis=-1)
        if self.average == "simple":
            y_ = xp.nanmean(w_ * x.values, axis=-2)
        if self.average == "volume":
            y_ = xp.nansum(w_ * x.values * sample_weight.values, axis=-2)
            y_ = y_ / xp.nansum(w_ * sample_weight.values, axis=-2)
        self.zeta_ = X.iloc[..., -1:, :]
        self.zeta_.values = y_[:, :, None, :]
        y_ = xp.repeat(y_[..., None, :], len(x.odims), -2)
        obj = x.copy()
        keeps = (
            1
            - xp.nan_to_num(x.nan_triangle)
            + xp.nan_to_num(
                x[x.valuation == x.valuation_date].values[0, 0, ...] * 0 + 1
            )
        )
        obj.values = y_ * keeps
        obj.valuation_date = obj.valuation.max()
        obj.values = obj.values * (1 - xp.nan_to_num(x.nan_triangle)) + xp.nan_to_num(
            (X.cum_to_incr().values / sample_weight.values)
        )

        obj.values[obj.values == 0] = xp.nan
        obj._set_slicers()
        obj.valuation_date = obj.valuation.max()
        future_trend = self.trend if not self.future_trend else self.future_trend
        self.incremental_ = obj * sample_weight.values
        self.incremental_ = self.incremental_.trend(
            1/(1+future_trend)-1, axis='valuation', start=X.valuation_date,
            end=self.incremental_.valuation_date)
        self.ldf_ = obj.incr_to_cum().link_ratio
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
        X_new = X.copy()
        for item in ["ldf_"]:
            X_new.__dict__[item] = self.__dict__[item]
        return X_new
