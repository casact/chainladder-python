# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development import Development, DevelopmentBase
from chainladder.utils import WeightedRegression
import numpy as np
import pandas as pd
import warnings


class IncrementalAdditive(DevelopmentBase):
    """ The Incremental Additive Method.

    Parameters
    ----------
    trend: float (default=0.0)
        A multiplicative trend amount used to trend each incremental development
        period the valuation_date of the Triangle.
    future_trend: float (default=None)
        The trend to apply to the incremental development periods in the lower
        half of the completed Triangle. If None, then will be set to the value of
        the trend parameter.
    n_periods: integer, optional (default=-1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods=-1
    average: str optional (default='volume')
        type of averaging to use for average incremental factor calculation.  Options include
        'regression', 'volume' and 'simple'.
    drop: tuple or list of tuples
        Drops specific origin/development combination(s)
    drop_high: bool or list of bool (default=None)
        Drops highest link ratio(s) from LDF calculation
    drop_low: bool or list of bool (default=None)
        Drops lowest link ratio(s) from LDF calculation
    drop_above: float or list of floats (default = numpy.inf)
        Drops all link ratio(s) above the given parameter from incremental factor calculation
    drop_below: float or list of floats (default = -numpy.inf)
        Drops all link ratio(s) below the given parameter from incremental factor calculation
    preserve: int (default = 1)
        The minimum number of incremental factor(s) required for incremental factor calculation
    drop_valuation: str or list of str (default=None)
        Drops specific valuation periods. str must be date convertible.

    Attributes
    ----------
    ldf_: Triangle
        The estimated loss development patterns
    cdf_: Triangle
        The estimated cumulative development patterns
    tri_zeta: Triangle
        The raw incrementals as a percent of exposure trended to the valuation
        date of the Triangle.
    fit_zeta: Triangle
        The raw incrementals as a percent of exposure trended to the valuation
        date of the Triangle. Only those used in the fitting.
    zeta_: Triangle
        The fitted incrementals as a percent of exposure trended to the valuation
        date of the Triangle.
    cum_zeta_: Triangle
        The fitted cumulative percent of exposure trended to the valuation date of 
        the Triangle
    w_: ndarray
        The weight used in the zeta fitting
    w_tri_: Triangle
        Triangle of w_
    sample_weight: Triangle
        The exposure used to obtain incremental factor
    incremental_: Triangle
        A triangle of full incremental values.


    """

    def __init__(
        self, trend=0.0, n_periods=-1, average="volume", future_trend=0,
        drop=None, drop_high=None, drop_low=None, drop_above=np.inf, drop_below=-np.inf, drop_valuation=None, preserve = 1):
        self.trend = trend
        self.n_periods = n_periods
        self.average = average
        self.future_trend = future_trend
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_above = drop_above
        self.drop_below = drop_below
        self.drop_valuation = drop_valuation
        self.preserve = preserve
        self.drop = drop
        self.is_additive = True

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
        #check dev lag
        if type(X.ddims) != np.ndarray:
            raise ValueError("Triangle must be expressed with development lags")
        #convert to numpy
        if X.array_backend == "sparse":
            X = X.set_backend("numpy")
        else:
            X = X.copy()
        if sample_weight.array_backend == "sparse":
            sample_weight = sample_weight.set_backend("numpy")
        else:
            sample_weight = sample_weight.copy()
        #get backend
        xp = X.get_array_module()
        self.xp = xp
        #short cut to use sample_weight as is
        sample_weight.is_cumulative = False
        #get incremental factor
        X_incr = X.cum_to_incr()
        if hasattr(X, "trend_"):
            if self.trend != 0:
                warnings.warn(
                    "IncrementalAdditive Trend assumption is ignored when X has a trend_ property."
                )
            X_trended = X_incr * X_incr.trend_.values
        else:
            X_trended = X_incr.trend(self.trend, axis='valuation')
        x = X_trended / sample_weight.values
        #assign weights according to n_periods and drops
        if hasattr(X, "w_"):
            self.w_tri_ = self._set_weight_func(x * X.w_,X_trended)
        else:
            self.w_tri_ = self._set_weight_func(x,X_trended)
        self.w_ = self.w_tri_.values
        #calculate factors
        super().fit(sample_weight.values,X_trended.values,self.w_)
        #keep attributes
        self.tri_zeta = x.copy()
        self.sample_weight = sample_weight
        self.fit_zeta_ = self.tri_zeta * self.w_
        self.zeta_ = self._param_property(x,self.params_.slope_[...,0][..., None, :])
        
        #to consolidate under full_triangle_
        y_ = xp.repeat(self.zeta_.values, len(x.odims), -2)
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
        
        #to migrate under _zeta_to_ldf method under common, so ldf_ can be correct after tail
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
        for item in ["ldf_", "w_", "zeta_", "incremental_", "tri_zeta", "fit_zeta_", "sample_weight"]:
            X_new.__dict__[item] = self.__dict__[item]
        return X_new

    def _param_property(self, factor, params):
        from chainladder import options
        
        obj = factor[factor.origin == factor.origin.min()]
        xp = factor.get_array_module()
        obj.values = params
        obj.valuation_date = pd.to_datetime(options.ULT_VAL)
        obj.is_pattern = True
        obj.is_additive = True
        obj.is_cumulative = False
        obj.virtual_columns.columns = {}
        obj._set_slicers()
        return obj
