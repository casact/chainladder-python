# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.tails import TailBase
from chainladder.utils import WeightedRegression
from chainladder.development import DevelopmentBase, Development
import numpy as np
import pandas as pd
from chainladder.utils.cupy import cp
import copy
import warnings


class TailCurve(TailBase):
    """Allows for extraploation of LDFs to form a tail factor.

    Parameters
    ----------
    curve : str ('exponential', 'inverse_power')
        The type of curve extrapolation you'd like to use
    fit_period : tuple (start, stop)
        A tuple representing the range of ldfs to use in the curve fit.
        The use of ``None`` will use the edge of the triangle.  For example,
        (48, None) will use development factors for age 48 and beyond.
    extrap_periods : int
        Then number of development periods from attachment point to extrapolate
        the fit.
    errors : str ('raise' or 'ignore')
        Whether to raise an error or ignore observations that violate the
        distribution being fit.  The most common is ldfs < 1.0 will not work
        in either the ``exponential`` or ``inverse_power`` fits.
    attachment_age: int (default=None)
        The age at which to attach the fitted curve.  If None, then the latest
        age is used. Measures of variability from original `ldf_` are retained
        when being used in conjunction with the MackChainladder method.

    Attributes
    ----------
    ldf_ : Triangle
        ldf with tail applied.
    cdf_ : Triangle
        cdf with tail applied.
    tail_ : DataFrame
        Point estimate of tail at latest maturity available in the Triangle.
    slope_ : DataFrame
        Slope parameter of the curve fit.
    intercept : DataFrame
        Intercept parameter of the curve fit.
    sigma_ : Triangle
        sigma with tail factor applied.
    std_err_ : Triangle
        std_err with tail factor applied
    """
    def __init__(self, curve='exponential', fit_period=(None, None),
                 extrap_periods=100, errors='ignore', attachment_age=None):
        self.curve = curve
        self.fit_period = fit_period
        self.extrap_periods = extrap_periods
        self.errors = errors
        self.attachment_age = attachment_age

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the tail will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if type(self.fit_period) == slice:
            warnings.warn("Slicing for fit_period is deprecated and will be removed. Please use a tuple (start_age, end_age).")
            fit_period = self.fit_period
        else:
            grain = {'Y': 12, 'Q': 3, 'M': 1}[X.development_grain]
            start = self.fit_period[0] if self.fit_period[0] is None else int(self.fit_period[0] / grain - 1)
            end = self.fit_period[1] if self.fit_period[1] is None else int(self.fit_period[1] / grain - 1)
            fit_period = slice(start, end, None)
        super().fit(X, y, sample_weight)
        xp = cp.get_array_module(self.ldf_.values)
        _y = self.ldf_.values[..., :X.shape[-1]-1].copy()
        _w = xp.zeros(_y.shape)
        _w[..., fit_period] = 1.0
        if self.errors == 'ignore':
            _w[_y <= 1.0] = 0
            _y[_y <= 1.0] = 1.01
        elif self.errors == 'raise' and xp.any(y < 1.0):
            raise ZeroDivisionError('Tail fit requires all LDFs to be greater than 1.0')
        _y = xp.log(_y - 1)
        n_obs = X.shape[-1]-1
        k, v = X.shape[:2]
        _x = self._get_x(_w, _y)
        # Get LDFs
        coefs = WeightedRegression(axis=3).fit(_x, _y, _w)
        self._slope_, self._intercept_ = coefs.slope_, coefs.intercept_
        extrapolate = xp.cumsum(
            xp.ones(tuple(list(_y.shape)[:-1] +
                    [self.extrap_periods + n_obs])), -1)
        tail = self._predict_tail(extrapolate)
        if self.attachment_age:
            attach_idx = xp.min(xp.where(X.ddims>=self.attachment_age))
        else:
            attach_idx = len(X.ddims) - 1
        self.ldf_.values = xp.concatenate(
            (self.ldf_.values[..., :attach_idx], tail[..., attach_idx:]), -1)
        obj = Development().fit_transform(X) if 'ldf_' not in X else X
        sigma, std_err = self._get_tail_stats(obj)
        self.sigma_.values[..., -1] = sigma[..., -1]
        self.std_err_.values[..., -1] = std_err[..., -1]
        self.cdf_ = DevelopmentBase._get_cdf(self)
        return self

    def _get_x(self, w, y):
        # For Exponential decay, no transformation on x is needed
        if self.curve == 'exponential':
            return None
        if self.curve == 'inverse_power':
            reg = WeightedRegression(3, False).fit(None, y, w).infer_x_w()
            xp = cp.get_array_module(reg.x)
            return xp.log(reg.x)

    def _predict_tail(self, extrapolate):
        xp = cp.get_array_module(extrapolate)
        if self.curve == 'exponential':
            tail_ldf = xp.exp(self._slope_*extrapolate + self._intercept_)
        if self.curve == 'inverse_power':
            tail_ldf = xp.exp(self._intercept_)*(extrapolate**self._slope_)
        return self._get_tail_prediction(tail_ldf)

    @property
    def slope_(self):
        """ Does not work with munich """
        idx = self.cdf_._idx_table()
        return pd.DataFrame(self._slope_[..., 0, 0],
                            index=idx.index, columns=idx.columns)

    @property
    def intercept_(self):
        """ Does not work with munich """
        idx = self.cdf_._idx_table()
        return pd.DataFrame(self._intercept_[..., 0, 0],
                            index=idx.index, columns=idx.columns)
