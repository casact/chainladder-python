# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.tails import TailBase
from chainladder.utils import WeightedRegression
from chainladder.development import DevelopmentBase, Development
import numpy as np
from chainladder.utils.cupy import cp


class TailCurve(TailBase):
    """Allows for extraploation of LDFs to form a tail factor.

    Parameters
    ----------
    curve : str ('exponential', 'inverse_power')
        The type of curve extrapolation you'd like to use
    fit_period : slice
        A slice object representing the range (by index) of ldfs to use in
        the curve fit.
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
    ldf_ :
        ldf with tail applied.
    cdf_ :
        cdf with tail applied.
    sigma_ :
        sigma with tail factor applied.
    std_err_ :
        std_err with tail factor applied
    """
    def __init__(self, curve='exponential', fit_period=slice(None, None, None),
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
        super().fit(X, y, sample_weight)
        xp = cp.get_array_module(self.ldf_.values)
        _y = self.ldf_.values[..., :X.shape[-1]-1].copy()
        _w = xp.zeros(_y.shape)
        if type(self.fit_period) is not slice:
            raise TypeError('fit_period must be slice.')
        else:
            _w[..., self.fit_period] = 1.0
        if self.errors == 'ignore':
            _w[_y <= 1.0] = 0
            _y[_y <= 1.0] = 1.01
        elif self.errors == 'raise' and xp.any(y < 1.0):
            raise ZeroDivisionError('Tail fit requires all LDFs to be' +
                                    ' greater than 1.0')
        _y = xp.log(_y - 1)
        n_obs = X.shape[-1]-1
        k, v = X.shape[:2]
        _x = self._get_x(_w, _y)
        # Get LDFs
        coefs = WeightedRegression(axis=3).fit(_x, _y, _w)
        slope, intercept = coefs.slope_, coefs.intercept_
        extrapolate = xp.cumsum(
            xp.ones(tuple(list(_y.shape)[:-1] +
                    [self.extrap_periods + n_obs])), -1)
        tail = self._predict_tail(slope, intercept, extrapolate)
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
        self.slope_ = slope
        self.intercept_ = intercept
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

    def _predict_tail(self, slope, intercept, extrapolate):
        xp = cp.get_array_module(extrapolate)
        if self.curve == 'exponential':
            tail_ldf = xp.exp(slope*extrapolate + intercept)
        if self.curve == 'inverse_power':
            tail_ldf = xp.exp(intercept)*(extrapolate**slope)
        return self._get_tail_prediction(tail_ldf)
