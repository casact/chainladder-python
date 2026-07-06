# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder.utils.sparse import sp
from sklearn.base import BaseEstimator
import warnings


class WeightedRegression(BaseEstimator):
    """Helper class that fits a system of regression equations
    as a closed-form solution.  This greatly speeds up
    the implementation of the Mack stochastic properties.
    """

    def __init__(self, axis=None, thru_orig=False, xp=None, average=None):
        self.axis = axis
        self.thru_orig = thru_orig
        self.xp = xp
        self.average = average

    def infer_x_w(self):
        xp = self.xp
        if self.w is None:
            self.w = xp.ones(self.y.shape)
        if self.x is None:
            self.x = xp.cumsum(xp.ones(self.y.shape), self.axis)
        return self

    def fit(self, X, y=None, sample_weight=None, average=None):
        self.x = X
        self.y = y
        self.w = sample_weight
        self.average = average

        if self.x is None:
            self.infer_x_w()

        if self.thru_orig:
            self._fit_OLS_thru_orig()
        else:
            self._fit_OLS()

        return self

    def _fit_OLS_thru_orig(self):
        from chainladder.utils.utility_functions import num_to_nan

        x = self.x
        y = self.y
        w = self.w
        axis = self.axis
        average_ = self.average
        xp = self.xp

        if average_ is None:
            self.exponent_ = xp.nan_to_num(y * 0)
            denominator = num_to_nan(xp.nansum((y * 0 + 1) * w * x * x, axis))
            coef = num_to_nan(xp.nansum(w * x * y, axis)) / denominator

        else:
            # calculate the coef using regression framework
            exponent_map = {"regression": 0, "volume": 1, "simple": 2, "geometric": 1}
            exponent = xp.nan_to_num(
                xp.array([exponent_map[a] for a in average_[0, 0, 0]]) * (y * 0 + 1)
            )
            self.exponent_ = exponent
            w = num_to_nan(w / (x**exponent))
            denominator = num_to_nan(xp.nansum((y * 0 + 1) * w * x * x, axis))
            reg_coef = num_to_nan(xp.nansum(w * x * y, axis)) / denominator

            # special case for geometric average, still using the framework,
            # but using the log link function and taking the differences
            is_geo = xp.array([a == "geometric" for a in average_[0, 0, 0]])
            if is_geo.any():

                if xp.any((y == 0) & (x == 0)):
                    warnings.warn(
                        "Zero values present in triangle data used for geometric "
                        "averaging; link ratios calculated may be invalid. "
                        "Consider using a different average method.",
                        UserWarning,
                    )

                w_geo = num_to_nan(self.w)
                geo_coef = xp.exp(
                    xp.nanmean(w_geo * xp.log(y), axis)
                    - xp.nanmean((y * 0 + 1) * w_geo * xp.log(x), axis)
                )

                # consolidate
                coef = xp.where(
                    is_geo.reshape((1,) * (reg_coef.ndim - 1) + (is_geo.shape[-1],)),
                    geo_coef,
                    reg_coef,
                )
            else:
                coef = reg_coef

        fitted_value = xp.repeat(xp.expand_dims(coef, axis), x.shape[axis], axis)
        fitted_value = fitted_value * x * (y * 0 + 1)

        residual = (y - fitted_value)

        wss_residual = xp.nansum(residual**2 * w, axis)
        mse_denom = xp.nansum((y * 0 + 1) * (xp.nan_to_num(w) != 0), axis) - 1
        mse_denom = num_to_nan(mse_denom)
        mse = wss_residual / mse_denom

        std_err = xp.sqrt(mse / denominator)
        sigma = std_err * xp.sqrt(mse_denom + 1)
        
        coef = coef[..., None]
        sigma = sigma[..., None]
        std_err = std_err[..., None]
        
        self._w_reg = w

        self.slope_ = coef
        self.sigma_ = sigma
        self.std_err_ = std_err

        return self

    def _fit_OLS(self):
        """Given a set of w, x, y, and an axis, this Function
        returns OLS slope and intercept.
        TODO:
            Make this work with n_periods = 1 without numpy warning.
        """
        from chainladder.utils.utility_functions import num_to_nan

        w, x, y, axis = self.w.copy(), self.x.copy(), self.y.copy(), self.axis
        xp = self.xp
        if xp != sp:
            x[w == 0] = xp.nan
            y[w == 0] = xp.nan
        else:
            w2 = w.copy()
            w2 = sp.COO(
                data=w2.data, coords=w2.coords, fill_value=sp.nan, shape=w2.shape
            )
            x, y = x * w2, y * w2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            slope = num_to_nan(
                xp.nansum(w * x * y, axis)
                - xp.nansum(x * w, axis) * xp.nanmean(y, axis)
            ) / num_to_nan(
                xp.nansum(w * x * x, axis)
                - xp.nanmean(x, axis) * xp.nansum(w * x, axis)
            )
            intercept = xp.nanmean(y, axis) - slope * xp.nanmean(x, axis)

        self.slope_ = slope[..., None]
        self.intercept_ = intercept[..., None]

        return self

    def sigma_fill(self, interpolation):
        """This Function is designed to take an array of sigmas and does log-
        linear extrapolation where n_obs=1 and sigma cannot be calculated.
        """
        if interpolation == "log-linear":
            self.sigma_ = self.loglinear_interpolation(self.sigma_)
        if interpolation == "mack":
            self.sigma_ = self.mack_interpolation(self.sigma_)
        return self

    def std_err_fill(self):
        """Fill undefined std_err_ where regression has insufficient observations."""
        xp = self.xp
        self.std_err_ = xp.nan_to_num(self.std_err_) + xp.nan_to_num(
            (1 - xp.nan_to_num(self.std_err_ * 0 + 1))
            * self.sigma_
            / xp.sqrt(self.x ** (2 - self.exponent_))[..., 0:1, :].swapaxes(-1, -2)
        )
        return self

    def loglinear_interpolation(self, y):
        """Use Cases: generally for filling in last element of sigma_"""
        from chainladder.utils.utility_functions import num_to_nan

        xp = self.xp
        ly = y.copy()
        ly = xp.log(xp.where(ly == 0, 1e-320, ly))
        w = xp.nan_to_num(ly * 0 + 1)
        reg = WeightedRegression(self.axis, False, xp=xp).fit(None, ly, w)
        slope, intercept = reg.slope_, reg.intercept_
        fill_ = xp.exp(reg.x * slope + intercept) * (1 - w)
        out = xp.nan_to_num(y) + xp.nan_to_num(fill_)
        return num_to_nan(out)

    def mack_interpolation(self, y):
        """Use Mack's approximation to fill last element of sigma_ which is the
        same as loglinear extrapolation using the preceding two element to
        the missing value. This function needs a recursive definition...
        """
        from chainladder.utils.utility_functions import num_to_nan

        xp = self.xp
        ly = y.copy()
        ly = xp.where(ly == 0, 1e-320, ly)
        w = xp.nan_to_num(ly * 0 + 1)
        slicer_n, slicer_d, slicer_a = (
            ([slice(None)] * 4),
            ([slice(None)] * 4),
            ([slice(None)] * 4),
        )
        slicer_n[self.axis], slicer_d[self.axis], slicer_a[self.axis] = (
            slice(1, -1, 1),
            slice(0, -2, 1),
            slice(0, 2, 1),
        )
        slicer_n, slicer_d, slicer_a = (
            tuple(slicer_n),
            tuple(slicer_d),
            tuple(slicer_a),
        )
        fill_ = xp.sqrt(
            abs(
                xp.minimum(
                    (ly[slicer_n] ** 4 / ly[slicer_d] ** 2),
                    xp.minimum(ly[slicer_d] ** 2, ly[slicer_n] ** 2),
                )
            )
        )
        fill_ = xp.concatenate((w[slicer_a], xp.nan_to_num(fill_)), axis=self.axis) * (
            1 - w
        )
        out = xp.nan_to_num(ly) + fill_
        return num_to_nan(out)
