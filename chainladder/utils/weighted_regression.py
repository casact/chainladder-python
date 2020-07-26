# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
from sklearn.base import BaseEstimator

class WeightedRegression(BaseEstimator):
    ''' Helper class that fits a system of regression equations
        simultaneously on a multi-dimensional array.  Look into
        SUR as a replacement.
    '''
    def __init__(self, axis=None, thru_orig=False):
        self.axis = axis
        self.thru_orig = thru_orig

    def infer_x_w(self):
        xp = cp.get_array_module(self.y)
        if self.w is None:
            self.w = xp.ones(self.y.shape)
        if self.x is None:
            self.x = xp.cumsum(xp.ones(self.y.shape), self.axis)
        return self

    def fit(self, X, y=None, sample_weight=None):
        self.x = X
        self.y = y
        self.w = sample_weight
        if self.x is None:
            self.infer_x_w()
        if self.thru_orig:
            self._fit_OLS_thru_orig()
        else:
            self._fit_OLS()
        return self

    def _fit_OLS(self):
        ''' Given a set of w, x, y, and an axis, this Function
            returns OLS slope and intercept.
            TODO:
                Make this work with n_periods = 1 without numpy warning.
        '''
        w, x, y, axis = self.w.copy(), self.x.copy(), self.y.copy(), self.axis
        xp = cp.get_array_module(x)
        if xp != sp:
            x[w == 0] = xp.nan
            y[w == 0] = xp.nan
        else:
            w2 = w.copy()
            w2.fill_value = sp.nan
            x, y  = x*sp(w2), y*sp(w2)
        slope = (
            (xp.nansum(w*x*y, axis)-xp.nansum(x*w, axis)*xp.nanmean(y, axis)) /
            (xp.nansum(w*x*x, axis)-xp.nanmean(x, axis)*xp.nansum(w*x, axis)))
        intercept = xp.nanmean(y, axis) - slope * xp.nanmean(x, axis)
        self.slope_ = slope[..., None]
        self.intercept_ = intercept[..., None]
        return self

    def _fit_OLS_thru_orig(self):
        w, x, y, axis = self.w, self.x, self.y, self.axis
        xp = cp.get_array_module(x)
        coef = xp.nansum(w*x*y, axis)/xp.nansum((y*0+1)*w*x*x, axis)
        fitted_value = xp.repeat(xp.expand_dims(coef, axis),
                                 x.shape[axis], axis)
        fitted_value = (fitted_value*x*(y*0+1))
        residual = (y-fitted_value)*xp.sqrt(w)
        wss_residual = xp.nansum(residual**2, axis)
        mse_denom = xp.nansum((y*0+1)*(w!=0), axis)-1
        if xp != sp:
            mse_denom[mse_denom == 0] = xp.nan
        else:
            mse_denom.fill_value = sp.nan
            mse_denom.data[mse_denom.data==0] = sp.nan
            mse_denom = sp(mse_denom, prune=True)
        mse = wss_residual / mse_denom
        std_err = xp.sqrt(mse/xp.nansum(w*x*x*(y*0+1), axis))
        std_err = std_err[..., None]
        if xp != sp:
            std_err[std_err == 0] = xp.nan
        coef = coef[..., None]
        sigma = xp.sqrt(mse)[..., None]
        self.slope_ = coef
        self.sigma_ = sigma
        self.std_err_ = std_err
        return self

    def sigma_fill(self, interpolation):
        ''' This Function is designed to take an array of sigmas and does log-
            linear extrapolation where n_obs=1 and sigma cannot be calculated.
        '''
        if interpolation == 'log-linear':
            self.sigma_ = self.loglinear_interpolation(self.sigma_)
        if interpolation == 'mack':
            self.sigma_ = self.mack_interpolation(self.sigma_)
        return self

    def std_err_fill(self):
        '''currently handled in development.py which doesn't feel right'''
        return self

    def loglinear_interpolation(self, y):
        ''' Use Cases: generally for filling in last element of sigma_
        '''
        xp = cp.get_array_module(y)
        if xp != sp:
            y[y == 0] = xp.nan
        else:
            y.fill_value = sp.nan
            y  = sp(y)
        ly = xp.log(y)
        w = xp.nan_to_num(ly*0+1)
        reg = WeightedRegression(self.axis, False).fit(None, ly, w)
        slope, intercept = reg.slope_, reg.intercept_
        fill_ = xp.exp(reg.x*slope+intercept)*(1-w)
        out = xp.nan_to_num(y) + xp.nan_to_num(fill_)
        if xp == sp:
            out.fill_value = sp.nan
            return sp(out)
        else:
            return out

    def mack_interpolation(self, y):
        ''' Use Mack's approximation to fill last element of sigma_ which is the
            same as loglinear extrapolation using the preceding two element to
            the missing value. This function needs a recursive definition...
        '''
        xp = cp.get_array_module(y)
        w = xp.nan_to_num(y*0+1)
        slicer_n, slicer_d, slicer_a = \
            ([slice(None)]*4), ([slice(None)]*4), ([slice(None)]*4)
        slicer_n[self.axis], slicer_d[self.axis], slicer_a[self.axis] = \
            slice(1, -1, 1), slice(0, -2, 1), slice(0, 2, 1)
        slicer_n, slicer_d, slicer_a = (tuple(slicer_n), tuple(slicer_d),
                                        tuple(slicer_a))
        fill_ = xp.sqrt(
            abs(xp.minimum((y[slicer_n]**4 / y[slicer_d]**2),
                xp.minimum(y[slicer_d]**2, y[slicer_n]**2))))
        fill_ = xp.concatenate((w[slicer_a], xp.nan_to_num(fill_)), axis=self.axis)*(1-w)
        if xp == sp:
            out = xp.nan_to_num(y) + fill_
            out.fill_value = sp.nan
            return sp(out)
        else:
            return xp.nan_to_num(y) + fill_
