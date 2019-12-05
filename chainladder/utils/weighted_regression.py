# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
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
        if self.w is None:
            self.w = np.ones(self.y.shape)
        if self.x is None:
            self.x = np.cumsum(np.ones(self.y.shape), self.axis)
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
        x[w == 0] = np.nan
        y[w == 0] = np.nan
        slope = (
            (np.nansum(w*x*y, axis)-np.nansum(x*w, axis)*np.nanmean(y, axis)) /
            (np.nansum(w*x*x, axis)-np.nanmean(x, axis)*np.nansum(w*x, axis)))
        intercept = np.nanmean(y, axis) - slope * np.nanmean(x, axis)
        self.slope_ = np.expand_dims(slope, -1)
        self.intercept_ = np.expand_dims(intercept, -1)
        return self

    def _fit_OLS_thru_orig(self):
        w, x, y, axis = self.w, self.x, self.y, self.axis
        coef = np.nansum(w*x*y, axis)/np.nansum((y*0+1)*w*x*x, axis)
        fitted_value = np.repeat(np.expand_dims(coef, axis),
                                 x.shape[axis], axis)
        fitted_value = (fitted_value*x*(y*0+1))
        residual = (y-fitted_value)*np.sqrt(w)
        wss_residual = np.nansum(residual**2, axis)
        mse_denom = np.nansum(y*0+1, axis)-1
        mse_denom[mse_denom == 0] = np.nan
        mse = wss_residual / mse_denom
        std_err = np.sqrt(mse/np.nansum(w*x*x*(y*0+1), axis))
        std_err = np.expand_dims(std_err, -1)
        std_err[std_err == 0] = np.nan
        coef = np.expand_dims(coef, -1)
        sigma = np.expand_dims(np.sqrt(mse), -1)
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
        y[y == 0] = np.nan
        ly = np.log(y)
        w = np.nan_to_num(ly*0+1)
        reg = WeightedRegression(self.axis, False).fit(None, ly, w)
        slope, intercept = reg.slope_, reg.intercept_
        fill_ = np.exp(reg.x*slope+intercept)*(1-w)
        return np.nan_to_num(y) + fill_

    def mack_interpolation(self, y):
        ''' Use Mack's approximation to fill last element of sigma_ which is the
            same as loglinear extrapolation using the preceding two element to
            the missing value. This function needs a recursive definition...
        '''
        w = np.nan_to_num(y*0+1)
        slicer_n, slicer_d, slicer_a = \
            ([slice(None)]*4), ([slice(None)]*4), ([slice(None)]*4)
        slicer_n[self.axis], slicer_d[self.axis], slicer_a[self.axis] = \
            slice(1, -1, 1), slice(0, -2, 1), slice(0, 2, 1)
        slicer_n, slicer_d, slicer_a = (tuple(slicer_n), tuple(slicer_d),
                                        tuple(slicer_a))
        fill_ = np.sqrt(
            abs(np.minimum((y[slicer_n]**4 / y[slicer_d]**2),
                np.minimum(y[slicer_d]**2, y[slicer_n]**2))))
        fill_ = np.concatenate((w[slicer_a], fill_), axis=self.axis)*(1-w)
        return np.nan_to_num(y) + fill_
