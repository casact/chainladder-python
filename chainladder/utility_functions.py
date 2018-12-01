"""
This module contains various utilities shared across most of the other
*chainladder* modules.

"""
import pandas as pd
import os
import numpy as np
from chainladder.triangle import Triangle


def load_dataset(key):
    """ Function to load datasets included in the chainladder package.

        Arguments:
        key: str
        The name of the dataset, e.g. RAA, ABC, UKMotor, GenIns, etc.

        Returns:
    	pandas.DataFrame of the loaded dataset.
   """
    path = os.path.dirname(os.path.abspath(__file__))
    origin = 'origin'
    development = 'development'
    values = 'values'
    keys = None
    if key.lower() in ['mcl', 'usaa', 'quarterly', 'auto']:
        values = ['incurred', 'paid']
    if key.lower() == 'casresearch':
        origin = 'AccidentYear'
        development = 'DevelopmentYear'
        keys = ['GRNAME', 'LOB']
        values = ['IncurLoss', 'CumPaidLoss', 'BulkLoss', 'EarnedPremDIR',
                  'EarnedPremCeded', 'EarnedPremNet']
    if key.lower() in ['liab', 'auto']:
        keys = ['lob']
    df = pd.read_pickle(os.path.join(path, 'data', key + '.pkl'))
    return Triangle(df, origin=origin, development=development,
                    values=values, keys=keys)




class WeightedRegression:
    ''' Helper class that fits a system of regression equations
        simultaneously on a multi-dimensional array.  Look into
        SUR as a replacement.
    '''
    def __init__(self, w=None, x=None, y=None, axis=None, thru_orig=False):
        self.x = x
        self.y = y
        self.w = w
        self.axis = axis
        self.thru_orig = thru_orig

    def infer_x_w(self):
        if self.w is None:
            self.w = np.ones(self.y.shape)
        if self.x is None:
            self.x = np.reshape(np.arange(self.y.shape[self.axis]),
                                (1, 1, 1, self.y.shape[self.axis])) + 1.
            temp = np.array(self.x.shape)
            temp = np.where(temp == temp.max())[0][0]
            self.x = np.swapaxes(self.x, temp, self.axis)
        return self

    def fit(self):
        if self.x is None:
            self.infer_x_w()
        if self.thru_orig:
            self.fit_OLS_thru_orig()
        else:
            self.fit_OLS()
        return self

    def fit_OLS(self):
        ''' Given a set of w, x, y, and an axis, this Function
            returns OLS slope and intercept.
        '''
        w, x, y, axis = self.w.copy(), self.x.copy(), self.y.copy(), self.axis

        x[w == 0] = np.nan
        y[w == 0] = np.nan
        slope = (np.nansum(w*x*y, axis) - np.nansum(x*w, axis)*np.nanmean(y, axis)) / \
            (np.nansum(w*x*x, axis) - np.nanmean(x, axis)*np.nansum(w*x, axis))
        intercept = np.nanmean(y, axis) - slope * np.nanmean(x, axis)
        self.slope_ = np.expand_dims(slope, -1)
        self.intercept_ = np.expand_dims(intercept, -1)
        return self

    def fit_OLS_thru_orig(self):
        w, x, y, axis = self.w, self.x, self.y, self.axis
        coef = np.nansum(w*x*y, axis)/np.nansum((y*0+1)*w*x*x, axis)
        fitted_value = np.repeat(np.expand_dims(coef, axis), x.shape[axis], axis)
        fitted_value = (fitted_value*x*(y*0+1))
        residual = (y-fitted_value)*np.sqrt(w)
        wss_residual = np.nansum(residual**2, axis)
        mse_denom = np.nansum(y*0+1, axis)-1
        mse_denom[mse_denom == 0] = np.nan
        mse = wss_residual / mse_denom
        std_err = np.sqrt(mse/np.nansum(w*x*x*(y*0+1), axis))
        std_err = np.expand_dims(std_err, -1)
        coef = np.expand_dims(coef, -1)
        sigma = np.expand_dims(np.sqrt(mse), -1)
        self.slope_ = coef
        self.sigma_ = sigma
        self.std_err_ = std_err
        return self

    def sigma_fill(self):
        ''' This Function is designed to take an array of sigmas and does log-
            linear extrapolation where n_obs = 1 and sigma cannot be calculated.
        '''
        self.sigma_ = self.loglinear_interpolation(self.sigma_)
        return self

    def std_err_fill(self):
        y = self.sigma_*np.sqrt(np.expand_dims(self.w[:, :, 0, :], -1))
        w = np.nan_to_num(self.std_err_*0+1)
        std_err_ = y * (1 - w)
        self.std_err_ = np.nan_to_num(self.std_err_) + std_err_
        return self

    def loglinear_interpolation(self, y):
        ''' Use Cases: generally for filling in last element of sigma_
        '''
        ly = np.log(y)
        w = np.nan_to_num(ly*0+1)
        reg = WeightedRegression(y=ly, w=w, axis=self.axis, thru_orig=False).fit()
        slope, intercept = reg.slope_, reg.intercept_
        fill_ = np.exp(reg.x*slope+intercept)*(1-w)
        return np.nan_to_num(y) + fill_
