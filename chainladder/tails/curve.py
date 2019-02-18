"""
:ref:`TailCurve<curve>` is a curve
"""

from chainladder.tails import TailBase
from chainladder.utils import WeightedRegression
import numpy as np


class TailCurve(TailBase):
    """Allows for the entry of a constant tail factor to LDFs.

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
                 extrap_periods=100, errors='ignore'):
        self.curve = curve
        self.fit_period = fit_period
        self.extrap_periods = extrap_periods
        self.errors = errors

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
        _y = self.ldf_.triangle[..., :-1].copy()
        _w = np.zeros(_y.shape)
        if type(self.fit_period) is not slice:
            raise TypeError('fit_period must be slice.')
        else:
            _w[..., self.fit_period] = 1.0
        if self.errors == 'ignore':
            _w[_y <= 1.0] = 0
            _y[_y <= 1.0] = 1.01
        elif self.errors == 'raise' and np.any(y < 1.0):
            raise ZeroDivisionError('Tail fit requires all LDFs to be \
                                     greater than 1.0')
        _y = np.log(_y - 1)
        n_obs = _y.shape[-1]
        n_obs = X.shape[-1]-1
        k, v = X.shape[:2]
        _x = self._get_x(_w, _y)
        # Get LDFs
        coefs = WeightedRegression(_w, _x, _y, 3).fit()
        slope, intercept = coefs.slope_, coefs.intercept_
        extrapolate = np.cumsum(
            np.ones(tuple(list(_y.shape)[:-self._ave_period[0]-1] +
                    [self.extrap_periods])), -1) + n_obs
        tail = self._predict_tail(slope, intercept, extrapolate)
        self.ldf_.triangle[..., -tail.shape[-1]:] = tail
        sigma, std_err = self._get_tail_stats(X)
        self.sigma_.triangle[..., -1] = sigma[..., -1]
        self.std_err_.triangle[..., -1] = std_err[..., -1]
        self.slope_ = slope
        self.intercept_ = intercept
        return self

    def _get_tail_weighted_time_period(self, X):
        """ Method to approximate the weighted-average development age of tail
        using log-linear extrapolation

        Returns: float32
        """
        y = X.ldf_.triangle.copy()
        y[y <= 1] = np.nan
        reg = WeightedRegression(y=np.log(y - 1), axis=3).fit()
        tail = np.prod(self.ldf_.triangle[..., -self._ave_period[0]-1:],
                       -1, keepdims=True)
        reg = WeightedRegression(y=np.log(y - 1), axis=3).fit()
        time_pd = (np.log(tail-1)-reg.intercept_)/reg.slope_
        return time_pd

    def _get_tail_stats(self, X):
        """ Method to approximate the tail sigma using
        log-linear extrapolation applied to tail average period
        """
        time_pd = self._get_tail_weighted_time_period(X)
        reg = WeightedRegression(y=np.log(X.sigma_.triangle), axis=3).fit()
        sigma_ = np.exp(time_pd*reg.slope_+reg.intercept_)
        y = X.std_err_.triangle
        y[y == 0] = np.nan
        reg = WeightedRegression(y=np.log(y), axis=3).fit()
        std_err_ = np.exp(time_pd*reg.slope_+reg.intercept_)
        return sigma_, std_err_

    def _get_x(self, w, y):
        ''' For Exponential decay, no transformation on x is needed '''
        if self.curve == 'exponential':
            return None
        if self.curve == 'inverse_power':
            reg = WeightedRegression(w, None, y, 3, False).infer_x_w()
            return np.log(reg.x)

    def _predict_tail(self, slope, intercept, extrapolate):
        if self.curve == 'exponential':
            tail = np.exp(slope*extrapolate + intercept)
            ave = 1 + tail[..., :self._ave_period[0]]
            all = np.expand_dims(
                np.product(1 + tail[..., self._ave_period[0]:], -1), -1)
            tail = np.concatenate((ave, all), -1)
            return tail
        if self.curve == 'inverse_power':
            return np.expand_dims(np.product(
                1+np.exp(intercept)*(extrapolate**slope), -1), -1)
