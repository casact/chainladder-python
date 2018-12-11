import copy
import numpy as np
from chainladder.development import DevelopmentBase
from chainladder import WeightedRegression


class TailBase(DevelopmentBase):
    ''' Base class for all tail methods.  Tail objects are equivalent
        to development objects with an additional set of tail statistics'''

    def __init__(self):
        pass

    def fit(self, X, y=None, sample_weight=None):
        self.X_ = X.X_
        if DevelopmentBase not in set(X.__class__.__mro__):
            self.X_ = DevelopmentBase().fit(X.X_)
        self._params = copy.deepcopy(X._params)
        self.w_ = copy.deepcopy(X.w_)
        self._params.ddims = \
            np.append(self._params.ddims,
                      [str(int(len(self._params.ddims) + 1)) + '-Ult'])
        self.average_ = copy.deepcopy(X.average_)
        self.y_ = y
        self.sample_weight_ = sample_weight

    def predict(self, X):
        return self


class CurveFit(TailBase):
    ''' Base sub-class used for curve fit methods of tail factors '''
    def __init__(self, fit_per=slice(None, None, None), extrap_per=100,
                 errors='ignore'):
        self.fit_per = fit_per
        self.extrap_per = extrap_per
        self.errors = errors

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        params = self._params
        _y = params.triangle[:, :, 0:1, :]
        _w = np.zeros(_y.shape)
        if type(self.fit_per) is not slice:
            raise TypeError('fit_per must be slice.')
        else:
            _w[:, :, :, self.fit_per] = 1.0
        if self.errors == 'ignore':
            _w[_y <= 1.0] = 0
            _y[_y <= 1.0] = 1.01
        elif self.errors == 'raise' and np.any(y < 1.0):
            raise ZeroDivisionError('Tail fit requires all LDFs to be \
                                     greater than 1.0')
        _y = np.log(_y - 1)
        n_obs = _y.shape[3]
        k, v = params.triangle.shape[:2]
        _x = self.get_x(_w, _y)
        # Get LDFs
        coefs = WeightedRegression(_w, _x, _y, 3).fit()
        slope, intercept = coefs.slope_, coefs.intercept_
        extrapolate = np.cumsum(np.ones(tuple(list(_y.shape)[:-1] + [self.extrap_per])), -1) + n_obs
        tail = self.predict_tail(slope, intercept, extrapolate)
        sigma, std_err = self._get_tail_stats(tail)
        tail = np.concatenate((tail, sigma, std_err), 2)
        params.triangle = np.append(params.triangle, tail, 3)
        return self

    def _get_tail_weighted_time_period(self, tail):
        """ Method to approximate the weighted-average development age of tail
        using log-linear extrapolation

        Returns: float32
        """
        y = self.ldf_.triangle
        reg = WeightedRegression(y=np.log(y - 1), axis=3).fit()
        time_pd = (np.log(tail-1)-reg.intercept_)/reg.slope_
        return time_pd

    def _get_tail_stats(self, tail):
        """ Method to approximate the tail sigma using
        log-linear extrapolation applied to tail average period
        """
        time_pd = self._get_tail_weighted_time_period(tail)
        reg = WeightedRegression(y=np.log(self.sigma_.triangle), axis=3).fit()
        sigma_ = np.exp(time_pd*reg.slope_+reg.intercept_)
        reg = WeightedRegression(y=np.log(self.std_err_.triangle), axis=3).fit()
        std_err_ = np.exp(time_pd*reg.slope_+reg.intercept_)
        return sigma_, std_err_
