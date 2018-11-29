import copy
import numpy as np
from chainladder.development import Development
from chainladder import WeightedRegression


class TailBase(Development):
    ''' Base class for all tail methods.  Tail objects are equivalent
        to development objects with an additional set of tail statistics'''
    def fit(self, X, y=None, sample_weight=None):
        self.X_ = X.X_
        self._params = copy.deepcopy(X._params)
        self._params.ddims = \
            np.append(self._params.ddims,
                      [str(int(len(self._params.ddims) + 1)) + '-Ult'])
        self.y_ = y
        self.sample_weight_ = sample_weight

    def predict(self, X):
        return self

    def _get_tail_sigma(self):
        """ Method to produce the sigma of the tail factor
        """
        sigma_ = self._params.triangle[:,:,1:2,:]
        # Mack Method
        if self.sigma_est == 'mack':
            y = sigma_
            return np.sqrt(abs(min((y[:, : , :, -1]**4 / y[:, : , :, -2]**2),
                                    min(y[:, : , :, -2]**2, y[:, : , :, -1]**2))))

class CurveFit(TailBase):
    ''' Base sub-class used for curve fit methods of tail factors '''
    def __init__(self, fit_per=slice(None, None, None), extrap_per=100,
                 errors='ignore', sigma_interpolation='loglinear'):
        self.fit_per = fit_per
        self.extrap_per = extrap_per
        self.errors = errors
        self.sigma_interpolation = sigma_interpolation

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
        _x = self.get_x(_w,_y)
        coefs = WeightedRegression(_w, _x, _y, 3).fit()
        slope, intercept = coefs.slope_, coefs.intercept_
        extrapolate = np.arange(n_obs + 1, n_obs + self.extrap_per)
        extrapolate = np.reshape(extrapolate, (1, 1, 1, 1, len(extrapolate)))
        tail = self.predict_tail(slope, intercept, extrapolate)
        sigma = tail.copy() * np.nan
        std_err = tail.copy() * 0
        tail = np.concatenate((tail, sigma, std_err), 2)
        params.triangle = np.append(params.triangle, tail, 3)
        return self
