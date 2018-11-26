''' Exponential Tail
Use constant.py as a template
reference:
    https://www.casact.org/pubs/forum/13fforum/02-Tail-Factors-Working-Party.pdf
'''

import numpy as np
from chainladder.tails import TailBase
from chainladder import weighted_regression


class Exponential(TailBase):
    """Exponential Tail Fit
    Parameters
    ----------
    n_per : int, default: -1
    The number of LDFs to be used in the computation of the tail factor. A
    value of -1 indicates LDFs from all ages will be used.
    extrap_per : int, default: 100
    The number of periods over which the LDFs will be extrapolated. The
    product of LDFs from the extrapolations will be used as the tail factor.
    errors : {'ignore', 'raise'}, default: 'ignore'
    Exponentail decay requires LDFs strictly larger than 1.0.  If LDFs are
    less than 1.0, they will be removed from the calculation ('ignore') or
    they will raise an error ('raise').
    sigma_est : {'lin', 'exp','mack'}, default: 'mack'
    Computation basis of the tail sigma. Linear ('lin') and Exponential
    ('exp') decay are applied directly to the LDF sigmas.  Mack ('mack')
    uses the Mack approximation for
    Attributes
    ----------
    ldf_ : Triangle

    cdf_ : Triangle
    Labels of each point

    Examples
    --------
    >>> from chainladder.tails import Exponential

    See also
    --------
    InversePower
    Another curve fit approach
    Notes
    ------
    None
    """

    def __init__(self, n_per=-1, extrap_per=100, errors='ignore', sigma_est='mack'):
        self.n_per = n_per
        self.extrap_per = extrap_per
        self.errors = errors
        self.sigma_est = sigma_est

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        params = self._params
        _y = params.triangle[:, :, 0:1, :]
        _w = np.ones(_y.shape)
        if self.errors == 'ignore':
            _w[_y <= 1.0] = 0
            _y[_y <= 1.0] = 1.01
        elif self.errors == 'raise' and np.any(y < 1.0):
            raise ZeroDivisionError('Exponential fit requires all LDFs to be \
                                     greater than 1.0')
        if type(self.n_per) is not int:
            raise TypeError('n_per must be an integer')
        elif self.n_per < -1 or self.n_per > _y.shape[3] or self.n_per == 0:
            raise ValueError('n_per must not exceed the length of the \
                              LDF array.')
        elif self.n_per > 1:
            _w[:, :, :, :-self.n_per] = 0
        _y = np.log(_y - 1)
        n_obs = _y.shape[3]
        k, v = params.triangle.shape[:2]
        slope, intercept = weighted_regression(_y, 3, None, _w)
        extrapolate = np.arange(n_obs + 1, n_obs + self.extrap_per)
        extrapolate = np.reshape(extrapolate, (1, 1, 1, 1, len(extrapolate)))
        tail = np.product(1 + np.exp(slope*extrapolate + intercept), 4)
        sigma = tail.copy() * 0
        std_err = tail.copy() * 0
        tail = np.concatenate((tail, sigma, std_err), 2)
        print(tail)
        params.triangle = np.append(params.triangle, tail, 3)
        return self
