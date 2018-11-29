import numpy as np
from chainladder.tails import TailBase


class Constant(TailBase):
    ''' This class must take an object of type Development
        Development will already have LDFs fit.

        TODO: Expand constant to be an array that varies by kdim and vdim
    '''
    def __init__(self, tail=1.0):
        self.tail = tail

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        sigma_ = 0
        std_err_ = 0
        params = self._params
        k, v, o, d = params.triangle.shape
        tail = np.reshape(np.array([self.tail, sigma_, std_err_]), (1, 1, 3, 1))
        tail = np.repeat(np.repeat(tail, k, 0), v, 1)
        params.triangle = np.append(params.triangle, tail, 3)
        self._params = params
        return self
