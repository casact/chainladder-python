import numpy as np
import copy
from chainladder.tails.constant import Constant
from chainladder.tails.base import TailBase


class MethodBase(TailBase):
    def __init__(self):
        pass

    def validate_X(self, X):
        obj = copy.deepcopy(X)
        if TailBase not in set(X.__class__.__mro__):
            obj = Constant().fit(obj)
        self.X_ = obj.X_
        self.w_ = obj.w_
        self._params = obj._params
        self.average_ = obj.average_

    @property
    def full_cdf_(self):
        obj = copy.deepcopy(self.X_)
        obj.triangle = np.repeat(self.cdf_.triangle,
                                 self.X_.triangle.shape[2], 2)
        obj.ddims = self.cdf_.ddims
        obj.nan_override = True
        return obj

    @property
    def full_expectation_(self):
        obj = copy.deepcopy(self.X_)
        obj.triangle = self.ultimate_.triangle / self.full_cdf_.triangle
        obj.triangle = np.concatenate((obj.triangle, self.ultimate_.triangle), 3)
        obj.ddims = list(obj.ddims) + ['Ult']
        obj.nan_override = True
        return obj

    @property
    def ultimate_(self):
        obj = copy.deepcopy(self.X_)
        obj.triangle = np.repeat(self.X_.latest_diagonal.triangle,
                                 self.full_cdf_.shape[3], 3)
        obj.triangle = (self.full_cdf_.triangle*obj.triangle)*self.X_.nan_triangle()
        obj = obj.latest_diagonal
        obj.ddims = ['Ultimate']
        return obj

    @property
    def ibnr_(self):
        obj = copy.deepcopy(self.ultimate_)
        obj.triangle = self.ultimate_.triangle-self.X_.latest_diagonal.triangle
        obj.ddims = ['IBNR']
        return obj

    @property
    def full_triangle_(self):
        obj = copy.deepcopy(self.X_)
        w = 1-np.nan_to_num(obj.nan_triangle())
        obj.nan_override = True
        e_tri = np.repeat(self.ultimate_.triangle, obj.shape[3], 3) / \
            self.full_cdf_.triangle
        e_tri = e_tri * w
        obj.triangle = np.nan_to_num(obj.triangle) + e_tri
        obj.triangle = np.concatenate((obj.triangle,
                                       self.ultimate_.triangle), 3)
        obj.ddims = np.array([str(item) for item in obj.ddims]+['Ult'])
        return obj
