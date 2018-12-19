import numpy as np
import copy
from sklearn.base import BaseEstimator
from chainladder.tails.constant import Constant


class MethodBase(BaseEstimator):
    def __init__(self):
        pass

    def validate_X(self, X):
        obj = copy.deepcopy(X)
        if len(obj.ddims) - len(obj.triangle_d.ddims) == 1:
            obj = Constant().fit_transform(obj)
        self._params = X.triangle_d
        self.average_ = X.average_
        return obj

    @property
    def full_cdf_(self):
        obj = copy.deepcopy(self.X_)
        obj.triangle = np.repeat(obj.cdf_.triangle,
                                 self.X_.triangle.shape[2], 2)
        obj.ddims = obj.cdf_.ddims
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
