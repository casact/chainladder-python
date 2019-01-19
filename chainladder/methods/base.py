import numpy as np
import copy
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development


class MethodBase(BaseEstimator):
    def __init__(self):
        pass

    def validate_X(self, X):
        obj = copy.deepcopy(X)
        if obj.__dict__.get('ldf_', None) is None:
            obj = Development().fit_transform(obj)
        if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
            obj = TailConstant().fit_transform(obj)
        self.cdf_ = obj.cdf_
        self.ldf_ = obj.ldf_
        self.average_ = obj.average_
        return obj

    def fit(self, X, y=None, sample_weight=None):
        """Applies the chainladder technique to triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored
        sample_weight : None
            ignored
        """
        self.X_ = self.validate_X(X)
        return self

    def predict(self, X):
        """Predicts the chainladder ultimate on a new triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        Returns
        -------
        X_new: Triangle

        """
        obj = copy.deepcopy(self)
        obj.X_ = copy.deepcopy(X)
        return obj

    @property
    def full_expectation_(self):
        obj = copy.deepcopy(self.X_)
        obj.triangle = self.ultimate_.triangle / self.cdf_.triangle
        obj.triangle = \
            np.concatenate((obj.triangle, self.ultimate_.triangle), 3)
        obj.ddims = list(obj.ddims) + ['Ult']
        obj.nan_override = True
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
            self.cdf_.triangle
        e_tri = e_tri * w
        obj.triangle = np.nan_to_num(obj.triangle) + e_tri
        obj.triangle = np.concatenate((obj.triangle,
                                       self.ultimate_.triangle), 3)
        obj.ddims = np.array([str(item) for item in obj.ddims]+['Ult'])
        return obj
