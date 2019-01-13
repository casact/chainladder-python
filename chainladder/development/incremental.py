from sklearn.base import BaseEstimator
from chainladder.development.base import Development
import numpy as np
import copy


class IncrementalAdditive(BaseEstimator):
    def __init__(self, trend=0.0, n_periods=-1):
        self.trend = trend
        self.n_periods = n_periods

    def fit(self, X, y=None, sample_weight=None):
        obj = X/sample_weight
        x = obj.trend(self.trend, axis='origin')
        w_ = Development(n_periods=self.n_periods-1).fit(x).w_
        w_[w_ == 0] = np.nan
        w_ = np.concatenate((w_, (w_[..., -1:]*x.nan_triangle())[..., -1:]),
                            axis=-1)
        y_ = np.repeat(np.expand_dims(np.nanmean(w_*x.triangle, axis=-2), -2),
                       len(x.odims), -2)
        obj = copy.deepcopy(x)
        keeps = 1-np.nan_to_num(x.nan_triangle())+np.nan_to_num(x.get_latest_diagonal(compress=False).triangle[0,0,...]*0+1)
        obj.triangle = (1+self.trend)**np.flip((np.abs(np.expand_dims(np.arange(obj.shape[-2]), 0).T -
                     np.expand_dims(np.arange(obj.shape[-2]), 0))),0)*y_*keeps
        obj.triangle = obj.triangle*(x.expand_dims(1-np.nan_to_num(x.nan_triangle())))+np.nan_to_num((X/sample_weight).triangle)
        obj.triangle[obj.triangle == 0] = np.nan
        obj.nan_override = True
        self.incremental_ = obj*sample_weight
        self.ldf_ = obj.incr_to_cum().link_ratio
        return self

    def transform(self, X):
        X.cdf_ = self.cdf_
        X.ldf_ = self.ldf_
        X.sigma_ = self.cdf_*0
        X.std_err_ = self.cdf_*0
        X.incremental_ = self.incremental_
        return X

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.transform(X)

    @property
    def cdf_(self):
        if self.__dict__.get('ldf_', None) is None:
            return
        else:
            obj = copy.deepcopy(self.ldf_)
            cdf_ = np.flip(np.cumprod(np.flip(obj.triangle, -1), -1), -1)
            obj.triangle = cdf_
            return obj
