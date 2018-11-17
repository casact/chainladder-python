import numpy as np
import copy
from sklearn.base import BaseEstimator


class Development(BaseEstimator):
    def __init__(self, n_per=-1, avg_type='simple', by=None):
        self.n_per = n_per
        self.avg_type = avg_type
        self.by = by

    @property
    def cdf_(self):
        if self.ldf_ is not None:
            obj = copy.deepcopy(self.ldf_)
            cdf_ = np.flip(np.cumprod(np.flip(obj.triangle, axis=3),
                                      axis=3), axis=3)
            obj.triangle = cdf_
            obj.odims = ['CDF']
            return obj

    def fit(self, X, y=None, sample_weight=None):
        if self.by == -1:
            temp = X.sum()
        elif self.by is None:
            temp = copy.deepcopy(X)
        else:
            temp = X.groupby(self.by).sum()
        tri_array = temp.triangle
        key_labels = temp.key_labels
        kdims = temp.kdims
        weight = (tri_array * 0 + 1)[:, :, :, :-1]
        _x_reg = tri_array[:, :, :, :-1]
        _y_reg = tri_array[:, :, :, 1:]
        _x = np.nan_to_num(_x_reg * (_y_reg * 0 + 1))
        _y = np.nan_to_num(_y_reg)
        _w = np.nan_to_num(weight * (np.minimum(_y, 1) * (_y_reg * 0 + 1)))

        if type(self.avg_type) is str:
            avg_type = [self.avg_type] * (tri_array.shape[2] - 1)
        else:
            avg_type = self.avg_type

        average = np.array(avg_type)
        weight_dict = {'regression': 0, 'volume': 1, 'simple': 2}
        val = np.array([weight_dict.get(item.lower(), 2)
                        for item in average])
        for i in [2, 1, 0]:
            val = np.repeat(np.expand_dims(val, 0), tri_array.shape[i], axis=0)
        val = np.nan_to_num(val * (_y_reg * 0 + 1))
        _w = np.nan_to_num(_w / (np.maximum(np.nan_to_num(_x_reg), 1) *
                                 (_x_reg * 0 + 1))**(val))
        d = np.sum(_w * _x * _x, axis=2)
        n = np.sum(_w * _x * _y, axis=2)
        bool_safe = (d == 0)
        n[bool_safe] = 1.
        d[bool_safe] = 1.
        ldf_ = n/d
        obj = copy.deepcopy(X)
        obj.triangle = np.expand_dims(ldf_, axis=2)
        obj.odims = ['LDF']
        obj.ddims = np.array([f'{i+1}-{i+2}'
                              for i in range(len(obj.ddims))])
        obj.ddims[-1] = f'{len(obj.ddims)}-Ult'
        tail = 1.0
        obj.triangle = np.append(obj.triangle,
                                 np.ones((obj.shape[0], obj.shape[1],
                                         obj.shape[2], 1)) * tail, axis=3)
        obj.key_labels = key_labels
        obj.kdims = kdims
        self.ldf_ = obj
        self.latest_diagonal_ = temp.latest_diagonal
        return self

    def predict(self, X):
        cdf_groupby = self.cdf_.triangle.copy()
        X_groupby = X.latest_diagonal
        X_groupby.ddims = ['CDF']
        if self.by is not None:
            temp = X.latest_diagonal.groupby(self.by)
            cdf_groupby = (np.expand_dims(np.swapaxes(cdf_groupby, -1, -2), 1))
            cdf_groupby = cdf_groupby*temp.init
            cdf_groupby[cdf_groupby == 0] = np.nan
            cdf_groupby = np.nanmean(cdf_groupby, axis=0)
            X_groupby.triangle = np.flip(cdf_groupby, axis=2)
        else:
            X_groupby.triangle = np.swapaxes(cdf_groupby, -1, -2)
        return X.latest_diagonal, X_groupby

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit_predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
