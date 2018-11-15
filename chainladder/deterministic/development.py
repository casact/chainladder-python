import numpy as np
from sklearn.base import BaseEstimator


class Development(BaseEstimator):
    def __init__(self, n_per=-1, avg_type='simple'):
        self.n_per = n_per
        self.avg_type = avg_type

    @property
    def cdf_(self):
        if self.ldf_ is not None:
            return np.flip(np.cumprod(np.flip(self.ldf_, axis=2),
                                      axis=2), axis=2)

    def fit(self, X, y=None):
        weight = X.triangle * 0 + 1
        tri_array = X.triangle
        _x_reg = tri_array[:, :, :, :-1]
        _x = np.nan_to_num(_x_reg *
                           (tri_array[:, :, :, 1:] * 0 + 1))
        _y = np.nan_to_num(tri_array[:, :, :, 1:])
        _w = np.nan_to_num(weight[:, :, :, :-1] *
                           (np.minimum(_y, 1) *
                            (tri_array[:, :, :, 1:] * 0 + 1)))

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
        val = np.nan_to_num(val * (tri_array[:, :, :, 1:] * 0 + 1))
        _w = np.nan_to_num(_w / (np.maximum(np.nan_to_num(_x_reg), 1) *
                                 (_x_reg * 0 + 1))**(val))
        ldf_ = np.sum(_w * _x * _y, axis=2)/np.sum(_w * _x * _x, axis=2)
        self.ldf_ = ldf_

    def predict(self, X):
        latest = X.latest_diagonal
        cdf_ = np.flip(self.cdf_, axis=2)
        tail = 1.0
        cdf_ = np.append(np.array([tail]), cdf_ * tail)
        return latest * cdf_

    def fit_transform(self):
        pass

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)
