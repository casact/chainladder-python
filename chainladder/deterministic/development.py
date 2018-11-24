import numpy as np
import copy
from sklearn.base import BaseEstimator


class Development(BaseEstimator):
    def __init__(self, n_per=-1, avg_type='simple', by=None):
        self.n_per = n_per
        self.avg_type = avg_type
        self.by = by

    @property
    def ldf_(self):
        return self._param_property(0)

    @property
    def sigma_(self):
        return self._param_property(1)

    @property
    def std_err_(self):
        return self._param_property(2)

    @property
    def cdf_(self):
        if self.__params is not None:
            obj = self.ldf_
            cdf_ = np.flip(np.cumprod(np.flip(obj.triangle, axis=3),
                                      axis=3), axis=3)
            obj.triangle = cdf_
            obj.odims = ['CDF']
            return obj

    def _param_property(self, idx):
        if self.__params is not None:
            obj = copy.deepcopy(self.__params)
            temp = obj.triangle[:, :, idx:idx+1, :]
            obj.triangle = temp
            obj.odims = obj.odims[idx:idx+1]
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
        params = self._weighted_regresison_through_origin(tri_array)
        params = np.swapaxes(params, 2, 3)
        obj = copy.deepcopy(X)
        obj.triangle = params
        obj.odims = ['LDF', 'Sigma', 'Std Err']
        obj.ddims = np.array([f'{i+1}-{i+2}'
                              for i in range(len(obj.ddims)-1)])
        obj.key_labels = key_labels
        obj.kdims = kdims
        self.__params = obj
        self.latest_diagonal_ = temp.latest_diagonal
        return self

    def _weighted_regresison_through_origin(self, tri_array):
        if type(self.avg_type) is str:
            avg_type = [self.avg_type] * (tri_array.shape[3] - 1)
        else:
            avg_type = self.avg_type
        average = np.array(avg_type)
        weight_dict = {'regression': 0, 'volume': 1, 'simple': 2}
        x_ = tri_array[:, :, :, :-1]
        y_ = tri_array[:, :, :, 1:]
        val = np.array([weight_dict.get(item.lower(), 2)
                        for item in average])
        for i in [2, 1, 0]:
            val = np.repeat(np.expand_dims(val, 0), tri_array.shape[i], axis=0)
        val = np.nan_to_num(val * (y_ * 0 + 1))
        w_ = 1 / (x_**(val))
        ldf = np.nansum(w_*x_*y_, 2)/np.nansum((y_*0+1)*w_*x_*x_, 2)
        fitted_value = np.repeat(np.expand_dims(ldf, 2), x_.shape[0], 2)
        fitted_value = (fitted_value*x_*(y_*0+1))
        residual = (y_-fitted_value)*np.sqrt(w_)
        wss_residual = np.nansum(residual**2, 2)
        mse_denom = np.nansum(y_*0+1, 2)-1
        mse_denom[mse_denom == 0] = np.nan
        mse = wss_residual / mse_denom
        std_err = np.sqrt(mse/np.nansum(w_*x_*x_*(y_*0+1), axis=2))
        std_err = np.expand_dims(std_err, 3)
        ldf = np.expand_dims(ldf, 3)
        sigma = np.expand_dims(np.sqrt(mse), 3)
        return np.concatenate((ldf, sigma, std_err), 3)

    def predict(self, X):
        cdf_groupby = self.cdf_.triangle.copy()
        obj = copy.deepcopy(X)
        obj.triangle = (obj.triangle[:, :, 1:, :-1]*0+1) * \
            np.repeat(cdf_groupby, len(obj.odims)-1, 2)
        obj.ddims = self.cdf_.ddims
        obj.odims = obj.odims[1:]
        cdf_groupby = obj.latest_diagonal.triangle
        k, v, o, d = cdf_groupby.shape
        cdf_groupby = np.flip(np.append(np.flip(cdf_groupby, 2),
                              np.ones((k, v, 1, d)), 2), 2)
        X_groupby = X.latest_diagonal
        if self.by is not None:
            temp = X.latest_diagonal.groupby(self.by)
            cdf_groupby = cdf_groupby*temp.init
            cdf_groupby[cdf_groupby == 0] = np.nan
            cdf_groupby = np.nanmean(cdf_groupby, axis=0)
            X_groupby.triangle = cdf_groupby
        else:
            X_groupby.triangle = cdf_groupby
        X_groupby.triangle = np.append(X.latest_diagonal.triangle,
                                       X_groupby.triangle, 3)
        X_groupby.ddims = ['Latest', 'CDF']
        return X_groupby

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit_predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
