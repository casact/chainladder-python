import numpy as np
import copy
from sklearn.base import BaseEstimator
from chainladder import WeightedRegression


class DevelopmentBase(BaseEstimator):
    def __init__(self, n_periods=-1, average='volume',
                 sigma_interpolation='log-linear'):
        self.n_periods = n_periods
        self.average = average
        self.sigma_interpolation = sigma_interpolation

    def _assign_n_periods_weight(self, X):
        if type(self.n_periods) is int:
            return self._assign_n_periods_weight_int(X, self.n_periods)[..., :-1]
        elif type(self.n_periods) is list:
            if len(self.n_periods) != X.triangle.shape[-1]-1:
                raise ValueError(f'n_periods list must be of lenth {X.triangle.shape[-1]-1}.')
            else:
                return self._assign_n_periods_weight_list(X)
        else:
            raise ValueError('n_periods must be of type <int> or <list>')

    def _assign_n_periods_weight_list(self, X):
        dict_map = {item: self._assign_n_periods_weight_int(X, item)
                    for item in set(self.n_periods)}
        conc = [dict_map[item][..., num:num+1, :]
                for num, item in enumerate(self.n_periods)]
        return np.swapaxes(np.concatenate(tuple(conc), -2), -2, -1)

    def _assign_n_periods_weight_int(self, X, n_periods):
        ''' Zeros out weights depending on number of periods desired
            Only works for type(n_periods) == int
        '''
        if n_periods < 1 or n_periods >= X.shape[-2] - 1:
            return X.triangle*0+1
        else:
            flip_nan = np.nan_to_num(X.triangle*0+1)
            k, v, o, d = flip_nan.shape
            w = np.concatenate((1-flip_nan[..., -(o-n_periods-1):, :],
                                np.ones((k, v, n_periods+1, d))), 2)*flip_nan
            return w*X.expand_dims(X.nan_triangle())

    def fit(self, X, y=None, sample_weight=None):
        # Capture the fit inputs
        tri_array = X.triangle.copy()
        tri_array[tri_array == 0] = np.nan
        if type(self.average) is str:
            average = [self.average] * (tri_array.shape[-1] - 1)
        else:
            average = self.average
        average = np.array(average)
        self.average_ = average
        weight_dict = {'regression': 2, 'volume': 1, 'simple': 0}
        _x = tri_array[..., :-1]
        _y = tri_array[..., 1:]
        val = np.array([weight_dict.get(item.lower(), 2)
                        for item in average])
        for i in [2, 1, 0]:
            val = np.repeat(np.expand_dims(val, 0), tri_array.shape[i], axis=0)
        val = np.nan_to_num(val * (_y * 0 + 1))
        _w = self._assign_n_periods_weight(X) / (_x**(val))
        self.w_ = self._assign_n_periods_weight(X)
        params = WeightedRegression(_w, _x, _y, axis=2, thru_orig=True) \
            .fit().sigma_fill(self.sigma_interpolation)
        params.std_err_ = np.nan_to_num(params.std_err_) + \
            np.nan_to_num((1-np.nan_to_num(params.std_err_*0+1)) *
            params.sigma_/np.swapaxes(np.sqrt(_x**(2-val))[..., 0:1, :], -1, -2))
        params = np.concatenate((params.slope_,
                                 params.sigma_,
                                 params.std_err_), 3)
        params = np.swapaxes(params, 2, 3)
        self.ldf_ = self._param_property(X, params, 0)
        self.sigma_ = self._param_property(X, params, 1)
        self.std_err_ = self._param_property(X, params, 2)
        return self

    def predict(self, X):
        ''' If X and self are of different shapes, align self
            to X, else return self

            Returns:
                X, residual - (k,v,o,d) original array
                ldf, std_err, sigma - (k,v,1,d)
                residual - (k,v,o,d)
                latest_diagonal - (k,v,o,1)
        '''
        X.std_err_ = self.std_err_
        X.cdf_ = self.cdf_
        X.ldf_ = self.ldf_
        X.sigma_ = self.sigma_
        X.sigma_interpolation = self.sigma_interpolation
        X.average_ = self.average_
        X.w_ = self.w_
        return X

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit_predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    @property
    def cdf_(self):
        if self.__dict__.get('ldf_', None) is None:
            return
        else:
            obj = copy.deepcopy(self.ldf_)
            cdf_ = np.flip(np.cumprod(np.flip(obj.triangle, -1), -1), -1)
            obj.triangle = cdf_
            return obj

    def _param_property(self, X, params, idx):
        obj = copy.deepcopy(X)
        obj.triangle = np.ones(X.shape)[..., :-1]*params[..., idx:idx+1, :]
        obj.ddims = np.array([f'{obj.ddims[i]}-{obj.ddims[i+1]}'
                              for i in range(len(obj.ddims)-1)])
        obj.nan_override = True
        return obj


class Development(DevelopmentBase):
    pass
