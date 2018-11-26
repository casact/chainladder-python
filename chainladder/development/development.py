import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator
from chainladder import weighted_regression_through_origin, weighted_regression


class Development(BaseEstimator):
    def __init__(self, n_per=-1, avg_type='simple', sigma_interpolation='lin'):
        self.n_per = n_per
        self.avg_type = avg_type
        self.sigma_interpolation = sigma_interpolation

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
            obj = self.ldf_
            cdf_ = np.flip(np.cumprod(np.flip(obj.triangle, axis=3),
                                      axis=3), axis=3)
            obj.triangle = cdf_
            obj.odims = ['CDF']
            return obj

    def _param_property(self, idx):
            obj = copy.deepcopy(self._params)
            temp = obj.triangle[:, :, idx:idx+1, :]
            obj.triangle = temp
            obj.odims = obj.odims[idx:idx+1]
            return obj

    def fit(self, X, y=None, sample_weight=None):
        # Capture the fit inputs
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        tri_array = X.triangle.copy()
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
        params = weighted_regression_through_origin(x_, y_, w_, axis = 2)
        params = np.swapaxes(params, 2, 3)
        obj = copy.deepcopy(X)
        obj.triangle = params
        obj.odims = ['LDF', 'Sigma', 'Std Err']
        obj.ddims = np.array([f'{i+1}-{i+2}'
                              for i in range(len(obj.ddims)-1)])
        # We're replacing origin dimension with different statistics
        # This is a bastardization of the Triangle object so marking as hidden
        # and only accessible through @property
        self._params = obj
        self.sigma_fill()
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
        if np.all(self.X_.vdims != X.vdims):
            raise KeyError('Values dimension not aligned')
        if np.all(self.X_.ddims != X.ddims):
            raise KeyError('Development dimension not aligned')
        if not np.all([item in X.key_labels for item in self.X_.key_labels]):
            raise KeyError('Key dimension not aligned')
        obj = copy.deepcopy(self)
        if np.all(X.key_labels != self.X_.key_labels):
            # Need to apply the development object to the expanded set of keys
            # that exist on X.
            new_param = self._params.triangle.copy()
            temp1 = X.keys.reset_index().rename(columns={'index': 'old_index'})
            temp2 = self.X_.keys.reset_index().rename(columns={'index': 'new_index'})
            old_k_by_new_k = temp1.merge(temp2, how='left', on=self.X_.key_labels)
            old_k_by_new_k = \
                np.array(pd.pivot_table(old_k_by_new_k,
                                        index='old_index',
                                        columns='new_index',
                                        values=self.X_.key_labels[0],
                                        aggfunc='count').fillna(0))
            new_param = np.expand_dims(new_param, 0)
            for i in range(3):
                old_k_by_new_k = np.expand_dims(old_k_by_new_k, -1)
            new_param = new_param*old_k_by_new_k
            new_param = np.max(np.nan_to_num(new_param), axis=1)
            obj._params.triangle = new_param
            obj._params.kdims = X.kdims
            obj._params.key_labels = X.key_labels
        obj.X_ = X
        return obj

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit_predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def sigma_fill(self):
        y = self._params.triangle[:, :, 1:2, :]
        w = np.nan_to_num(y*0+1)
        y = np.log(y)
        slope, intercept = weighted_regression(y=y, w=w, axis=3)
        x = np.reshape(np.arange(y.shape[3]), (1, 1, 1, y.shape[3]))
        temp = np.array(x.shape)
        temp = np.where(temp == temp.max())[0][0]
        x = np.swapaxes(x, temp, 3) *(1-w)
        sigma_fill_ = np.exp(x*slope+intercept)*(1-w)
        self._params.triangle[:,:,1:2,:] = np.nan_to_num(self._params.triangle[:,:,1:2,:]) + \
            sigma_fill_
        return self
