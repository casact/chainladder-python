import numpy as np
import copy
from sklearn.base import BaseEstimator
from chainladder import WeightedRegression

"""
All transforms must return a Triangle object.  Because we need
to pass through more properties than just the data, we will extend the
triangle classto be three triangles
(k,v,o,d, m) - data, link_ratio, multi-origin cdf - the od_tri
(k, v, o, 1, m) - latest, ultimate, exposure - the o_tri
(k,v,1,d, m) - cdf, ldf, sigma, std_err - the d_tri

All fit properties that need to be passed on must be accessible from one of
these triangle and use @property so that we can access them by name rather than
worry about which triangle they come from.  @properties also allows us to
wrap them in functions so that we can use the same numpy array to store a
(k,v,o,d) and (k,v,o,d-1) object without the end user having to worry about
shape. That means these @properties need to be bound to the Triangle class.
This can be done through:
def foo(self):
    print('hi')
setattr(Class, 'foo', property(foo))
They can all share kdims and vdims, but will likely have different odims and ddims
Assignment of these properties will occur in the transform method
The data for the properties will be generated in the fit method, but at that
point, they will not be bound to the triangle. They will only be properties of
the sklearn model
"""


class DevelopmentBase(BaseEstimator):
    def __init__(self, n_per=-1, average='volume',
                 sigma_interpolation='log-linear'):
        self.n_per = n_per
        self.average = average
        self.sigma_interpolation = sigma_interpolation

    def _assign_n_per_weight(self, X):
        if type(self.n_per) is int:
            return self._assign_n_per_weight_int(X, self.n_per)[:, :, :, :-1]
        elif type(self.n_per) is list:
            if len(self.n_per) != X.triangle.shape[3]-1:
                raise ValueError(f'n_per list must be of lenth {X.triangle.shape[3]-1}.')
            else:
                return self._assign_n_per_weight_list(X)
        else:
            raise ValueError('n_per must be of type <int> or <list>')

    def _assign_n_per_weight_list(self, X):
        dict_map = {item: self._assign_n_per_weight_int(X, item)
                    for item in set(self.n_per)}
        conc = [dict_map[item][:, :, num:num+1, :]
                for num, item in enumerate(self.n_per)]
        return np.swapaxes(np.concatenate(tuple(conc), 2), 2, 3)

    def _assign_n_per_weight_int(self, X, n_per):
        ''' Zeros out weights depending on number of periods desired
            Only works for type(n_per) == int
        '''
        if n_per < 1 or n_per >= X.shape[2] - 1:
            return X.triangle*0+1
        else:
            flip_nan = np.nan_to_num(X.triangle*0+1)
            k, v, o, d = flip_nan.shape
            w = np.concatenate((1-flip_nan[:, :, -(o-n_per-1):, :],
                                np.ones((k, v, n_per+1, d))), 2)*flip_nan
            return w*X.expand_dims(X.nan_triangle())

    def fit(self, X, y=None, sample_weight=None):
        # Capture the fit inputs
        tri_array = X.triangle.copy()
        tri_array[tri_array == 0] = np.nan
        if type(self.average) is str:
            average = [self.average] * (tri_array.shape[3] - 1)
        else:
            average = self.average
        average = np.array(average)
        self.average_ = average
        weight_dict = {'regression': 0, 'volume': 1, 'simple': 2}
        _x = tri_array[:, :, :, :-1]
        _y = tri_array[:, :, :, 1:]
        val = np.array([weight_dict.get(item.lower(), 2)
                        for item in average])
        for i in [2, 1, 0]:
            val = np.repeat(np.expand_dims(val, 0), tri_array.shape[i], axis=0)
        val = np.nan_to_num(val * (_y * 0 + 1))
        _w = self._assign_n_per_weight(X) / (_x**(val))
        self.w_ = self._assign_n_per_weight(X)
        params = WeightedRegression(_w, _x, _y, axis=2, thru_orig=True) \
            .fit().sigma_fill(self.sigma_interpolation)
        params.std_err_ = np.nan_to_num(params.std_err_) + \
            np.nan_to_num((1-np.nan_to_num(params.std_err_*0+1)) *
            params.sigma_/np.swapaxes(np.sqrt(_x**(2-val))[:, :, 0:1, :], -1, -2))
        params = np.concatenate((params.slope_,
                                 params.sigma_,
                                 params.std_err_), 3)
        params = np.swapaxes(params, 2, 3)
        obj = copy.deepcopy(X)
        obj.triangle = params
        obj.odims = ['LDF', 'Sigma', 'Std Err']
        obj.ddims = np.array([f'{obj.ddims[i]}-{obj.ddims[i+1]}'
                              for i in range(len(obj.ddims)-1)])
        self._params = obj
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
        if np.all(self._params.vdims != X.vdims):
            raise KeyError('Values not aligned')
        if np.all(self._params.kdims != X.kdims):
            raise KeyError('Keys not aligned')
        X.triangle_d = self._params
        X.triangle_d.nan_overide = True
        setattr(X.__class__, '_param_property', _param_property)
        setattr(X.__class__, 'std_err_', property(std_err_))
        setattr(X.__class__, 'cdf_', property(cdf_))
        setattr(X.__class__, 'ldf_', property(ldf_))
        setattr(X.__class__, 'sigma_', property(sigma_))
        setattr(X.__class__, 'sigma_interpolation', self.sigma_interpolation)
        setattr(X.__class__, 'average_', self.average_)
        setattr(X.__class__, 'w_', self.w_)
        return X

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit_predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


class Development(DevelopmentBase):
    pass

    # ---------------------------------------------------------------- #
    # ---------------------- Transform Methods ----------------------- #
    # ---------------------------------------------------------------- #


def std_err_(self):
    return self._param_property(2)


def ldf_(self):
    return self._param_property(0)


def sigma_(self):
    return self._param_property(1)


def cdf_(self):
        obj = self.ldf_
        cdf_ = np.flip(np.cumprod(np.flip(obj.triangle, axis=3),
                                  axis=3), axis=3)
        obj.triangle = cdf_
        obj.odims = ['CDF']
        return obj


def _param_property(self, idx):
        obj = copy.deepcopy(self.triangle_d)
        temp = obj.triangle[:, :, idx:idx+1, :]
        obj.triangle = temp
        obj.odims = obj.odims[idx:idx+1]
        return obj
