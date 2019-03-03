"""
Loss Development
================
"""
import numpy as np
import copy
import warnings
from sklearn.base import BaseEstimator
from chainladder import WeightedRegression


class DevelopmentBase(BaseEstimator):
    def fit_transform(self, X, y=None, sample_weight=None):
        """ Equivalent to fit(X).transform(X)

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs based on the model.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        self.fit(X, y, sample_weight)
        return self.transform(X)

    @staticmethod
    def _get_cdf(obj):
        if obj.__dict__.get('ldf_', None) is None:
            return
        else:
            obj2 = copy.deepcopy(obj.ldf_)
            cdf_ = np.flip(np.cumprod(np.flip(obj2.values, -1), -1), -1)
            obj2.values = cdf_
            return obj2



class Development(DevelopmentBase):
    """ A Transformer that allows for basic loss development pattern selection.

    Parameters
    ----------
    n_periods : integer, optional (default=-1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods=-1
    average : string, optional (default='volume')
        type of averaging to use for ldf average calculation.  Options include
        'volume', 'simple', and 'regression'
    sigma_interpolation : string optional (default='log-linear')
        Options include 'log-linear' and 'mack'

    Attributes
    ----------
    ldf_ : Triangle
        The estimated loss development patterns
    cdf_ : Triangle
        The estimated cumulative development patterns
    sigma_ : Triangle
        Sigma of the ldf regression
    std_err_ : Triangle
        Std_err of the ldf regression
    w_ : Triangle
        The weight used in the ldf regression

    """
    def __init__(self, n_periods=-1, average='volume',
                 sigma_interpolation='log-linear'):
        self.n_periods = n_periods
        self.average = average
        self.sigma_interpolation = sigma_interpolation

    def _assign_n_periods_weight(self, X):
        if type(self.n_periods) is int:
            return self._assign_n_periods_weight_int(X, self.n_periods)[..., :-1]
        elif type(self.n_periods) is list:
            if len(self.n_periods) != X.values.shape[-1]-1:
                raise ValueError('n_periods list must be of lenth {}.'
                                 .format(X.values.shape[-1]-1))
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
            return X.values*0+1
        else:
            flip_nan = np.nan_to_num(X.values*0+1)
            k, v, o, d = flip_nan.shape
            w = np.concatenate((1-flip_nan[..., -(o-n_periods-1):, :],
                                np.ones((k, v, n_periods+1, d))), 2)*flip_nan
            return w*X.expand_dims(X.nan_triangle())

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the munich adjustment will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        tri_array = X.values.copy()
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
        params = WeightedRegression(axis=2, thru_orig=True).fit(_x, _y, _w)
        if self.n_periods != 1:
            params = params.sigma_fill(self.sigma_interpolation)
        else:
            warnings.warn('Setting n_periods=1 does not allow enough degrees '
                          'of freedom to support calculation of all regression'
                          ' statistics.  Only LDFs have been calculated.')
        params.std_err_ = np.nan_to_num(params.std_err_) + \
            np.nan_to_num((1-np.nan_to_num(params.std_err_*0+1)) *
            params.sigma_/np.swapaxes(np.sqrt(_x**(2-val))[..., 0:1, :], -1, -2))
        params = np.concatenate((params.slope_,
                                 params.sigma_,
                                 params.std_err_), 3)
        params = np.swapaxes(params, 2, 3)
        self.ldf_ = self._param_property(X, params, 0)
        self.cdf_ = self._get_cdf(self)
        self.sigma_ = self._param_property(X, params, 1)
        self.std_err_ = self._param_property(X, params, 2)
        return self

    def transform(self, X):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = copy.deepcopy(X)
        for item in ['std_err_', 'cdf_', 'ldf_', 'average_',
                     'sigma_', 'w_', 'sigma_interpolation']:
            X_new.__dict__[item] = self.__dict__[item]
        return X_new

    def _param_property(self, X, params, idx):
        obj = copy.deepcopy(X)
        obj.values = np.ones(X.shape)[..., :-1]*params[..., idx:idx+1, :]
        obj.ddims = X.link_ratio.ddims
        obj.valuation = obj._valuation_triangle(obj.ddims)
        obj.nan_override = True
        return obj
