# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import copy
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder import WeightedRegression
from chainladder.core import EstimatorIO


class DevelopmentBase(BaseEstimator, TransformerMixin, EstimatorIO):
    @staticmethod
    def _get_cdf(obj):
        if 'ldf_' not in obj:
            return
        else:
            obj2 = copy.copy(obj.ldf_)
            cdf_ = np.flip(np.cumprod(np.flip(obj2.values, -1), -1), -1)
            obj2.ddims = [item.replace(item[item.find("-")+1:], '9999')
                          for item in obj2.ddims]
            obj2.values = cdf_
            obj2._set_slicers()
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
    drop : tuple or list of tuples
        Drops specific origin/development combination(s)
    drop_high : bool or list of bool (default=None)
        Drops highest link ratio(s) from LDF calculation
    drop_low : bool or list of bool (default=None)
        Drops lowest link ratio(s) from LDF calculation
    drop_valuation : str or list of str (default=None)
        Drops specific valuation periods. str must be date convertible.


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
                 sigma_interpolation='log-linear', drop=None,
                 drop_high=None, drop_low=None, drop_valuation=None):
        self.n_periods = n_periods
        self.average = average
        self.sigma_interpolation = sigma_interpolation
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_valuation = drop_valuation
        self.drop = drop

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
        ''' private method to standardize the n_periods input to a list '''
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
            val_offset = {
                'Y': {'Y': 1},
                'Q': {'Y':4, 'Q': 1},
                'M': {'Y':12, 'Q': 3, 'M': 1}}
            val_date_min = \
                X.valuation[X.valuation<=X.valuation_date].drop_duplicates().sort_values()
            val_date_min = \
                val_date_min[-n_periods * \
                val_offset[X.development_grain][X.origin_grain] - 1]
            w = X[X.valuation>=val_date_min]
            return np.nan_to_num((w/w).values)*X._expand_dims(X._nan_triangle())


    def _drop_adjustment(self, X, link_ratio):
        weight = X._nan_triangle()[:, :-1]
        if self.drop_high == self.drop_low == \
           self.drop == self.drop_valuation is None:
            return weight
        if self.drop_high is not None:
            weight = weight*self._drop_hilo('high', X, link_ratio)
        if self.drop_low is not None:
            weight = weight*self._drop_hilo('low', X, link_ratio)
        if self.drop is not None:
            weight = weight*self._drop(X)
        if self.drop_valuation is not None:
            weight = weight*self._drop_valuation(X)
        return weight

    def _drop_hilo(self, kind, X, link_ratio):
        link_ratio[link_ratio == 0] = np.nan
        lr_valid_count = np.sum(~np.isnan(link_ratio)[0, 0], axis=0)
        if kind == 'high':
            vals = np.nanmax(link_ratio, -2, keepdims=True)
            drop_hilo = self.drop_high
        else:
            vals = np.nanmin(link_ratio, -2, keepdims=True)
            drop_hilo = self.drop_low
        hilo = 1*(vals != link_ratio)
        if type(drop_hilo) is bool:
            drop_hilo = [drop_hilo]*(len(X.development)-1)
        for num, item in enumerate(self.average_):
            if not drop_hilo[num]:
                hilo[..., num] = hilo[..., num]*0+1
            else:
                if lr_valid_count[num] < 3:
                    hilo[..., num] = hilo[..., num]*0+1
                    warnings.warn('drop_high and drop_low cannot be computed '
                                  'when less than three LDFs are present. '
                                  'Ignoring exclusions in some cases.')
        return hilo

    def _drop_valuation(self, X):
        if type(self.drop_valuation) is not list:
            drop_valuation = [self.drop_valuation]
        else:
            drop_valuation = self.drop_valuation
        arr = 1-np.nan_to_num(X[X.valuation.isin(
            pd.PeriodIndex(drop_valuation,
                           freq=X.origin_grain))].values[0, 0]*0+1)
        ofill = X.shape[-2]-arr.shape[-2]
        dfill = X.shape[-1]-arr.shape[-1]
        if ofill > 0:
            arr = np.concatenate((arr, np.repeat(
                np.ones(arr.shape[-1])[np.newaxis], ofill, 0)), 0)
        if dfill > 0:
            arr = np.concatenate((arr, np.repeat(
                np.ones(arr.shape[-2])[..., np.newaxis], dfill, -1)), -1)
        return arr[:, :-1]

    def _drop(self, X):
        drop = [self.drop] if type(self.drop) is not list else self.drop
        arr = X._nan_triangle()
        for item in drop:
            arr[np.where(X.origin == item[0])[0][0],
                np.where(X.development == item[1])[0][0]] = 0
        return arr[:, :-1]

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
        if (type(X.ddims) != np.ndarray):
            raise ValueError('Triangle must be expressed with development lags')
        tri_array = X.values.copy()
        tri_array[tri_array == 0] = np.nan
        if type(self.average) is str:
            average = [self.average] * (tri_array.shape[-1] - 1)
        else:
            average = self.average
        average = np.array(average)
        self.average_ = average
        weight_dict = {'regression': 0, 'volume': 1, 'simple': 2}
        x, y = tri_array[..., :-1], tri_array[..., 1:]
        val = np.array([weight_dict.get(item.lower(), 1)
                        for item in average])
        for i in [2, 1, 0]:
            val = np.repeat(val[np.newaxis], tri_array.shape[i], axis=0)
        val = np.nan_to_num(val * (y * 0 + 1))
        link_ratio = np.divide(y, x, where=np.nan_to_num(x) != 0)
        self.w_ = np.array(self._assign_n_periods_weight(X) *
                           self._drop_adjustment(X, link_ratio),
                           dtype='float16')
        w = self.w_ / (x**(val))
        params = WeightedRegression(axis=2, thru_orig=True).fit(x, y, w)
        if self.n_periods != 1:
            params = params.sigma_fill(self.sigma_interpolation)
        else:
            warnings.warn('Setting n_periods=1 does not allow enough degrees '
                          'of freedom to support calculation of all regression'
                          ' statistics.  Only LDFs have been calculated.')
        params.std_err_ = np.nan_to_num(params.std_err_) + \
            np.nan_to_num(
                (1-np.nan_to_num(params.std_err_*0+1)) *
                params.sigma_ /
                np.swapaxes(np.sqrt(x**(2-val))[..., 0:1, :], -1, -2))
        params = np.concatenate(
            (params.slope_, params.sigma_, params.std_err_), 3)
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
        X_new = copy.copy(X)
        triangles = ['std_err_', 'cdf_', 'ldf_', 'sigma_']
        for item in triangles + ['average_', 'w_', 'sigma_interpolation']:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new

    def _param_property(self, X, params, idx):
        obj = copy.copy(X)
        obj.values = np.ones(X.shape)[..., :-1]*params[..., idx:idx+1, :]
        obj.ddims = X.link_ratio.ddims
        obj.valuation = obj._valuation_triangle(obj.ddims)
        obj.nan_override = True
        obj._set_slicers()
        return obj
