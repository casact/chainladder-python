# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import copy
import numpy as np
from chainladder.utils import WeightedRegression
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.development import DevelopmentBase, Development
from chainladder.core import EstimatorIO
from chainladder.core.common import Common


class TailBase(BaseEstimator, TransformerMixin, EstimatorIO, Common):
    ''' Base class for all tail methods.  Tail objects are equivalent
        to development objects with an additional set of tail statistics'''
    def fit(self, X, y=None, sample_weight=None):
        if X.array_backend == 'sparse':
            obj = X.set_backend('numpy')
            xp = np
        else:
            xp = X.get_array_module()
            obj = copy.copy(X)
        if 'ldf_'not in obj:
            obj = Development().fit_transform(obj)
        self._ave_period = {'Y': (1, 12),
                            'Q': (4, 3),
                            'M': (12, 1)}[obj.development_grain]
        ddims = np.concatenate(
            (obj.ddims, [(item+1)*self._ave_period[1] + obj.ddims[-1]
                         for item in range(self._ave_period[0])], [9999]), 0)
        self.ldf_ = copy.copy(obj.ldf_)
        tail = xp.ones(self.ldf_.shape)[..., -1:]
        tail = xp.repeat(tail, self._ave_period[0]+1, -1)
        self.ldf_.values = xp.concatenate((self.ldf_.values, tail), -1)
        self.ldf_.ddims = np.array(['{}-{}'.format(ddims[i], ddims[i+1])
                                    for i in range(len(ddims)-1)])
        if hasattr(obj, 'sigma_'):
            self.sigma_ = copy.copy(getattr(obj, 'sigma_'))
        else:
            self.sigma_ = obj.ldf_ - obj.ldf_
        if hasattr(obj, 'std_err_'):
            self.std_err_ = copy.copy(getattr(obj, 'std_err_'))
        else:
            self.std_err_ = obj.ldf_ - obj.ldf_
        zeros = tail[..., 0:1, -1:]*0
        self.sigma_.values = xp.concatenate(
            (self.sigma_.values, zeros), -1)
        self.std_err_.values = xp.concatenate(
            (self.std_err_.values, zeros), -1)
        self.sigma_.ddims = self.std_err_.ddims = \
            np.concatenate(
                (obj.ldf_.ddims,
                 np.array(['{}-9999'.format(int(obj.ddims[-1]))])))
        self.ldf_._set_slicers()
        self.sigma_._set_slicers()
        self.std_err_._set_slicers()
        return self

    def transform(self, X):
        if 'ldf_'not in X:
            X_new = Development().fit_transform(X)
        else:
            X_new = copy.deepcopy(X)
        xp = X_new.ldf_.get_array_module() if X_new.ldf_.array_backend != 'sparse' else np
        X_new.std_err_.values = xp.concatenate(
            (X_new.std_err_.values,
             self.std_err_.values[..., -1:]), -1)
        extend = self.cdf_.values[..., -self._ave_period[0]-1:]*0+1
        X_new.cdf_.values = xp.concatenate((X_new.cdf_.values, extend), -1)
        X_new.cdf_.values = X_new.cdf_.values * \
            self.cdf_.values[..., -self._ave_period[0]-1:-self._ave_period[0]]
        X_new.cdf_.values = xp.concatenate(
            (X_new.cdf_.values[..., :-1], self.cdf_.values[..., -1:]), axis=-1)
        X_new.ldf_.values = xp.concatenate(
            (X_new.ldf_.values,
             self.ldf_.values[..., -self._ave_period[0]-1:]), -1)
        X_new.sigma_.values = xp.concatenate(
            (X_new.sigma_.values, self.sigma_.values[..., -1:]), -1)
        X_new.cdf_.ddims = X_new.ldf_.ddims = self.ldf_.ddims
        X_new.sigma_.ddims = X_new.std_err_.ddims = self.sigma_.ddims
        X_new.sigma_._set_slicers()
        X_new.ldf_._set_slicers()
        X_new.cdf_._set_slicers()
        X_new.std_err_._set_slicers()
        X_new._ave_period = self._ave_period
        X_new.tail_ = TailBase._tail_(X_new)
        return X_new

    def _get_tail_prediction(self, tail_ldf):
        xp = self.ldf_.get_array_module()
        accum_point = self.ldf_.shape[-1] - 1
        ave = 1 + tail_ldf[..., :accum_point]
        all = xp.prod(1 + tail_ldf[..., accum_point:], -1)[..., None]
        tail = xp.concatenate((ave, all), -1)
        return tail

    def _get_initial_ldf(self, xp, tail):
        ''' Quadratic series expansion solution to return seed LDF for tail'''
        arr = self.decay**xp.arange(1000)
        a = xp.sum(arr**2)
        b = xp.sum(arr)
        c = -xp.log(tail)
        return (-b+xp.sqrt(b**2-4*a*c))/(2*a)

    def _apply_decay(self, X, tail, attach_idx=None):
        ''' Created Tail vector with decay over time. '''
        xp = self.ldf_.get_array_module()
        if attach_idx:
            decay_range = self.ldf_.shape[-1] - attach_idx
        else:
            decay_range = self.ldf_.shape[-1]-X.shape[-1]+1
        if xp.max(xp.array(tail)) == 1.0:
            ldfs = 1 + 0*(self.decay**xp.arange(1000))
        else:
            ldfs = 1+self._get_initial_ldf(xp, tail)*(self.decay**xp.arange(1000))
        ldfs = ldfs[..., :decay_range]
        tail = tail/xp.prod(ldfs[..., :-1], axis=-1, keepdims=True)
        ldfs = xp.concatenate((ldfs[..., :-1], tail), axis=-1)
        self.ldf_.values = xp.concatenate(
            (self.ldf_.values[..., :-decay_range],
            (self.ldf_.values[..., -decay_range:]*0+1)*ldfs), axis=-1)
        return self

    def _get_tail_stats(self, X):
        """ Method to approximate the tail sigma using
        log-linear extrapolation applied to tail average period
        """
        from chainladder.utils.utility_functions import num_to_nan
        time_pd = self._get_tail_weighted_time_period(X)
        xp = X.sigma_.get_array_module()
        reg = WeightedRegression(axis=3, xp=xp).fit(None, xp.log(X.sigma_.values), None)
        sigma_ = xp.exp(time_pd*reg.slope_+reg.intercept_)
        y = X.std_err_.values
        y = num_to_nan(y)
        reg = WeightedRegression(axis=3, xp=xp).fit(None, xp.log(y), None)
        std_err_ = xp.exp(time_pd*reg.slope_+reg.intercept_)
        return sigma_, std_err_

    def _get_tail_weighted_time_period(self, X):
        """ Method to approximate the weighted-average development age of tail
        using log-linear extrapolation

        Returns: float32
        """
        y = X.ldf_.values.copy()
        xp = X.ldf_.get_array_module()
        y[y <= 1] = xp.nan
        reg = WeightedRegression(axis=3, xp=xp).fit(None, xp.log(y - 1), None)
        tail = xp.prod(self.ldf_.values[..., -self._ave_period[0]-1:],
                       -1, keepdims=True)
        reg = WeightedRegression(axis=3, xp=xp).fit(None, xp.log(y - 1), None)
        time_pd = (xp.log(tail-1)-reg.intercept_)/reg.slope_
        return time_pd

    @staticmethod
    def _tail_(self):
        df = self.cdf_[self.cdf_.development==
                       self.cdf_.development.iloc[-1-self._ave_period[0]]]
        if np.all(df.values.min(axis=2) == df.values.max(axis=2)):
            idx = self.cdf_._idx_table()
            df = df[df.origin==df.origin.min()].to_frame()
        return df

    @property
    def tail_(self):
        return TailBase._tail_(self)
