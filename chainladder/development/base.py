# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.utils import WeightedRegression
from chainladder.core.io import EstimatorIO
from chainladder.core.common import Common


class DevelopmentBase(BaseEstimator, TransformerMixin, EstimatorIO, Common):
    def _set_fit_groups(self, X):
        """Used for assigning group_index in fit"""
        backend = "numpy" if X.array_backend in ["sparse", "numpy"] else "cupy"
        if self.groupby is None:
            return X.set_backend(backend)
        if callable(self.groupby) or type(self.groupby) in [list, str, pd.Series]:
            return X.groupby(self.groupby).sum().set_backend(backend)
        else:
            raise ValueError("Cannot determine groupings.")

    def _set_transform_groups(self, X):
        """Used for assigning group_index in transform"""
        if self.groupby is None:
            return X.index
        else:
            indices = X.groupby(self.groupby).groups.indices
        return (
            pd.Series(
                {vi: k for k, v in indices.items() for vi in v},
                name=self.ldf_.key_labels[0],
            )
            .sort_index()
            .to_frame()
        )

    def _assign_n_periods_weight(self, X, n_periods):
        """Used to apply the n_periods weight"""

        def _assign_n_periods_weight_int(X, n_periods):
            xp = X.get_array_module()
            val_offset = {
                "Y": {"Y": 1},
                "Q": {"Y": 4, "Q": 1},
                "M": {"Y": 12, "Q": 3, "M": 1},
            }
            if n_periods < 1 or n_periods >= X.shape[-2] - 1:
                return X.values * 0 + 1
            else:
                val_date_min = X.valuation[X.valuation <= X.valuation_date]
                val_date_min = val_date_min.drop_duplicates().sort_values()
                z = -n_periods * val_offset[X.development_grain][X.origin_grain] - 1
                val_date_min = val_date_min[z]
                w = X[X.valuation >= val_date_min]
                return xp.nan_to_num((w / w).values) * X.nan_triangle

        xp = X.get_array_module()

        dict_map = {
            item: _assign_n_periods_weight_int(X, item) for item in set(n_periods)
        }

        conc = [
            dict_map[item][..., num : num + 1] for num, item in enumerate(n_periods)
        ]
        return xp.concatenate(tuple(conc), -1)

    def _drop_adjustment(self, X, link_ratio):
        weight = X.nan_triangle[:, :-1]

        if self.drop is not None:
            weight = weight * self._drop(X)

        if self.drop_valuation is not None:
            weight = weight * self._drop_valuation(X)

        if (self.drop_high is not None) | (self.drop_low is not None):
            n_periods_ = self._validate_axis_assumption(
                self.n_periods, X.development[:-1]
            )

            w_ = self._assign_n_periods_weight(X, n_periods_)
            w_ = w_.astype("float")
            w_[w_ == 0] = np.nan

            link_ratio_w_ = link_ratio * w_

            weight = weight * self._drop_n(
                self.drop_high,
                self.drop_low,
                X,
                link_ratio_w_,
                self.preserve,
            )

        if (self.drop_above != np.inf) | (self.drop_below != 0.00):
            weight = weight * self._drop_x(
                self.drop_above, self.drop_below, X, link_ratio, self.preserve
            )

        return weight

    # for drop_high and drop_low
    def _drop_n(self, drop_high, drop_low, X, link_ratio, preserve):
        #this is safe because each triangle by index and column has 
        link_ratios_len = link_ratio.shape[3]

        def drop_array_helper(drop_type):
            drop_type_array = np.array(link_ratios_len * [0])

            if drop_type is None:
                return np.array(link_ratios_len * [0])
            else:
                # only a single parameter is provided
                if isinstance(drop_type, int):
                    drop_type_array = np.array(link_ratios_len * [drop_type])
                elif isinstance(drop_type, bool):
                    drop_type_array = np.array(link_ratios_len * int(drop_type))

                # an array of parameters is provided
                else:
                    for index in range(len(drop_type)):
                        drop_type_array[index] = int(drop_type[index])

                # convert boolean to ints (1s)
                for index in range(len(drop_type_array)):
                    if isinstance(drop_type_array[index], bool):
                        drop_type_array[index] = int(drop_type_array[index] == True)
                    else:
                        drop_type_array[index] = drop_type_array[index]

                return drop_type_array

        #explicitly setting up 3D arrays for drop parameters to avoid broadcasting bugs
        drop_high_array = np.zeros((link_ratio.shape[0],link_ratio.shape[1],link_ratios_len))
        drop_high_array[:,:,:] = drop_array_helper(drop_high)
        drop_low_array = np.zeros((link_ratio.shape[0],link_ratio.shape[1],link_ratios_len))
        drop_low_array[:,:,:] = drop_array_helper(drop_low)
        n_period_array = np.zeros((link_ratio.shape[0],link_ratio.shape[1],link_ratios_len))
        n_period_array[:,:,:] = drop_array_helper(self.n_periods)
        preserve_array = np.zeros((link_ratio.shape[0],link_ratio.shape[1],link_ratios_len))
        preserve_array[:,:,:] = drop_array_helper(preserve)        
        
        #operationalizing the -1 option for n_period
        n_period_array = np.where(n_period_array == -1, link_ratios_len, n_period_array)
        
        #ranking factors by itself and volume
        link_ratio_ranks = np.lexsort((X.values[...,:-1],link_ratio),axis = 2).argsort(axis=2)

        #setting up default return
        weights = ~np.isnan(link_ratio.transpose((0,1,3,2)))

        #counting valid factors
        ldf_count = weights.sum(axis=3)
        
        #applying n_period
        ldf_count_n_period = np.where(ldf_count > n_period_array, n_period_array, ldf_count)
        
        #applying drop_high and drop_low
        max_rank_unpreserve = ldf_count_n_period - drop_high_array
        min_rank_unpreserve = drop_low_array

        #applying preserve
        warning_flag = np.any(max_rank_unpreserve - min_rank_unpreserve < preserve)
        max_rank = np.where(max_rank_unpreserve - min_rank_unpreserve < preserve, ldf_count_n_period, max_rank_unpreserve)
        min_rank = np.where(max_rank_unpreserve - min_rank_unpreserve < preserve, 0, min_rank_unpreserve)
            
        index_array_weights = (link_ratio_ranks.transpose((0,1,3,2)) < max_rank.reshape(max_rank.shape[0],max_rank.shape[1],max_rank.shape[2],1)) & (
            link_ratio_ranks.transpose((0,1,3,2)) > min_rank.reshape(min_rank.shape[0],min_rank.shape[1],min_rank.shape[2],1) - 1
        )

        weights = index_array_weights
        
        if warning_flag:
            if preserve == 1:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(preserve)
                    + " (use preserve = ...)"
                    + " link ratio(s) is required for development estimation."
                )
            else:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(preserve)
                    + " link ratio(s) is required for development estimation."
                )
            warnings.warn(warning)

        return weights.transpose((0,1,3,2))

    # for drop_above and drop_below
    def _drop_x(self, drop_above, drop_below, X, link_ratio, preserve):
        #this is safe because each triangle by index and column has 
        link_ratios_len = link_ratio.shape[3]

        def drop_array_helper(drop_type, default_value):
            drop_type_array = np.array(link_ratios_len * [default_value])

            # only a single parameter is provided
            if isinstance(drop_type, int):
                drop_type_array = np.array(link_ratios_len * [float(drop_type)])
            elif isinstance(drop_type, float):
                drop_type_array = np.array(link_ratios_len * [drop_type])

            # an array of parameters is provided
            else:
                for index in range(len(drop_type)):
                    drop_type_array[index] = float(drop_type[index])

            return drop_type_array

        #explicitly setting up 3D arrays for drop parameters to avoid broadcasting bugs
        drop_above_array = np.zeros((link_ratio.shape[0],link_ratio.shape[1],link_ratios_len))
        drop_above_array[:,:,:] = drop_array_helper(drop_above, np.inf)
        drop_below_array = np.zeros((link_ratio.shape[0],link_ratio.shape[1],link_ratios_len))
        drop_below_array[:,:,:] = drop_array_helper(drop_below, 0.0)
        preserve_array = np.zeros((link_ratio.shape[0],link_ratio.shape[1],link_ratios_len))
        preserve_array[:,:,:] = drop_array_helper(preserve, preserve)        

        #transposing
        link_ratio_T = link_ratio.transpose((0,1,3,2))
        
        #setting up default return
        weights = ~np.isnan(link_ratio_T)
        
        #dropping
        index_array_weights = (link_ratio_T < drop_above_array[...,None]) & (
            link_ratio_T > drop_below_array[...,None]
        )

        #counting remaining factors
        ldf_count = index_array_weights.sum(axis=3)
        
        #applying preserve
        warning_flag = np.any(ldf_count < preserve_array)
        weights = np.where(ldf_count[...,None] < preserve_array[...,None], weights, index_array_weights)

        if warning_flag:
            if preserve == 1:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(preserve)
                    + " (use preserve = ...)"
                    + " link ratio(s) is required for development estimation."
                )
            else:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(preserve)
                    + " link ratio(s) is required for development estimation."
                )
            warnings.warn(warning)

        return weights.transpose((0,1,3,2))

    def _drop_valuation(self, X):
        xp = X.get_array_module()
        if type(self.drop_valuation) is not list:
            drop_valuation = [self.drop_valuation]
        else:
            drop_valuation = self.drop_valuation
        v = pd.PeriodIndex(drop_valuation, freq=X.origin_grain).to_timestamp(how="e")
        arr = 1 - xp.nan_to_num(X[X.valuation.isin(v)].values[0, 0] * 0 + 1)
        ofill = X.shape[-2] - arr.shape[-2]
        dfill = X.shape[-1] - arr.shape[-1]
        if ofill > 0:
            arr = xp.concatenate(
                (arr, xp.repeat(xp.ones(arr.shape[-1])[None], ofill, 0)), 0
            )
        if dfill > 0:
            arr = xp.concatenate(
                (arr, xp.repeat(xp.ones(arr.shape[-2])[..., None], dfill, -1)), -1
            )
        return arr[:, :-1]

    def _drop(self, X):
        xp = X.get_array_module()
        drop = [self.drop] if type(self.drop) is not list else self.drop
        arr = X.nan_triangle.copy()
        for item in drop:
            arr[
                np.where(X.origin == item[0])[0][0],
                np.where(X.development == item[1])[0][0],
            ] = 0
        return arr[:, :-1]
