# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.utils import WeightedRegression
from chainladder.utils.utility_functions import num_to_nan
from chainladder.core.io import EstimatorIO
from chainladder.core.common import Common


class DevelopmentBase(BaseEstimator, TransformerMixin, EstimatorIO, Common):
    def fit(self, X, y=None, sample_weight=None):
        average_ = self._validate_assumption(y, self.average, axis=3)
        self.average_ = average_.flatten()
        exponent = self.xp.array(
            [{"regression": 0, "volume": 1, "simple": 2}[x] for x in average_[0, 0, 0]]
        )
        exponent = self.xp.nan_to_num(exponent * (y * 0 + 1))
        w = num_to_nan(sample_weight / (X ** (exponent)))
        self.params_ = WeightedRegression(axis=2, thru_orig=True, xp=self.xp).fit(
            X, y, w
        )
        return self

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
                "S": {"Y": 2, "S": 1},
                "Q": {"Y": 4, "S": 2, "Q": 1},
                "M": {"Y": 12, "S": 6, "Q": 3, "M": 1},
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
            item: _assign_n_periods_weight_int(X, item)
            for item in set(n_periods.flatten())
        }

        conc = [
            dict_map[item][..., num : num + 1]
            for num, item in enumerate(n_periods.flatten())
        ]
        return xp.concatenate(tuple(conc), -1)

    def _drop_adjustment(self, X, link_ratio):
        weight = X.nan_triangle[:, :-1]
        if self.drop is not None:
            weight = weight * self._drop(X)

        if self.drop_valuation is not None:
            weight = weight * self._drop_valuation(X)

        if (self.drop_high is not None) | (self.drop_low is not None):
            n_periods_ = self._validate_assumption(X, self.n_periods, axis=3)[
                0, 0, 0, :-1
            ]

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
        # this is safe because each triangle by index and column has
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

        # explicitly setting up 3D arrays for drop parameters to avoid broadcasting bugs
        drop_high_array = np.zeros(
            (link_ratio.shape[0], link_ratio.shape[1], link_ratios_len)
        )
        drop_high_array[:, :, :] = drop_array_helper(drop_high)
        drop_low_array = np.zeros(
            (link_ratio.shape[0], link_ratio.shape[1], link_ratios_len)
        )
        drop_low_array[:, :, :] = drop_array_helper(drop_low)
        n_period_array = np.zeros(
            (link_ratio.shape[0], link_ratio.shape[1], link_ratios_len)
        )
        n_period_array[:, :, :] = drop_array_helper(self.n_periods)
        preserve_array = np.zeros(
            (link_ratio.shape[0], link_ratio.shape[1], link_ratios_len)
        )
        preserve_array[:, :, :] = drop_array_helper(preserve)

        # operationalizing the -1 option for n_period
        n_period_array = np.where(n_period_array == -1, link_ratios_len, n_period_array)

        # ranking factors by itself and volume
        link_ratio_ranks = np.lexsort((X.values[..., :-1], link_ratio), axis=2).argsort(
            axis=2
        )

        # setting up default return
        weights = ~np.isnan(link_ratio.transpose((0, 1, 3, 2)))

        # counting valid factors
        ldf_count = weights.sum(axis=3)

        # applying n_period
        ldf_count_n_period = np.where(
            ldf_count > n_period_array, n_period_array, ldf_count
        )

        # applying drop_high and drop_low
        max_rank_unpreserve = ldf_count_n_period - drop_high_array
        min_rank_unpreserve = drop_low_array

        # applying preserve
        warning_flag = np.any(max_rank_unpreserve - min_rank_unpreserve < preserve)
        max_rank = np.where(
            max_rank_unpreserve - min_rank_unpreserve < preserve,
            ldf_count_n_period,
            max_rank_unpreserve,
        )
        min_rank = np.where(
            max_rank_unpreserve - min_rank_unpreserve < preserve, 0, min_rank_unpreserve
        )

        index_array_weights = (
            link_ratio_ranks.transpose((0, 1, 3, 2))
            < max_rank.reshape(
                max_rank.shape[0], max_rank.shape[1], max_rank.shape[2], 1
            )
        ) & (
            link_ratio_ranks.transpose((0, 1, 3, 2))
            > min_rank.reshape(
                min_rank.shape[0], min_rank.shape[1], min_rank.shape[2], 1
            )
            - 1
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

        return weights.transpose((0, 1, 3, 2))

    # for drop_above and drop_below
    def _drop_x(self, drop_above, drop_below, X, link_ratio, preserve):

        # this is safe because each triangle by index and column has
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

        # explicitly setting up 3D arrays for drop parameters to avoid broadcasting bugs
        drop_above_array = np.zeros(
            (link_ratio.shape[0], link_ratio.shape[1], link_ratios_len)
        )
        drop_above_array[:, :, :] = drop_array_helper(drop_above, np.inf)
        drop_below_array = np.zeros(
            (link_ratio.shape[0], link_ratio.shape[1], link_ratios_len)
        )
        drop_below_array[:, :, :] = drop_array_helper(drop_below, 0.0)
        preserve_array = np.zeros(
            (link_ratio.shape[0], link_ratio.shape[1], link_ratios_len)
        )
        preserve_array[:, :, :] = drop_array_helper(preserve, preserve)

        # transposing
        link_ratio_T = link_ratio.transpose((0, 1, 3, 2))

        # setting up default return
        weights = ~np.isnan(link_ratio_T)

        # dropping
        index_array_weights = (link_ratio_T < drop_above_array[..., None]) & (
            link_ratio_T > drop_below_array[..., None]
        )

        # counting remaining factors
        ldf_count = index_array_weights.sum(axis=3)

        # applying preserve
        warning_flag = np.any(ldf_count < preserve_array)
        weights = np.where(
            ldf_count[..., None] < preserve_array[..., None],
            weights,
            index_array_weights,
        )

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

        return weights.transpose((0, 1, 3, 2))

    def _drop_valuation(self, X):
        xp = X.get_array_module()

        if type(self.drop_valuation) is not list:
            drop_valuation = [self.drop_valuation]
        else:
            drop_valuation = self.drop_valuation

        drop_valuation_vector = pd.PeriodIndex(
            drop_valuation, freq=X.development_grain
        ).to_timestamp(how="e")

        tri_w = X * 0 + 1
        tri_w = tri_w[~tri_w.valuation.isin(drop_valuation_vector)]
        tri_w = xp.nan_to_num(tri_w.values[0, 0])

        return tri_w[:, :-1]

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

    def _param_array_helper(self, size, param, default_value):
        # setting default
        param_array = pd.Series(size * [default_value]).astype("object")
        # only a single parameter is provided
        if isinstance(param, list):
            param_array[range(len(param))] = np.array(param)
            param_array.loc[param_array.isna()] = default_value
            param_array = param_array.astype(type(default_value))
        # an array of parameters is provided
        else:
            param_array.loc[:] = param
            param_array.loc[param_array.isna()] = default_value
            param_array = param_array.astype(type(default_value))
        return param_array.to_numpy()

    def _set_weight_func(self, factor, secondary_rank=None):
        w = (~np.isnan(factor.values)).astype(float)
        w = w * self._assign_n_periods_weight_func(factor)
        if self.drop is not None:
            w = w * self._drop_func(factor)

        if self.drop_valuation is not None:
            w = w * self._drop_valuation_func(factor)

        if (self.drop_above != np.inf) | (self.drop_below != 0.0):
            w = w * self._drop_x_func(factor)

        if (self.drop_high is not None) | (self.drop_low is not None):
            w = w * self._drop_n_func(factor * num_to_nan(w), secondary_rank)

        w_tri = factor.copy()
        w_tri.values = num_to_nan(w)

        return w_tri

    def _assign_n_periods_weight_func(self, factor):
        """Used to apply the n_periods weight"""
        # getting dimensions of factor for various manipulation
        factor_len = factor.shape[3]
        # putting n_periods into array
        n_periods_array = self._param_array_helper(factor_len, self.n_periods, -1)

        def _assign_n_periods_weight_int(X, n_periods):
            xp = X.get_array_module()
            val_offset = {
                "Y": {"Y": 1},
                "S": {"Y": 2, "S": 1},
                "Q": {"Y": 4, "S": 2, "Q": 1},
                "M": {"Y": 12, "S": 6, "Q": 3, "M": 1},
            }
            if n_periods < 1 or n_periods >= X.shape[-2]:
                return X.values * 0 + 1
            else:
                val_date_min = X.valuation[X.valuation <= X.valuation_date]
                val_date_min = val_date_min.drop_duplicates().sort_values()
                z = -n_periods * val_offset[X.development_grain][X.origin_grain]
                val_date_min = val_date_min[z]
                w = X[X.valuation >= val_date_min]
                return xp.nan_to_num((w / w).values) * X.nan_triangle

        xp = factor.get_array_module()

        dict_map = {
            item: _assign_n_periods_weight_int(factor, item)
            for item in set(n_periods_array)
        }
        conc = [
            dict_map[item][..., num : num + 1]
            for num, item in enumerate(n_periods_array)
        ]
        return xp.concatenate(tuple(conc), -1)

    def _drop_func(self, factor):
        # get the appropriate backend for nan_triangle and nan_to_num
        xp = factor.get_array_module()
        # turn single drop_valuation parameter to list if necessary
        drop_list = self.drop if isinstance(self.drop, list) else [self.drop]
        # get an starting array of weights
        arr = factor.nan_triangle.copy()
        # accommodate ldf triangle as factor, where the dimensions are '12-24'
        dev_list = (
            factor.development.str.split("-", expand=True)[0]
            if factor.development.dtype == object
            else factor.development.astype("string")
        )
        # create ndarray of drop_list for further operation in numpy
        drop_np = np.asarray(drop_list)
        # find indices of drop_np
        origin_ind = np.where(
            np.array([factor.origin.astype("string")]) == drop_np[:, [0]]
        )[1]
        dev_ind = np.where(np.array([dev_list]) == drop_np[:, [1]])[1]
        # set weight of dropped factors to 0
        arr[(origin_ind, dev_ind)] = 0
        return xp.nan_to_num(arr)[None, None]

    def _drop_valuation_func(self, factor):

        # get the appropriate backend for nan_to_num
        xp = factor.get_array_module()
        # turn single drop_valuation parameter to list if necessary
        if isinstance(self.drop_valuation, list):
            drop_valuation_list = self.drop_valuation
        else:
            drop_valuation_list = [self.drop_valuation]
        # turn drop_valuation to same valuation freq as factor
        v = pd.PeriodIndex(
            drop_valuation_list, freq=factor.development_grain
        ).to_timestamp(how="e")
        # warn that some drop_valuation are outside of factor
        if np.any(~v.isin(factor.valuation)):
            warnings.warn("Some valuations could not be dropped.")
        # return triangle of 0/1 where dropped factors have 0
        b = xp.nan_to_num(factor.iloc[0, 0][~factor.valuation.isin(v)].values * 0 + 1)
        # check to make sure some factors are still left
        if b.sum() == 0:
            raise Exception("The entire triangle has been dropped via drop_valuation.")
        return b

    def _drop_x_func(self, factor):
        # getting dimensions of factor for various manipulation
        factor_val = factor.values.copy()
        factor_len = factor_val.shape[3]
        indices = factor_val.shape[0]
        columns = factor_val.shape[1]

        # explicitly setting up 3D arrays for drop parameters to avoid broadcasting bugs
        drop_above_array = np.zeros((indices, columns, factor_len))
        drop_above_array[:, :, :] = self._param_array_helper(
            factor_len, self.drop_above, np.inf
        )[None, None]
        drop_below_array = np.zeros((indices, columns, factor_len))
        drop_below_array[:, :, :] = self._param_array_helper(
            factor_len, self.drop_below, 0.0
        )[None, None]
        preserve_array = np.zeros((indices, columns, factor_len))
        preserve_array[:, :, :] = self._param_array_helper(
            factor_len, self.preserve, self.preserve
        )[None, None]
        # transposing so columns of factors (same dev age) are in the last index.
        # not sure if this is really necessary. will leave for a better dev to find out
        factor_val_T = factor_val.transpose((0, 1, 3, 2))

        # setting up starting array of weights
        w = ~np.isnan(factor_val_T)

        # dropping
        index_array_weights = (factor_val_T < drop_above_array[..., None]) & (
            factor_val_T > drop_below_array[..., None]
        )

        # counting remaining factors
        ldf_count = index_array_weights.sum(axis=3)

        # applying preserve
        warning_flag = np.any(ldf_count < preserve_array)
        w = np.where(
            ldf_count[..., None] < preserve_array[..., None], w, index_array_weights
        )

        if warning_flag:
            if self.preserve == 1:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(self.preserve)
                    + " (use preserve = ...)"
                    + " link ratio(s) is required for development estimation."
                )
            else:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(self.preserve)
                    + " link ratio(s) is required for development estimation."
                )
            warnings.warn(warning)

        return w.transpose((0, 1, 3, 2)).astype(float)

    # for drop_high and drop_low
    def _drop_n_func(self, factor, secondary_rank=None):
        # getting dimensions of factor for various manipulation
        factor_val = factor.values.copy()
        # secondary rank is the optional triangle that breaks ties in factor
        # the original use case is for dropping the link ratio of 1 with the lowest loss value
        # (pass in a reverse rank of loss to drop link of ratio of 1 with the highest loss value)
        # leaving to user to ensure that secondary rank is the same dimensions as factor
        # also leaving to user to pick whether to trim head or tail
        if secondary_rank is None:
            sec_rank_val = factor_val.copy()
        else:
            sec_rank_val = secondary_rank.values.copy()
        factor_len = factor_val.shape[3]
        indices = factor_val.shape[0]
        columns = factor_val.shape[1]

        # explicitly setting up 3D arrays for drop parameters to avoid broadcasting bugs
        drop_high_array = np.zeros((indices, columns, factor_len))
        drop_high_array[:, :, :] = self._param_array_helper(
            factor_len, self.drop_high, 0
        )[None, None]
        drop_low_array = np.zeros((indices, columns, factor_len))
        drop_low_array[:, :, :] = self._param_array_helper(
            factor_len, self.drop_low, 0
        )[None, None]
        preserve_array = np.zeros((indices, columns, factor_len))
        preserve_array[:, :, :] = self._param_array_helper(
            factor_len, self.preserve, self.preserve
        )[None, None]

        # ranking factors by itself and secondary rank
        factor_ranks = np.lexsort((sec_rank_val, factor_val), axis=2).argsort(axis=2)

        # setting up starting weights
        w = ~np.isnan(factor_val.transpose((0, 1, 3, 2)))

        # counting valid factors
        ldf_count = w.sum(axis=3)

        # getting max index after drop high
        max_rank_unpreserve = ldf_count - drop_high_array

        # applying preserve
        preserve_trigger = (max_rank_unpreserve - drop_low_array) < preserve_array
        warning_flag = np.any(preserve_trigger)
        max_rank = np.where(preserve_trigger, ldf_count, max_rank_unpreserve)
        min_rank = np.where(preserve_trigger, 0, drop_low_array)

        # dropping
        index_array_weights = (
            factor_ranks.transpose((0, 1, 3, 2)) < max_rank[..., None]
        ) & (factor_ranks.transpose((0, 1, 3, 2)) > min_rank[..., None] - 1)

        if warning_flag:
            if self.preserve == 1:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(self.preserve)
                    + " (use preserve = ...)"
                    + " link ratio(s) is required for development estimation."
                )
            else:
                warning = (
                    "Some exclusions have been ignored. At least "
                    + str(self.preserve)
                    + " link ratio(s) is required for development estimation."
                )
            warnings.warn(warning)

        return index_array_weights.transpose((0, 1, 3, 2)).astype(float)
