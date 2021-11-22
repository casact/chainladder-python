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
        """ Used for assigning group_index in fit """
        backend = 'numpy' if X.array_backend in ['sparse', 'numpy'] else 'cupy'
        if self.groupby is None:
            return X.set_backend(backend)
        if callable(self.groupby) or type(self.groupby) in [list, str, pd.Series]:
            return X.groupby(self.groupby).sum().set_backend(backend)
        else:
            raise ValueError("Cannot determine groupings.")

    def _set_transform_groups(self, X):
        """ Used for assigning group_index in transform """
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
        """ Used to apply the n_periods weight """

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
        if self.drop_high == self.drop_low == self.drop == self.drop_valuation is None:
            return weight
        if (self.drop_high is not None) | (self.drop_low is not None):
            weight = weight * self._drop_n(self.drop_high, self.drop_low, X, link_ratio, self.preserve)
        # if self.drop_low is not None:
        #     weight = weight * self._drop_hilo("low", X, link_ratio)
        if self.drop is not None:
            weight = weight * self._drop(X)
        if self.drop_valuation is not None:
            weight = weight * self._drop_valuation(X)
            
        print("FINAL WEIGHT:\n", weight)
        return weight

    def _drop_n(self, drop_high, drop_low, X, link_ratio, preserve):
        link_ratios_len = len(link_ratio[0,0])
        
        def drop_array_helper(drop_type):
            drop_type_array = np.array(link_ratios_len*[0])
            
            if drop_type is None:
                return np.array(link_ratios_len*[0])
            else:
                # only a single parameter is provided
                if isinstance(drop_type, int):
                    drop_type_array = np.array(link_ratios_len*[drop_type])
                elif isinstance(drop_type, bool):
                    drop_type_array = np.array(link_ratios_len*int(drop_type))

                # an array of parameters is provided
                else:
                    for index in range(len(drop_type)):
                        drop_type_array[index] = int(drop_type[index])

                # convert boolean to ints (1s)
                for index in range(len(drop_type_array)):
                    if isinstance(drop_type_array[index], bool):
                        drop_type_array[index] = int(drop_type_array[index] == True)
                    else :
                        drop_type_array[index] = drop_type_array[index]

                return drop_type_array
                
        drop_high_array = drop_array_helper(drop_high)
        drop_low_array = drop_array_helper(drop_low)
        print("drop_high_array", drop_high_array)
        print("drop_low_array", drop_low_array)
            
        link_ratio_ranks = link_ratio[0][0].argsort(axis=0).argsort(axis=0)
        print(link_ratio_ranks)
        
        weights=link_ratio[0][0].T
        warning_flag = False
        
        # checking to see if the ranks are in range
        for index in range(len(drop_high_array)-1):
            max_rank = link_ratios_len - index - drop_high_array[index]
            # weights[index] = link_ratio_ranks.T[index] < max_rank
            if max_rank > preserve:
                weights[index] = link_ratio_ranks.T[index] < max_rank - 1
            else:
                weights[index] = 1
                warning_flag = True
        
        if warning_flag:
            warnings.warn(
                "Exclusion is ignored for some age-to-age factor when the selected exclusion result in the number of remaining factors being less than the minimum age-to-age data required (default preserve = 1).")
            
        return weights.T[None, None]
    
    def _drop_hilo(self, kind, X, link_ratio):
        print("in drop hilo")
        xp = X.get_array_module()
        link_ratio[link_ratio == 0] = xp.nan
        # small perturbation to have only one max/min per development age
        link_ratio = link_ratio + xp.random.rand(*list(link_ratio.shape)) / 1e8
        lr_valid_count = xp.sum(~xp.isnan(link_ratio)[0, 0], axis=0)
        if kind == "high":
            vals = xp.nanmax(link_ratio, -2, keepdims=True)
            drop_hilo = self.drop_high
        else:
            vals = xp.nanmin(link_ratio, -2, keepdims=True)
            drop_hilo = self.drop_low
        hilo = 1 * (vals != link_ratio)
        if type(drop_hilo) is bool:
            drop_hilo = [drop_hilo] * (len(X.development) - 1)
        for num in range((len(X.development) - 1)):
            if not drop_hilo[num]:
                hilo[..., num] = hilo[..., num] * 0 + 1
            else:
                if lr_valid_count[num] < 3:
                    hilo[..., num] = hilo[..., num] * 0 + 1
                    warnings.warn(
                        "drop_high and drop_low cannot be computed when less than three LDFs are present, exclusions are ignored."
                    )
        return hilo

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
