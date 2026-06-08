# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
from chainladder.utils.sparse import sp
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

from chainladder.utils.utility_functions import num_to_nan
from pandas.api.types import is_string_dtype

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core.typing import TriangleLike

class TriangleWeight(BaseEstimator,TransformerMixin):
    """
    Helper class that produces a triangle of weights based on pattern selections

    Parameters
    ----------
    n_periods: integer, optional (default = -1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods = -1
    drop: tuple or list of tuples
        Drops specific origin/development combination(s). See order of operations
        below when combined with multiple drop parameters.
    drop_high: bool, int, list of bools, or list of ints (default = None)
        Drops highest (by rank) link ratio(s) from LDF calculation
        If a boolean variable is passed, drop_high is set to 1, dropping only the
        highest value. Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    drop_low: bool, int, list of bools, or list of ints (default = None)
        Drops lowest (by rank) link ratio(s) from LDF calculation
        If a boolean variable is passed, drop_low is set to 1, dropping only the
        lowest value. Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    drop_above: float or list of floats (default = numpy.inf)
        Drops all link ratio(s) above the given parameter from the LDF calculation.
        Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    drop_below: float or list of floats (default = 0.00)
        Drops all link ratio(s) below the given parameter from the LDF calculation.
        Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    preserve: int (default = 1)
        The minimum number of link ratio(s) required for LDF calculation.
        See order of operations below when combined with multiple drop parameters.
    drop_valuation: str or list of str (default = None)
        Drops specific valuation periods. str must be date convertible.
        See order of operations below when combined with multiple drop parameters.

        .. note ::
    
            (Order of Drop Operations)
            
            When multiple drop parameters are used together, the weights are built in this order:
        
            1. ``n_periods`` — limit to the most recent origin periods.
            2. ``drop`` — remove specific origin/development cells.
            3. ``drop_valuation`` — remove entire valuation diagonal in the triangle.
            4. ``drop_high`` / ``drop_low`` — remove highest/lowest link ratios by rank
               (eligible factors from ``n_periods`` are used; protected by ``preserve``,
               which may relax exclusions from this step if too few ratios would remain then this step is skipped).
            5. ``drop_above`` / ``drop_below`` — remove link ratios outside a range
               (Protected by``preserve``, which may relax exclusions from this step if too few ratios would remain
               then this step is skipped).
            6. Calculate the loss development factors using ``average`` method.

    Attributes
    ----------
    w_: Triangle
        The weight
    """

    def __init__(
        self,
        n_periods: int = -1,
        drop: tuple | list[tuple] | None = None,
        drop_high: bool | int | list[bool] | list[int] | None = None,
        drop_low: bool | int | list[bool] | list[int] | None = None,
        preserve: int = 1,
        drop_valuation: str | list[str] = None,
        drop_above: float = np.inf,
        drop_below: float = 0.00,
    ):
        self.n_periods = n_periods
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.preserve = preserve
        self.drop_valuation = drop_valuation
        self.drop_above = drop_above
        self.drop_below = drop_below
        self.drop = drop

    def fit(self, X: TriangleLike, y: None = None, sample_weight: None = None):
        """
        Fit the model with X.

        Parameters
        ----------
        X : TriangleLike
            Set of LDFs to which the Munich adjustment will be applied.
        y : None
            Ignored
        sample_weight : None
            Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if hasattr(X, "w_"):
            self.w_ = self._set_weight_func(
                X=X * X.w_,
            )
        else:
            self.w_ = self._set_weight_func(
                X=X,
            )
        return self

    def transform(self, X: TriangleLike):
        """If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
        X_new : New triangle with transformed attributes.

        """
        X_new = X.copy()
        X_new.w_ = self.w_
        return X_new

    def _cascade_param(
            self, 
            size:int, 
            param: bool | int | float | str | None | list[bool|int|float|str|None], 
            default_param: bool | int | float | str | None
    ) -> np.ndarray:
        """
        Internal helper function to explicitly cascade a parameter to a given triangle size

        Parameters
        ----------
        size: integer
            the width of the triangle
        param: bool or int or float or str or None or list
            the selected parameter, such as n_periods or drop_low, etc. 
        default_param: bool or int or float or str or None
            the default param to fill where unspecificied

        Returns
        -------
        numpy array of the cascaded parameter

        """
        # setting dimension and default value of output
        out = pd.Series(size * [default_param]).astype("object")
        # an array of parameters is provided
        if isinstance(param, list):
            out[range(len(param))] = np.array(param)
        # only a single parameter is provided
        else:
            out.loc[:] = param
        # Fill missing with default
        out.loc[out.isna()] = default_param
        # return properly typed numpy array
        return out.astype(type(default_param)).to_numpy()

    def _set_weight_func(
            self,
            X: TriangleLike,
            secondary_rank: TriangleLike | None = None
    ) -> TriangleLike:
        """
        Combines weights from all parameters

        Parameters
        ----------
        X: TriangleLike
            Triangle of values to be weighted
        secondary_rank: TriangleLike
            Triangle of values to break ties for drop_high and drop_low

        Returns
        -------
        A Triangle of weights

        """
        w = (~np.isnan(X.values)).astype(float)
        w = w * self._assign_n_periods_weight_func(X)
        if self.drop is not None:
            w = w * self._drop_func(X)

        if self.drop_valuation is not None:
            w = w * self._drop_valuation_func(X)

        if (self.drop_above != np.inf) | (self.drop_below != 0.0):
            w = w * self._drop_x_func(X)

        if (self.drop_high is not None) | (self.drop_low is not None):
            w = w * self._drop_n_func(X * num_to_nan(w), secondary_rank)

        w_tri = X.copy()
        w_tri.values = num_to_nan(w)

        return w_tri

    def _assign_n_periods_weight_func(self, X: TriangleLike) -> TriangleLike:
        """
        Generates weights for the `n_periods` parameter

        Parameters
        ----------
        X: TriangleLike
            Triangle of values to be weighted

        Returns
        -------
        A Triangle of weights

        """
        # cascading n_periods across all columns
        dev_len = X.shape[3]
        n_periods_param = self._cascade_param(dev_len, self.n_periods, -1)

        #helper function that generates the weights for individual n_periods
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

        xp = X.get_array_module()

        # a dict of weights (val) by n_periods (key)
        dict_map = {
            item: _assign_n_periods_weight_int(X, item)
            for item in set(n_periods_param)
        }
        # collection of development columns based on n_periods specified for that column
        conc = [
            dict_map[item][..., num : num + 1]
            for num, item in enumerate(n_periods_param)
        ]
        return xp.concatenate(tuple(conc), -1).astype(float)

    def _drop_n_func(
            self,
            X: TriangleLike,
            secondary_rank: TriangleLike | None = None
    ) -> TriangleLike:
        """
        Generates weights for the `drop_high` and `drop_low` parameter

        Parameters
        ----------
        X: TriangleLike
            Triangle of values to be weighted
        secondary_rank: TriangleLike
            Triangle of values to break ties

        Returns
        -------
        A Triangle of weights

        """        
        # Preparing to set up 3D array for drop_n parameters
        X_val = X.values.copy()
        dev_len = X_val.shape[3]
        indices = X_val.shape[0]
        columns = X_val.shape[1]

        # secondary rank is the optional triangle that breaks ties in X
        # the original use case is for dropping the link ratio of 1 with the lowest loss value
        # (pass in a reverse rank of loss to drop link of ratio of 1 with the highest loss value)
        # leaving to caller to ensure that secondary rank is the same dimensions as X
        # also leaving to caller to pick whether to trim head or tail
        if secondary_rank is None:
            sec_rank_val = X_val.copy()
        else:
            sec_rank_val = secondary_rank.values.copy()

        # explicitly setting up 3D arrays for drop_n parameters to avoid broadcasting bugs
        drop_high_array = np.zeros((indices, columns, dev_len))
        drop_high_array[:, :, :] = self._cascade_param(
            dev_len, self.drop_high, 0
        )[None, None]
        drop_low_array = np.zeros((indices, columns, dev_len))
        drop_low_array[:, :, :] = self._cascade_param(
            dev_len, self.drop_low, 0
        )[None, None]
        preserve_array = np.zeros((indices, columns, dev_len))
        preserve_array[:, :, :] = self._cascade_param(
            dev_len, self.preserve, self.preserve
        )[None, None]

        # ranking values by itself and secondary rank
        X_ranks = np.lexsort((sec_rank_val, X_val), axis=2).argsort(axis=2)

        # counting valid values per development period
        valid_count = (~np.isnan(X_val)).sum(axis=2)

        # getting max index after drop high
        max_rank_unpreserve = valid_count - drop_high_array

        # applying preserve
        preserve_trigger = (max_rank_unpreserve - drop_low_array) < preserve_array
        
        # setting up flag to produce warning
        warning_flag = np.any(preserve_trigger)

        # getting ranks of values that correspond to the max and min after preserve
        max_rank = np.where(preserve_trigger, valid_count, max_rank_unpreserve)
        min_rank = np.where(preserve_trigger, 0, drop_low_array)

        # getting weights that are within the max and min ranks
        w = (
            X_ranks < max_rank[:,:,None,:]
        ) & (X_ranks > min_rank[:,:,None,:] - 1)

        # NOTE: The "Some exclusions have been ignored..." UserWarning below is
        # asserted by the test suite (see chainladder/development/tests/
        # test_development.py and test_incremental.py, which use
        # pytest.warns(..., match="exclusions have been ignored")).
        # Do not modify the warning message or remove the warnings.warn(...)
        # call without updating the corresponding pytest.warns matchers,
        # otherwise those tests will fail.
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

        return w.astype(float)
    
    def _drop_func(self, X: TriangleLike) -> TriangleLike:
        """
        Generates weights for the `drop` parameter

        Parameters
        ----------
        X: TriangleLike
            Triangle of values to be weighted

        Returns
        -------
        A Triangle of weights

        """        
        # get the appropriate backend for nan_to_num
        xp = X.get_array_module()
        # turn single drop_valuation parameter to list if necessary
        drop_list = self.drop if isinstance(self.drop, list) else [self.drop]
        # get an starting array of weights
        w = X.nan_triangle
        # accommodate ldf triangle where the dimensions are '12-24'
        dev_list = (
            X.development.str.split("-", expand=True)[0]
            if is_string_dtype(X.development)
            else X.development.astype("string")
        )
        # create ndarray of drop_list for further operation in numpy
        drop_np = np.asarray(drop_list)
        # find indices of drop_np
        origin_ind = np.where(
            np.array([X.origin.astype("string")]) == drop_np[:, [0]]
        )[1]
        dev_ind = np.where(np.array([dev_list]) == drop_np[:, [1]])[1]
        # set weight of dropped factors to 0
        w[(origin_ind, dev_ind)] = 0
        return xp.nan_to_num(w)[None, None]

    def _drop_valuation_func(self, X: TriangleLike) -> TriangleLike:
        """
        Generates weights for the `drop` parameter

        Parameters
        ----------
        X: TriangleLike
            Triangle of values to be weighted

        Returns
        -------
        A Triangle of weights

        """        
        # get the appropriate backend for nan_to_num
        xp = X.get_array_module()
        # turn single drop_valuation parameter to list if necessary
        if isinstance(self.drop_valuation, list):
            drop_valuation_list = self.drop_valuation
        else:
            drop_valuation_list = [self.drop_valuation]
        # turn drop_valuation to same valuation freq as X
        v = pd.PeriodIndex(
            drop_valuation_list, freq=X.development_grain
        ).to_timestamp(how="e")
        # warn that some drop_valuation are outside of X
        if np.any(~v.isin(X.valuation)):
            warnings.warn("Some valuations could not be dropped.")
        # return triangle of weight where dropped factors have 0
        w = xp.nan_to_num(X.iloc[0, 0][~X.valuation.isin(v)].values * 0 + 1)
        # check to make sure some factors are still left
        if w.sum() == 0:
            raise Exception("The entire triangle has been dropped via drop_valuation.")
        return w
    
    def _drop_x_func(self, X: TriangleLike) -> TriangleLike:
        """
        Generates weights for the `drop_above` and `drop_below` parameters

        Parameters
        ----------
        X: TriangleLike
            Triangle of values to be weighted

        Returns
        -------
        A Triangle of weights

        """        
        # Preparing to set up 3D array for drop_x parameters
        X_val = X.values.copy()
        dev_len = X_val.shape[3]
        indices = X_val.shape[0]
        columns = X_val.shape[1]

        # explicitly setting up 3D arrays for drop parameters to avoid broadcasting bugs
        drop_above_array = np.zeros((indices, columns, dev_len))
        drop_above_array[:, :, :] = self._cascade_param(
            dev_len, self.drop_above, np.inf
        )[None, None]
        drop_below_array = np.zeros((indices, columns, dev_len))
        drop_below_array[:, :, :] = self._cascade_param(
            dev_len, self.drop_below, 0.0
        )[None, None]
        preserve_array = np.zeros((indices, columns, dev_len))
        preserve_array[:, :, :] = self._cascade_param(
            dev_len, self.preserve, self.preserve
        )[None, None]

        # setting up starting array of weights
        w = ~np.isnan(X_val)

        # weights without considering preserve
        index_array_weights = (X_val < drop_above_array[:,:,None,:]) & (
            X_val > drop_below_array[:,:,None,:]
        )

        # counting remaining factors
        valid_count = index_array_weights.sum(axis=2)

        # applying preserve
        warning_flag = np.any(valid_count < preserve_array)
        w = np.where(
            valid_count[:,:,None,:] < preserve_array[:,:,None,:], w, index_array_weights
        )

        # NOTE: The "Some exclusions have been ignored..." UserWarning below is
        # asserted by the test suite (see chainladder/development/tests/
        # test_development.py and test_incremental.py, which use
        # pytest.warns(..., match="exclusions have been ignored")).
        # Do not modify the warning message or remove the warnings.warn(...)
        # call without updating the corresponding pytest.warns matchers,
        # otherwise those tests will fail.
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

        return w.astype(float)