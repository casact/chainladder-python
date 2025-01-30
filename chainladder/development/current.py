# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
from chainladder.utils import WeightedRegression
from chainladder.development.base import DevelopmentBase
from chainladder.utils.utility_functions import num_to_nan


class CurrentDevelopment(DevelopmentBase):
    """A Transformer that calculates development factors using the ratio of
    incremental values between consecutive development periods.

    Parameters
    ----------
    n_periods: integer, optional (default = -1)
        number of origin periods to be used in the calculation. For
        all origin periods, set n_periods = -1
    n_developments: integer, optional (default = -1)
        number of prior development periods to be used in the calculation. For
        all development periods, set n_developments = -1
    average: str, optional (default = 'volume')
        type of averaging to use for factor calculation. Options are
        'volume', 'simple', and 'regression'
    groupby: str, list, or callable, optional (default = None)
        if provided, indicates how to group triangles before calculating
        development patterns. Can be a string (for single column grouping),
        list of strings (for multi-column grouping), or callable function
    """

    def __init__(
        self,
        n_periods=-1,
        n_developments=-1,
        average="volume",
        sigma_interpolation="log-linear",
        drop=None,
        drop_high=None,
        drop_low=None,
        preserve=1,
        drop_valuation=None,
        drop_above=np.inf,
        drop_below=0.00,
        fillna=None,
        groupby=None,
    ):
        self.n_periods = n_periods
        self.n_developments = n_developments
        self.average = average
        self.groupby = groupby
        self.sigma_interpolation = sigma_interpolation
        self.drop = drop
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.preserve = preserve
        self.drop_valuation = drop_valuation
        self.drop_above = drop_above
        self.drop_below = drop_below
        self.fillna = fillna
        self.groupby = groupby

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X using incremental value ratios.

        Parameters
        ----------
        X : Triangle-like
            Triangle of incremental values to calculate development patterns.
        y : None
            Ignored
        sample_weight : None
            Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # Triangle must be incremental and in development mode
        obj = self._set_fit_groups(X).cum_to_incr().copy()
        xp = obj.get_array_module()

        tri_array = num_to_nan(obj.values.copy())

        if self.fillna:
            tri_array = num_to_nan((obj + self.fillna).values)
        else:
            tri_array = num_to_nan(obj.values.copy())

        average_ = self._validate_assumption(X, self.average, axis=3)[
            ..., : X.shape[3] - 1
        ]
        self.average_ = average_.flatten()
        n_periods_ = self._validate_assumption(X, self.n_periods, axis=3)[
            ..., : X.shape[3] - 1
        ]

        n_developments_ = self._validate_assumption(X, self.n_developments, axis=3)[
            ..., : X.shape[3] - 1
        ]

        # Initialize 4D array for link ratios with same shape as input minus last development period
        link_ratio = xp.zeros((*tri_array.shape[:-1], tri_array.shape[-1] - 1))
        
        # Initialize arrays to store x and y values
        x = xp.zeros_like(tri_array[..., :-1]) 
        y = xp.zeros_like(tri_array[..., 1:])

        # Iterate through the innermost 2D arrays of tri_array
        for i in range(tri_array.shape[0]):  # First dimension
            for j in range(tri_array.shape[1]):  # Second dimension
                inner_array = tri_array[i, j, :, :]  # Get the 2D slice
                y[i, j] = inner_array[..., 1:]

                # create the x array. This is starts as inner_array[..., :-1], but then the divisor for each y
                # is the sum of the prior n_dev periods.
                x_slice = inner_array[..., :-1]

                for col in range(x_slice.shape[1]):
                    n_dev = int(n_developments_[..., col])
                    # Sum the prior n_dev periods, but don't go before column 0
                    start_col = max(0, col - n_dev + 1)
                    x_slice[:, col] = xp.sum(inner_array[:, start_col : col + 1], axis=1)
                
                x[i, j] = x_slice
                link_ratio[i, j, :, :] = y[i, j] / x[i, j]

        exponent = xp.array(
            [{"regression": 0, "volume": 1, "simple": 2}[x] for x in average_[0, 0, 0]]
        )
        exponent = xp.nan_to_num(exponent * (y * 0 + 1))

        self.w_ = self._assign_n_periods_weight(
            obj, n_periods_
        ) * self._drop_adjustment(obj, link_ratio)
        w = num_to_nan(self.w_ / (x ** (exponent)))

        # Calculate development factors as ratio of incremental values
        params = WeightedRegression(axis=2, thru_orig=True, xp=xp).fit(x, y, w)
        params = xp.concatenate((params.slope_, params.sigma_, params.std_err_), 3)
        params = xp.swapaxes(params, 2, 3)

        # Set properties
        self.ldf_ = self._param_property(obj, params, 0)
        self.sigma_ = self._param_property(obj, params, 1)
        self.std_err_ = self._param_property(obj, params, 2)

        return self

    def transform(self, X):
        """Transform X using the fitted parameters.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = X.copy()
        X_new.group_index = self._set_transform_groups(X_new)
        triangles = [
            "std_err_",
            "ldf_",
            "sigma_",
        ]
        for item in triangles:
            setattr(X_new, item, getattr(self, item, None))

        X_new._set_slicers()
        return X_new
    
    def _param_property(self, X, params, idx):
        from chainladder import options

        obj = X[X.origin == X.origin.min()]
        xp = X.get_array_module()
        obj.values = xp.ones(obj.shape)[..., :-1] * params[..., idx : idx + 1, :]
        obj.ddims = X.link_ratio.ddims
        obj.valuation_date = pd.to_datetime(options.ULT_VAL)
        obj.is_pattern = True
        obj.is_cumulative = False
        obj.virtual_columns.columns = {}
        obj._set_slicers()

        return obj
