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
    pass


class Development(DevelopmentBase):
    """ A Transformer that allows for basic loss development pattern selection.

    Parameters
    ----------
    n_periods : integer, optional (default=-1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods=-1
    average : string or float, optional (default='volume')
        type of averaging to use for ldf average calculation.  Options include
        'volume', 'simple', and 'regression'. If numeric values are supplied,
        then (2-average) in the style of Zehnwirth & Barnett is used
        for the exponent of the regression weights.
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
    fillna: float, (default=None)
        Used to fill in zero or nan values of an triangle with some non-zero
        amount.  When an link-ratio has zero as its denominator, it is automatically
        excluded from the ``ldf_`` calculation.  For the specific case of 'volume'
        averaging in a deterministic method, this may be reasonable.  For all other
        averages and stochastic methods, this assumption should be avoided.


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
    weight_ : pandas.DataFrame
        The weight used in the ldf regression

    """

    def __init__(
        self,
        n_periods=-1,
        average="volume",
        sigma_interpolation="log-linear",
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        fillna=None,
    ):
        self.n_periods = n_periods
        self.average = average
        self.sigma_interpolation = sigma_interpolation
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_valuation = drop_valuation
        self.drop = drop
        self.fillna = fillna

    @property
    def weight_(self):
        return pd.DataFrame(
            self.w_[0, 0],
            index=self.ldf_.origin,
            columns=list(self.ldf_.development.values[:, 0]),
        )

    def _assign_n_periods_weight(self, X):
        if type(self.n_periods) is int:
            return self._assign_n_periods_weight_int(X, self.n_periods)[..., :-1]
        elif type(self.n_periods) is list:
            if len(self.n_periods) != X.values.shape[-1] - 1:
                raise ValueError(
                    "n_periods list must be of length {}.".format(
                        X.values.shape[-1] - 1
                    )
                )
            else:
                return self._assign_n_periods_weight_list(X)
        else:
            raise ValueError("n_periods must be of type <int> or <list>")

    def _assign_n_periods_weight_list(self, X):
        """ private method to standardize the n_periods input to a list """
        xp = X.get_array_module()
        dict_map = {
            item: self._assign_n_periods_weight_int(X, item)
            for item in set(self.n_periods)
        }
        conc = [
            dict_map[item][..., num : num + 1]
            for num, item in enumerate(self.n_periods)
        ]
        return xp.concatenate(tuple(conc), -1)

    def _assign_n_periods_weight_int(self, X, n_periods):
        """ Zeros out weights depending on number of periods desired
            Only works for type(n_periods) == int
        """
        xp = X.get_array_module()
        if n_periods < 1 or n_periods >= X.shape[-2] - 1:
            return X.values * 0 + 1
        else:
            val_offset = {
                "Y": {"Y": 1},
                "Q": {"Y": 4, "Q": 1},
                "M": {"Y": 12, "Q": 3, "M": 1},
            }
            val_date_min = (
                X.valuation[X.valuation <= X.valuation_date]
                .drop_duplicates()
                .sort_values()
            )
            val_date_min = val_date_min[
                -n_periods * val_offset[X.development_grain][X.origin_grain] - 1
            ]
            w = X[X.valuation >= val_date_min]
            return xp.nan_to_num((w / w).values) * X.nan_triangle

    def _drop_adjustment(self, X, link_ratio):
        weight = X.nan_triangle[:, :-1]
        if self.drop_high == self.drop_low == self.drop == self.drop_valuation is None:
            return weight
        if self.drop_high is not None:
            weight = weight * self._drop_hilo("high", X, link_ratio)
        if self.drop_low is not None:
            weight = weight * self._drop_hilo("low", X, link_ratio)
        if self.drop is not None:
            weight = weight * self._drop(X)
        if self.drop_valuation is not None:
            weight = weight * self._drop_valuation(X)
        return weight

    def _drop_hilo(self, kind, X, link_ratio):
        xp = X.get_array_module()
        link_ratio[link_ratio == 0] = xp.nan
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
        for num, item in enumerate(self.average_):
            if not drop_hilo[num]:
                hilo[..., num] = hilo[..., num] * 0 + 1
            else:
                if lr_valid_count[num] < 3:
                    hilo[..., num] = hilo[..., num] * 0 + 1
                    warnings.warn(
                        "drop_high and drop_low cannot be computed "
                        "when less than three LDFs are present. "
                        "Ignoring exclusions in some cases."
                    )
        return hilo

    def _drop_valuation(self, X):
        xp = X.get_array_module()
        if type(self.drop_valuation) is not list:
            drop_valuation = [self.drop_valuation]
        else:
            drop_valuation = self.drop_valuation
        arr = 1 - xp.nan_to_num(
            X[
                X.valuation.isin(
                    pd.PeriodIndex(drop_valuation, freq=X.origin_grain).to_timestamp(
                        how="e"
                    )
                )
            ].values[0, 0]
            * 0
            + 1
        )
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

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the munich adjustment will be applied.
        y : None
            Ignored
        sample_weight :
            Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if X.array_backend == "sparse":
            X = X.set_backend("numpy")
        else:
            X = X.copy()
        xp = X.get_array_module()
        from chainladder.utils.utility_functions import num_to_nan

        if type(X.ddims) != np.ndarray:
            raise ValueError("Triangle must be expressed with development lags")
        if self.fillna:
            tri_array = num_to_nan((X + self.fillna).values)
        else:
            tri_array = num_to_nan(X.values.copy())
        if type(self.average) is not list:
            self.average_ = np.array([self.average] * (tri_array.shape[-1] - 1))
        else:
            self.average_ = np.array(self.average)
        if type(self.n_periods) is not list:
            n_periods = [self.n_periods] * (tri_array.shape[-1] - 1)
        else:
            n_periods = self.n_periods
        n_periods = np.array(n_periods)
        self.n_periods_ = n_periods
        weight_dict = {"regression": 0, "volume": 1, "simple": 2}
        x, y = tri_array[..., :-1], tri_array[..., 1:]
        val = xp.nan_to_num(
            xp.array([weight_dict.get(item, item) for item in self.average_])[
                None, None, None
            ]
            * (y * 0 + 1)
        )
        link_ratio = y / x
        self.w_ = xp.array(
            self._assign_n_periods_weight(X) * self._drop_adjustment(X, link_ratio)
        )
        w = self.w_ / (x ** (val))
        params = WeightedRegression(axis=2, thru_orig=True, xp=xp).fit(x, y, w)
        if self.n_periods != 1:
            params = params.sigma_fill(self.sigma_interpolation)
        else:
            warnings.warn(
                "Setting n_periods=1 does not allow enough degrees "
                "of freedom to support calculation of all regression"
                " statistics.  Only LDFs have been calculated."
            )
        params.std_err_ = xp.nan_to_num(params.std_err_) + xp.nan_to_num(
            (1 - xp.nan_to_num(params.std_err_ * 0 + 1))
            * params.sigma_
            / xp.swapaxes(xp.sqrt(x ** (2 - val))[..., 0:1, :], -1, -2)
        )
        params = xp.concatenate((params.slope_, params.sigma_, params.std_err_), 3)
        params = xp.swapaxes(params, 2, 3)
        self.ldf_ = self._param_property(X, params, 0)
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
        X_new = X.copy()
        triangles = ["std_err_", "ldf_", "sigma_"]
        for item in triangles + ["average_", "w_", "sigma_interpolation", "n_periods_"]:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new

    def _param_property(self, X, params, idx):
        from chainladder import ULT_VAL

        obj = X[X.origin == X.origin.min()]
        xp = X.get_array_module()
        obj.values = xp.ones(obj.shape)[..., :-1] * params[..., idx : idx + 1, :]
        obj.ddims = X.link_ratio.ddims
        obj.valuation_date = pd.to_datetime(ULT_VAL)
        obj.is_pattern = True
        obj.is_cumulative = False
        obj.virtual_columns.columns = {}
        obj._set_slicers()
        return obj
