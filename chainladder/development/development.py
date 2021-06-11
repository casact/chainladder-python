# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import warnings
from chainladder.utils import WeightedRegression
from chainladder.development.base import DevelopmentBase

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
    groupby :
        An option to group levels of the triangle index together for the purposes
        of estimating patterns.  If omitted, each level of the triangle
        index will receive its own patterns.


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
    std_residuals_ : Triangle
        A Triangle representing the weighted standardized residuals of the
        estimator as described in Barnett and Zehnwirth.

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
        groupby=None
    ):
        self.n_periods = n_periods
        self.average = average
        self.sigma_interpolation = sigma_interpolation
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_valuation = drop_valuation
        self.drop = drop
        self.fillna = fillna
        self.groupby = groupby

    def _validate_axis_assumption(self, parameter, axis):
        if callable(parameter):
            return axis.map(parameter).to_list()
        if type(parameter) in [int, str, float]:
            return [parameter] * len(axis)
        return parameter

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
        from chainladder.utils.utility_functions import num_to_nan

        # Validate inputs
        if X.is_cumulative == False:
            obj = self._set_fit_groups(X).incr_to_cum().val_to_dev().copy()
        else:
            obj = self._set_fit_groups(X).val_to_dev().copy()
        xp = obj.get_array_module()

        # Make sure it is a dev tri
        if type(obj.ddims) != np.ndarray:
            raise ValueError("Triangle must be expressed with development lags")
        # validate hyperparameters
        if self.fillna:
            tri_array = num_to_nan((obj + self.fillna).values)
        else:
            tri_array = num_to_nan(obj.values.copy())
        self.average_ = np.array(
            self._validate_axis_assumption(self.average, obj.development[:-1]))
        n_periods_ = self._validate_axis_assumption(self.n_periods, obj.development[:-1])
        weight_dict = {"regression": 0, "volume": 1, "simple": 2}
        x, y = tri_array[..., :-1], tri_array[..., 1:]
        exponent = xp.array([weight_dict.get(item, item) for item in self.average_])
        exponent = xp.nan_to_num(exponent[None, None, None] * (y * 0 + 1))
        link_ratio = y / x
        self.w_ = xp.array(
            self._assign_n_periods_weight(obj, n_periods_) *
            self._drop_adjustment(obj, link_ratio))
        w = self.w_ / (x ** (exponent))
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
            / xp.swapaxes(xp.sqrt(x ** (2 - exponent))[..., 0:1, :], -1, -2)
        )
        params = xp.concatenate((params.slope_, params.sigma_, params.std_err_), 3)
        params = xp.swapaxes(params, 2, 3)
        self.ldf_ = self._param_property(obj, params, 0)
        self.sigma_ = self._param_property(obj, params, 1)
        self.std_err_ = self._param_property(obj, params, 2)

        resid = -obj.iloc[..., :-1] * self.ldf_.values + obj.iloc[..., 1:].values

        std = xp.sqrt((1/num_to_nan(w))*(self.sigma_**2).values)
        resid = resid/std
        self.std_residuals_ = resid[resid.valuation < obj.valuation_date]
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
        X_new.group_index = self._set_transform_groups(X_new)
        triangles = ["std_err_", "ldf_", "sigma_"]
        for item in triangles + ["average_", "w_", "sigma_interpolation"]:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new

    @property
    def weight_(self):
        return pd.DataFrame(
            self.w_[0, 0],
            index=self.ldf_.origin,
            columns=list(self.ldf_.development.values[:, 0]),
        )

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
