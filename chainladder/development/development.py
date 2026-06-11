# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

from chainladder.development.base import DevelopmentBase
from chainladder.utils import WeightedRegression
from chainladder.utils.utility_functions import num_to_nan

from typing import (
    Callable,
    Literal,
    # Self,  # Make use of this once Python 3.10 is deprecated.
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from chainladder.core.typing import TriangleLike
    from numpy.typing import ArrayLike
    from pandas import Series
    from types import ModuleType


class Development(DevelopmentBase):
    """
    A Transformer that allows for basic loss development pattern selection.

    Parameters
    ----------
    n_periods: integer, optional (default = -1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods = -1
    average: literal (or list of literals), or float, optional (default = 'volume')
        type of averaging to use for ldf average calculation.
        Options include 'volume', 'simple',  'regression', and 'geometric'. If numeric values are supplied,
        then (2-average) in the style of Zehnwirth & Barnett is used
        for the exponent of the regression weights.
    sigma_interpolation: string optional (default = 'log-linear')
        Options include 'log-linear' and 'mack'
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
    fillna: float, (default = None)
        Used to fill in zero or nan values of an triangle with some non-zero
        amount.  When an link-ratio has zero as its denominator, it is automatically
        excluded from the ``ldf_`` calculation.  For the specific case of 'volume'
        averaging in a deterministic method, this may be reasonable.  For all other
        averages and stochastic methods, this assumption should be avoided.
    groupby: Callable, list, str, Series (default = None)
        An option to group levels of the triangle index together for the purposes
        of estimating patterns.  If omitted, each level of the triangle
        index will receive its own patterns.

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
    ldf_: Triangle
        The estimated loss development patterns
    cdf_: Triangle
        The estimated cumulative development patterns
    sigma_: Triangle
        Sigma of the ldf regression
    std_err_: Triangle
        Std_err of the ldf regression
    std_residuals_: Triangle
        A Triangle representing the weighted standardized residuals of the
        estimator as described in Barnett and Zehnwirth.

    Examples
    --------

    There are lots of parameters to control the development pattern selection.
    One should exercise caution when multiple drop parameters are used together.

    Let's start with a triangle and inspect its link ratios.

    ..  testsetup::

        import chainladder as cl

    ..  testcode::

        tri = cl.load_sample("xyz")
        print(tri["Incurred"].link_ratio)

    ..  testoutput::

                 12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
        1998       NaN       NaN  1.108227  1.067528  1.064392  1.044146  1.114243  0.987596  0.979707  0.999179
        1999       NaN  1.237646  1.197135  1.144305  1.057447  1.055967  0.988085  1.011131  1.001436       NaN
        2000  1.196032  1.168062  1.239452  1.086354  1.168543  1.072291  1.015048  0.993094       NaN       NaN
        2001  1.353175  1.313547  1.264295  1.286967  1.085689  1.037834  1.006668       NaN       NaN       NaN
        2002  1.590040  1.308591  1.413078  1.179122  1.096524  0.989076       NaN       NaN       NaN       NaN
        2003  1.760957  1.786055  1.337353  1.089595  1.003210       NaN       NaN       NaN       NaN       NaN
        2004  2.364225  1.465057  1.218140  0.980211       NaN       NaN       NaN       NaN       NaN       NaN
        2005  1.654181  1.482965  1.004478       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2006  1.728479  1.043199       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2007  1.629204       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

    We can drop a specific origin/development combination by passing a tuple to the drop parameter. By using
    the `fit_transform` method, we can see the link ratios after the drop.

    ..  testcode::

        print(cl.Development(drop=("2004", 12)).fit_transform(tri["Incurred"]).link_ratio)

    ..  testoutput::

                 12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
        1998       NaN       NaN  1.108227  1.067528  1.064392  1.044146  1.114243  0.987596  0.979707  0.999179
        1999       NaN  1.237646  1.197135  1.144305  1.057447  1.055967  0.988085  1.011131  1.001436       NaN
        2000  1.196032  1.168062  1.239452  1.086354  1.168543  1.072291  1.015048  0.993094       NaN       NaN
        2001  1.353175  1.313547  1.264295  1.286967  1.085689  1.037834  1.006668       NaN       NaN       NaN
        2002  1.590040  1.308591  1.413078  1.179122  1.096524  0.989076       NaN       NaN       NaN       NaN
        2003  1.760957  1.786055  1.337353  1.089595  1.003210       NaN       NaN       NaN       NaN       NaN
        2004       NaN  1.465057  1.218140  0.980211       NaN       NaN       NaN       NaN       NaN       NaN
        2005  1.654181  1.482965  1.004478       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2006  1.728479  1.043199       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2007  1.629204       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

    We can also inspect the LDFs with the `ldf_` property.

    ..  testcode::

        ldf = cl.Development(drop=("2004", 12)).fit(tri["Incurred"]).ldf_
        print(ldf)

    ..  testoutput::

                  12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
        (All)  1.582216  1.339353  1.193406  1.095906  1.076968  1.033612  1.019016  0.997636  0.992918  0.999179

    We can also drop all link ratio(s) above the given parameter by passing a list of floats to the drop_above parameter.
    Let's inspect the link ratios after the drop.

    ..  testcode::

        tri = cl.load_sample("xyz")
        print(
            cl.Development(drop_above=[2.0, 1.5, 1.3, 1.2, 1.1, 1.07, 1.05, 1.03, 1.01, 1.00])
            .fit_transform(tri["Incurred"])
            .link_ratio
        )

    ..  testoutput::

                 12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
        1998       NaN       NaN  1.108227  1.067528  1.064392  1.044146       NaN  0.987596  0.979707  0.999179
        1999       NaN  1.237646  1.197135  1.144305  1.057447  1.055967  0.988085  1.011131  1.001436       NaN
        2000  1.196032  1.168062  1.239452  1.086354       NaN       NaN  1.015048  0.993094       NaN       NaN
        2001  1.353175  1.313547  1.264295       NaN  1.085689  1.037834  1.006668       NaN       NaN       NaN
        2002  1.590040  1.308591       NaN  1.179122  1.096524  0.989076       NaN       NaN       NaN       NaN
        2003  1.760957       NaN       NaN  1.089595  1.003210       NaN       NaN       NaN       NaN       NaN
        2004       NaN  1.465057  1.218140  0.980211       NaN       NaN       NaN       NaN       NaN       NaN
        2005  1.654181  1.482965  1.004478       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2006  1.728479  1.043199       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2007  1.629204       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

    Then, let the LDFs.

    ..  testcode::

        tri = cl.load_sample("xyz")
        ldf = (
            cl.Development(drop_above=[2.0, 1.5, 1.3, 1.2, 1.1, 1.07, 1.05, 1.03, 1.01, 1.00])
            .fit(tri["Incurred"])
            .ldf_
        )
        print(ldf)

    ..  testoutput::

                  12-24     24-36     36-48     48-60     60-72     72-84    84-96    96-108   108-120   120-132
        (All)  1.582216  1.301914  1.142205  1.071625  1.059935  1.022835  1.00511  0.997636  0.992918  0.999179

    We can also use multiple drop parameters together.
    Let's say, we want to drop all link ratio(s) above 1.25 and below 1.0, but only drop these link ratios when
    there are 3 or more link ratios remaining.

    ..  testcode::

        tri = cl.load_sample("xyz")
        print(
            cl.Development(drop_above=1.25, drop_below=1.0, preserve=3)
            .fit_transform(tri["Incurred"])
            .link_ratio
        )

    ..  testoutput::

                 12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
        1998       NaN       NaN  1.108227  1.067528  1.064392  1.044146  1.114243  0.987596  0.979707  0.999179
        1999       NaN  1.237646  1.197135  1.144305  1.057447  1.055967       NaN  1.011131  1.001436       NaN
        2000  1.196032  1.168062  1.239452  1.086354  1.168543  1.072291  1.015048  0.993094       NaN       NaN
        2001  1.353175       NaN       NaN       NaN  1.085689  1.037834  1.006668       NaN       NaN       NaN
        2002  1.590040       NaN       NaN  1.179122  1.096524       NaN       NaN       NaN       NaN       NaN
        2003  1.760957       NaN       NaN  1.089595  1.003210       NaN       NaN       NaN       NaN       NaN
        2004  2.364225       NaN  1.218140       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2005  1.654181       NaN  1.004478       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2006  1.728479  1.043199       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        2007  1.629204       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

    Notice that the link ratios above 1.25 and below 1.0 are not dropped, unless there are 3 or more link ratios
    remaining. For example, the 12-24 link ratio is not dropped because there are 2 valid non-NaN link ratios
    remaining after removing all link ratios above 1.25 and below 1.0. While the 24-36 link ratio is dropped
    because there are still 3 valid non-NaN link ratios remaining after removing all link ratios above 1.25 and
    below 1.0.

    Let's inspect the LDFs.

    ..  testcode::

        ldf = (
            cl.Development(drop_above=1.25, drop_below=1.0, preserve=3)
            .fit(tri["Incurred"])
            .ldf_
        )
        print(ldf)

    ..  testoutput::

                  12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
        (All)  1.675693  1.105627  1.127842  1.119324  1.076968  1.053434  1.027623  0.997636  0.992918  0.999179

    Using other average methods, we can see that the loss development factors are different.

    ..  testcode::

        tri = cl.load_sample("xyz")
        ldf = (
            cl.Development(average="simple").fit(tri["Incurred"]).ldf_
        )
        print(ldf)

    ..  testoutput::

                  12-24    24-36    36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132
        (All)  1.659537  1.35064  1.22277  1.119155  1.079301  1.039863  1.031011  0.997274  0.990571  0.999179


    """

    def __init__(
        self,
        n_periods: int = -1,
        average: Literal["volume", "simple", "regression", "geometric"] = "volume",
        sigma_interpolation: Literal["log-linear", "mack"] = "log-linear",
        drop: tuple | list[tuple] | None = None,
        drop_high: bool | int | list[bool] | list[int] | None = None,
        drop_low: bool | int | list[bool] | list[int] | None = None,
        preserve: int = 1,
        drop_valuation: str | list[str] = None,
        drop_above: float = np.inf,
        drop_below: float = 0.00,
        fillna: float | None = None,
        groupby: Callable | list | str | Series = None,
    ):
        self.n_periods = n_periods
        self.average = average
        self.sigma_interpolation = sigma_interpolation
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.preserve = preserve
        self.drop_valuation = drop_valuation
        self.drop_above = drop_above
        self.drop_below = drop_below
        self.drop = drop
        self.fillna = fillna
        self.groupby = groupby

        # Undeclared until fitted attributes - scikit-learn convention.
        self.average_: np.ndarray

    def fit(self, X: TriangleLike, y: None = None, sample_weight: None = None):
        """Fit the model with X.

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

        # Triangle must be cumulative and in "development" mode.
        obj: TriangleLike = self._set_fit_groups(X).incr_to_cum().val_to_dev().copy()
        xp: ModuleType = obj.get_array_module()

        if self.fillna:
            tri_array: ArrayLike = num_to_nan((obj + self.fillna).values)
        else:
            tri_array: ArrayLike = num_to_nan(obj.values.copy())

        average_: np.ndarray = self._validate_assumption(X, self.average, axis=3)[
            ..., : X.shape[3] - 1
        ]

        # noinspection PyAttributeOutsideInit
        self.average_: np.ndarray = average_.flatten()
        n_periods_: np.ndarray = self._validate_assumption(X, self.n_periods, axis=3)[
            ..., : X.shape[3] - 1
        ]

        x: ArrayLike
        y: ArrayLike
        x, y = tri_array[..., :-1], tri_array[..., 1:]

        link_ratio: ArrayLike = y / x

        if hasattr(X, "w_v2_"):
            self.w_v2_ = self._set_weight_func(
                factor=obj.age_to_age * X.w_v2_,
            )
        else:
            self.w_v2_ = self._set_weight_func(
                factor=obj.age_to_age,
            )

        self.w_ = self._assign_n_periods_weight(
            obj, n_periods_
        ) * self._drop_adjustment(obj, link_ratio)

        params = WeightedRegression(axis=2, thru_orig=True, xp=xp).fit(
            x, y, self.w_, average_
        )

        if self.n_periods != 1:
            params.sigma_fill(self.sigma_interpolation).std_err_fill()
            w_reg = params._w_reg
        else:
            warnings.warn(
                "Setting n_periods=1 does not allow enough degrees "
                "of freedom to support calculation of all regression "
                "statistics. Only LDFs have been calculated."
            )
            w_reg = params._w_reg

        params = xp.concatenate((params.slope_, params.sigma_, params.std_err_), 3)
        params = xp.swapaxes(params, 2, 3)

        self.ldf_ = self._param_property(obj, params, 0)
        self.sigma_ = self._param_property(obj, params, 1)
        self.std_err_ = self._param_property(obj, params, 2)

        resid = -obj.iloc[..., :-1] * self.ldf_.values + obj.iloc[..., 1:].values
        std = xp.sqrt((1 / num_to_nan(w_reg)) * (self.sigma_**2).values)
        resid = resid / num_to_nan(std)
        self.std_residuals_ = resid[resid.valuation < obj.valuation_date].fillzero()

        return self

    def transform(self, X):
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
        X_new.group_index = self._set_transform_groups(X_new)
        triangles = [
            "std_err_",
            "ldf_",
            "sigma_",
            "std_residuals_",
            "average_",
            "w_",
            "sigma_interpolation",
            "w_v2_",
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
