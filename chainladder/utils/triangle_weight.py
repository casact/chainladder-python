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
from chainladder.development.base import DevelopmentBase
from pandas.api.types import is_string_dtype

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core.typing import TriangleLike


class TriangleWeight(DevelopmentBase):
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

            When multiple drop parameters are used together, the weights are built in this order (steps 4 and 5 are reversed from `Development`):

            1. ``n_periods`` — limit to the most recent origin periods.
            2. ``drop`` — remove specific origin/development cells.
            3. ``drop_valuation`` — remove entire valuation diagonal in the triangle.
            4. ``drop_above`` / ``drop_below`` — remove link ratios outside a range
               (Protected by``preserve``, which may relax exclusions from this step if too few ratios would remain
               then this step is skipped).
            5. ``drop_high`` / ``drop_low`` — remove highest/lowest link ratios by rank
               (eligible factors from ``n_periods`` are used; protected by ``preserve``,
               which may relax exclusions from this step if too few ratios would remain then this step is skipped).
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

    def _cascade_param(self, size, param, default_param):
        return self._param_array_helper(size, param, default_param)

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
                factor=X * X.w_,
            )
        else:
            self.w_ = self._set_weight_func(
                factor=X,
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
