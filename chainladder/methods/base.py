# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pandas as pd
import warnings
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development
from chainladder.core.io import EstimatorIO
from chainladder.core.common import Common

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core import Triangle


class MethodBase(BaseEstimator, EstimatorIO, Common):
    _estimator_type = "chainladder"

    def validate_X(self, X):  # noqa: N802
        obj = X.copy()
        if "ldf_" not in obj:
            obj = Development().fit_transform(obj)
        if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
            obj = TailConstant().fit_transform(obj)
        return obj.val_to_dev()

    def _align_cdf(self, X, sample_weight=None):
        """Vertically align CDF to origin period latest diagonal."""
        return X.cdf_.align_pattern(X, sample_weight)

    def _set_ult_attr(self, ultimate):
        """Ultimate scaffolding"""
        from chainladder import options

        xp = ultimate.get_array_module()
        if ultimate.array_backend != "sparse":
            ultimate.values[~xp.isfinite(ultimate.values)] = xp.nan
        ultimate.ddims = pd.DatetimeIndex([options.ULT_VAL])
        ultimate.virtual_columns.columns = {}
        ultimate.is_cumulative = True
        ultimate._set_slicers()
        ultimate.valuation_date = ultimate.valuation.max()
        ultimate._drop_subtriangles()
        return ultimate

    @property
    def ldf_(self):
        return self.X_.ldf_

    @property
    def latest_diagonal(self):
        if self.X_.is_cumulative:
            return self.X_.latest_diagonal
        else:
            return self.X_.sum("development")

    def fit(self, X, y=None, sample_weight=None):
        """Applies the chainladder technique to triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : Ignored
        sample_weight : Triangle
            For exposure-based methods, the exposure to be used for fitting

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = self.validate_X(X)
        self.validate_weight(X, sample_weight)
        if sample_weight:
            self.sample_weight_ = sample_weight.set_backend(self.X_.array_backend)
        else:
            self.sample_weight_ = sample_weight
        return self

    def predict(self, X, sample_weight=None):
        """Predicts the chainladder ultimate on a new triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        sample_weight : Triangle
            For exposure-based methods, the exposure to be used for predictions

        Returns
        -------
        X_new: Triangle

        """
        X_new = X.val_to_dev()
        if sum(X_new.ddims > self.ldf_.ddims.max()) > 0:
            raise ValueError("X has ages that exceed those available in model.")
        # Before the line below, which borrows self.X_'s index when both sides
        # are a single row and so would erase what the caller actually passed.
        self.validate_ldf(X_new, self.ldf_)
        X_new = X_new + (self.X_.val_to_dev().iloc[0, 0].sum(2) * 0)
        self.validate_weight(X_new, sample_weight)
        if sample_weight:
            sample_weight = sample_weight.set_backend(X_new.array_backend)
        X_new.ldf_ = self.ldf_
        X_new, X_new.ldf_ = self.intersection(X_new, X_new.ldf_)
        return X_new

    def intersection(self, a, b):
        """Given two Triangles with mismatched indices, this method aligns
        their indices"""
        if len(a) == 1 and len(b) == 1:
            return a, b
        intersection = list(set(a.key_labels).intersection(set(b.key_labels)))
        if intersection == []:
            return a, b
        a_idx = a.index[intersection]
        b_idx = b.index[intersection]
        idx_intersection = list(
            set(
                a_idx.set_index(intersection).index.intersection(
                    b_idx.set_index(intersection).index
                )
            )
        )
        if (len(a) == 1 or len(b) == 1) and idx_intersection == []:
            return a, b
        b = b.iloc[
            b_idx[
                b_idx[intersection].set_index(intersection).index.isin(idx_intersection)
            ].index
        ]
        a = a.iloc[
            a_idx[
                a_idx[intersection].set_index(intersection).index.isin(idx_intersection)
            ].index
        ]
        return a, b

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.predict(X, sample_weight)

    def _include_process_variance(self):
        if hasattr(self.X_, "_get_process_variance"):
            full = self.full_triangle_
            obj = self.X_._get_process_variance(full)
            self.ultimate_.values = obj.values[..., -1:]
            process_var = obj - full
        else:
            process_var = None
        return process_var

    @staticmethod
    def validate_ldf(X: Triangle, ldf: Triangle) -> None:
        """
        Checks that a fitted pattern can be applied to X as it was passed in.
        The index and the columns of the two have to line up: values or columns
        X carries that the pattern does not cannot be predicted, and index
        levels the pattern carries that X does not cannot be applied.
        """
        # A pattern whose index is entirely the "(All)" sentinel that Triangle.sum
        # sets carries no group identity, so nothing about it constrains what it
        # may be applied to. Note the limit of that: sum() stamps "(All)" on
        # whatever subset it was called on, so a pattern summed from one line of
        # business is exempt here just as a pattern summed from everything is.
        # Telling those apart needs aggregation provenance on the Triangle.
        if len(ldf) == 1 and set(ldf.index.values.flatten()) == {"(All)"}:
            return
        shared = sorted(set(X.key_labels) & set(ldf.key_labels))
        if shared:
            missing = sorted(
                set(X.index.set_index(shared).index)
                - set(ldf.index.set_index(shared).index)
            )
            if missing:
                raise ValueError(
                    "X has index values the model was not fit on: "
                    + str(missing[:5])
                    + (", and others" if len(missing) > 5 else "")
                )
        columns = sorted(set(X.columns) - set(ldf.columns))
        if columns:
            raise ValueError("X has columns the model was not fit on: " + str(columns))
        finer = sorted(set(ldf.key_labels) - set(X.key_labels))
        if finer:
            raise ValueError(
                "The fitted pattern has index levels that X does not: "
                + str(finer)
                + ". It cannot be applied to a triangle that does not carry them."
            )

    @staticmethod
    def validate_weight(
        X: Triangle,
        sample_weight: Triangle,
    ) -> None:
        """
        Checks that the a aprior has valid dimensions
        """
        if (
            sample_weight
            and X.shape[:-1] != sample_weight.shape[:-1]
            and sample_weight.shape[2] != 1
            and sample_weight.shape[0] > 1
        ):
            warnings.warn(
                "X and sample_weight are not aligned. Broadcasting may occur.\n"
            )
