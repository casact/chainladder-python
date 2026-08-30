# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Reserve movement analysis between two valuation dates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from chainladder.core.io import EstimatorIO
from chainladder.core.triangle import Triangle
from chainladder.methods import Chainladder


class ReserveRollforward(BaseEstimator, EstimatorIO):
    """Explain reserve movement between a prior and current valuation.

    A cloned reserving method is fitted at the prior valuation and at the
    current valuation. Expected incremental payments from the prior projection
    are then compared with observed payments in the intervening period.

    Parameters
    ----------
    estimator : estimator, optional
        Reserving estimator with ``ibnr_`` and ``full_triangle_`` after fitting.
        Defaults to :class:`~chainladder.Chainladder`.
    prior_valuation_period : str, optional
        Start of the rollforward. Defaults to the valuation period immediately
        before ``X.valuation_date``. It may be an annual or quarterly period.

    Attributes
    ----------
    detail_ : DataFrame
        Cell-level expected and actual incremental payments.
    summary_ : DataFrame
        Opening IBNR, expected and actual payments, expected and actual closing
        IBNR, and the resulting estimate change.
    """

    def __init__(self, estimator=None, prior_valuation_period: str | None = None):
        self.estimator = estimator
        self.prior_valuation_period = prior_valuation_period

    @staticmethod
    def _to_long(triangle, value_name: str) -> pd.DataFrame:
        """Convert a Triangle to a consistent long-form DataFrame."""
        frame = triangle.to_frame(keepdims=True).reset_index()
        id_vars = triangle.key_labels + ["origin", "development"]
        if "values" in frame:
            frame = frame.loc[:, id_vars + ["values"]]
            frame = frame.rename(columns={"values": value_name})
            frame["column"] = triangle.columns[0]
            return frame
        return frame.melt(
            id_vars=id_vars,
            value_vars=triangle.columns,
            var_name="column",
            value_name=value_name,
        )

    @staticmethod
    def _full_triangle(fitted):
        """Return a fitted estimator's full triangle, including pipelines."""
        if hasattr(fitted, "full_triangle_"):
            return fitted.full_triangle_
        if hasattr(fitted, "named_steps"):
            for step in reversed(fitted.named_steps.values()):
                if hasattr(step, "full_triangle_"):
                    return step.full_triangle_
        raise ValueError("estimator must expose full_triangle_ after fitting.")

    @staticmethod
    def _total(triangle) -> float:
        """Sum finite values in a Triangle without changing its backend."""
        return float(np.nansum(triangle.set_backend("numpy").values))

    def _prior_valuation(self, X, current_valuation: pd.Timestamp) -> pd.Timestamp:
        """Select and validate the prior valuation period."""
        available = X.valuation[X.valuation <= current_valuation]
        available = pd.DatetimeIndex(available.drop_duplicates().sort_values())
        eligible = available[available < current_valuation]
        if len(eligible) == 0:
            raise ValueError("At least two observed valuation periods are required.")
        if self.prior_valuation_period is None:
            return eligible[-1]
        requested = pd.Period(self.prior_valuation_period, freq=X.development_grain)
        periods = pd.PeriodIndex(eligible, freq=X.development_grain)
        if requested not in periods:
            raise ValueError(
                "prior_valuation_period must be observed and before the current valuation."
            )
        return eligible[periods.get_loc(requested)]

    def fit(self, X, y=None, sample_weight=None):
        """Calculate reserve movement from prior to current valuation."""
        if not isinstance(X, Triangle):
            raise TypeError("X must be a chainladder Triangle.")
        current_valuation = X.valuation_date
        prior_valuation = self._prior_valuation(X, current_valuation)
        estimator = Chainladder() if self.estimator is None else self.estimator
        prior_data = X[X.valuation <= prior_valuation]
        prior_model = clone(estimator)
        current_model = clone(estimator)
        if sample_weight is None:
            prior_model.fit(prior_data)
            current_model.fit(X)
        else:
            prior_weight = sample_weight[sample_weight.valuation <= prior_valuation]
            prior_model.fit(prior_data, sample_weight=prior_weight)
            current_model.fit(X, sample_weight=sample_weight)

        expected = self._to_long(
            self._full_triangle(prior_model).cum_to_incr(), "expected_payment"
        )
        actual = self._to_long(X.cum_to_incr(), "actual_payment")
        valuation = pd.DataFrame(
            {
                "origin": np.repeat(X.odims, len(X.ddims)),
                "development": np.tile(X.ddims, len(X.odims)),
                "valuation": X.valuation,
            }
        )
        keys = X.key_labels + ["origin", "development", "column"]
        actual = actual.merge(valuation, on=["origin", "development"], how="left")
        actual = actual.loc[
            (actual["valuation"] > prior_valuation)
            & (actual["valuation"] <= current_valuation)
        ]
        detail = actual.merge(expected, on=keys, how="inner")
        detail = detail.loc[
            np.isfinite(detail["actual_payment"])
            & np.isfinite(detail["expected_payment"])
        ].copy()
        if detail.empty:
            raise ValueError("No projected payments align with the selected valuation period.")
        detail["payment_variance"] = (
            detail["actual_payment"] - detail["expected_payment"]
        )

        opening_ibnr = self._total(prior_model.ibnr_)
        closing_ibnr = self._total(current_model.ibnr_)
        expected_payment = detail["expected_payment"].sum()
        actual_payment = detail["actual_payment"].sum()
        expected_closing = opening_ibnr - expected_payment
        self.prior_model_ = prior_model
        self.current_model_ = current_model
        self.prior_valuation_ = prior_valuation
        self.current_valuation_ = current_valuation
        self.detail_ = detail
        self.summary_ = pd.DataFrame(
            {
                "prior_valuation": [prior_valuation],
                "current_valuation": [current_valuation],
                "opening_ibnr": [opening_ibnr],
                "expected_payment": [expected_payment],
                "actual_payment": [actual_payment],
                "payment_variance": [actual_payment - expected_payment],
                "expected_closing_ibnr": [expected_closing],
                "actual_closing_ibnr": [closing_ibnr],
                "estimate_change": [closing_ibnr - expected_closing],
            }
        )
        return self
