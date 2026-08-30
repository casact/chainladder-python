# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Financial measurement helpers for projected reserving cash flows."""
from __future__ import annotations

from numbers import Real
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from chainladder.core.io import EstimatorIO
from chainladder.core.triangle import Triangle


class DiscountedReserve(BaseEstimator, EstimatorIO):
    """Discount future incremental payments from a projected triangle.

    ``X`` should contain projected cumulative values, such as a reserving
    method's ``full_triangle_``. Only incremental payments after
    ``valuation_date`` are included in the present value.

    Parameters
    ----------
    annual_rates : float or dict
        An annual effective rate, or a mapping from payment period (for
        example, ``{"2025": 0.03, "2026": 0.032}``) to annual effective rate.
    valuation_date : str or Timestamp
        Measurement date. This is explicit because a projected triangle's own
        valuation date may be its ultimate date rather than the reporting date.

    Attributes
    ----------
    cashflows_ : DataFrame
        Future incremental payments with payment date, annual rate, discount
        factor, and present value.
    summary_ : DataFrame
        Undiscounted and discounted totals by triangle key and value column.
    """

    def __init__(self, annual_rates, valuation_date):
        self.annual_rates = annual_rates
        self.valuation_date = valuation_date

    @staticmethod
    def _to_long(triangle) -> pd.DataFrame:
        """Convert a Triangle to payment rows and attach calendar valuation."""
        frame = triangle.to_frame(keepdims=True).reset_index()
        identifiers = triangle.key_labels + ["origin", "development"]
        if "values" in frame:
            frame = frame.loc[:, identifiers + ["values"]]
            frame = frame.rename(columns={"values": "cashflow"})
            frame["column"] = triangle.columns[0]
        else:
            frame = frame.melt(
                id_vars=identifiers,
                value_vars=triangle.columns,
                var_name="column",
                value_name="cashflow",
            )
        valuation = pd.DataFrame(
            {
                "origin": np.repeat(triangle.odims, len(triangle.ddims)),
                "development": np.tile(triangle.ddims, len(triangle.odims)),
                "payment_date": triangle.valuation,
            }
        )
        return frame.merge(valuation, on=["origin", "development"], how="left")

    def _rates(self, cashflows: pd.DataFrame, development_grain: str) -> pd.Series:
        """Return the effective annual rate for each payment row."""
        if isinstance(self.annual_rates, Real):
            rates = pd.Series(float(self.annual_rates), index=cashflows.index)
        elif isinstance(self.annual_rates, dict):
            curve = pd.Series(self.annual_rates, dtype=float)
            curve.index = curve.index.astype(str)
            periods = pd.PeriodIndex(
                cashflows["payment_date"], freq=development_grain
            ).astype(str)
            rates = pd.Series(periods, index=cashflows.index).map(curve)
            if rates.isna().any():
                missing = ", ".join(sorted(pd.unique(periods[rates.isna()])))
                raise ValueError(f"annual_rates is missing payment periods: {missing}.")
        else:
            raise TypeError("annual_rates must be a float or a dictionary of period rates.")
        if not np.isfinite(rates).all() or (rates <= -1).any():
            raise ValueError("annual_rates must be finite and greater than -1.")
        return rates.astype(float)

    def fit(self, X, y=None):
        """Calculate present values for future projected payments."""
        if not isinstance(X, Triangle):
            raise TypeError("X must be a projected chainladder Triangle.")
        valuation_date = pd.Timestamp(self.valuation_date)
        if pd.isna(valuation_date):
            raise ValueError("valuation_date must be a valid date.")
        incremental = X.cum_to_incr() if X.is_cumulative else X.copy()
        cashflows = self._to_long(incremental)
        cashflows = cashflows.loc[
            (cashflows["payment_date"] > valuation_date)
            & np.isfinite(cashflows["cashflow"])
        ].copy()
        if cashflows.empty:
            raise ValueError("No finite future payments were found after valuation_date.")
        cashflows["annual_rate"] = self._rates(
            cashflows, incremental.development_grain
        )
        cashflows["years_to_payment"] = (
            (cashflows["payment_date"] - valuation_date).dt.total_seconds()
            / (365.25 * 24 * 60 * 60)
        )
        cashflows["discount_factor"] = (1 + cashflows["annual_rate"]) ** (
            -cashflows["years_to_payment"]
        )
        cashflows["present_value"] = (
            cashflows["cashflow"] * cashflows["discount_factor"]
        )
        self.valuation_date_ = valuation_date
        self.cashflows_ = cashflows
        group_columns = X.key_labels + ["column"]
        self.summary_ = (
            cashflows.groupby(group_columns, as_index=False)
            .agg(
                undiscounted=("cashflow", "sum"),
                present_value=("present_value", "sum"),
            )
        )
        self.summary_["discount_effect"] = (
            self.summary_["undiscounted"] - self.summary_["present_value"]
        )
        return self


class RiskAdjustment(BaseEstimator, EstimatorIO):
    """Add an explicit risk adjustment to a best-estimate reserve summary.

    This helper is intentionally transparent: ``method="margin"`` adds a
    fixed percentage of the best estimate, ``method="confidence_level"``
    applies a normal-distribution percentile to a supplied standard error, and
    ``method="percentile"`` uses simulated reserve outcomes. It does not, by
    itself, establish IFRS 17 compliance or select a risk appetite.

    Parameters
    ----------
    method : {"margin", "confidence_level", "percentile"}, default="margin"
        Calculation approach for the adjustment.
    margin : float, default=0.0
        Proportion of best estimate to add when ``method="margin"``.
    confidence_level : float, default=0.75
        Confidence level used by the normal approximation.
    standard_error : float, Series, or str, optional
        Standard error for ``method="confidence_level"``. A string selects a
        column from ``X``; a scalar is applied to every row.
    simulations : array-like, optional
        Simulated reserve outcomes for ``method="percentile"``. A one-dimensional
        array applies to a one-row summary; a two-dimensional array supplies one
        simulated distribution per row.
    value_column : str, default="present_value"
        Column containing the best-estimate reserve.
    """

    def __init__(
        self,
        method: str = "margin",
        margin: float = 0.0,
        confidence_level: float = 0.75,
        standard_error=None,
        simulations=None,
        value_column: str = "present_value",
    ):
        self.method = method
        self.margin = margin
        self.confidence_level = confidence_level
        self.standard_error = standard_error
        self.simulations = simulations
        self.value_column = value_column

    def _standard_error(self, data: pd.DataFrame) -> pd.Series:
        """Resolve the supplied standard error to the summary index."""
        if isinstance(self.standard_error, str):
            if self.standard_error not in data:
                raise ValueError("standard_error column was not found in X.")
            result = data[self.standard_error]
        elif isinstance(self.standard_error, Real):
            result = pd.Series(float(self.standard_error), index=data.index)
        elif isinstance(self.standard_error, pd.Series):
            result = self.standard_error.reindex(data.index)
        else:
            raise ValueError(
                "standard_error must be supplied for confidence_level method."
            )
        if result.isna().any() or not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("standard_error must be finite and non-negative.")
        return result.astype(float)

    def _percentile_adjustment(self, best_estimate: pd.Series) -> pd.Series:
        """Calculate adjustment from simulated reserve outcomes."""
        values = np.asarray(self.simulations, dtype=float)
        if values.ndim == 1:
            if len(best_estimate) != 1:
                raise ValueError("One-dimensional simulations require a one-row summary.")
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] != len(best_estimate):
            raise ValueError("simulations must provide one distribution for each summary row.")
        if values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError("simulations must contain finite reserve outcomes.")
        percentile = np.quantile(values, self.confidence_level, axis=1)
        return pd.Series(percentile, index=best_estimate.index) - best_estimate

    def fit(self, X, y=None):
        """Calculate the risk adjustment for a reserve summary DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame such as DiscountedReserve.summary_.")
        if self.value_column not in X:
            raise ValueError(f"X must contain a '{self.value_column}' column.")
        data = X.copy()
        best_estimate = data[self.value_column].astype(float)
        if self.method == "margin":
            if not isinstance(self.margin, Real) or self.margin < 0:
                raise ValueError("margin must be a non-negative number.")
            adjustment = best_estimate * float(self.margin)
        elif self.method == "confidence_level":
            if not 0.5 < self.confidence_level < 1:
                raise ValueError("confidence_level must be greater than 0.5 and less than 1.")
            z_score = NormalDist().inv_cdf(self.confidence_level)
            adjustment = self._standard_error(data) * z_score
        elif self.method == "percentile":
            if not 0.5 < self.confidence_level < 1:
                raise ValueError("confidence_level must be greater than 0.5 and less than 1.")
            adjustment = self._percentile_adjustment(best_estimate)
        else:
            raise ValueError(
                "method must be 'margin', 'confidence_level', or 'percentile'."
            )
        self.summary_ = data.copy()
        self.summary_["best_estimate"] = best_estimate
        self.summary_["risk_adjustment"] = adjustment
        self.summary_["reserve_including_risk_adjustment"] = (
            best_estimate + adjustment
        )
        return self
