# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from chainladder.core.io import EstimatorIO
from chainladder.methods import Chainladder


class Backtest(BaseEstimator, EstimatorIO):
    """Multi-period diagonal backtesting for reserving methods.

    Each backtest fits a fresh copy of ``estimator`` using data available at a
    historical valuation date. The projected cumulative values for origins
    present at that date are compared with the observed valuation diagonal at
    the selected horizon. This makes the class useful for short-tailed business
    where one or a few reporting periods are practical validation horizons.

    Parameters
    ----------
    estimator : estimator, optional
        Reserving estimator with a ``full_triangle_`` attribute after fitting.
        Defaults to :class:`~chainladder.Chainladder`.
    valuation_periods : str or list of str, optional
        Historical valuation periods to backtest.  For example, annual data
        accepts ``["2021", "2022"]`` and quarterly data accepts
        ``["2022Q3", "2022Q4"]``.  Each period must have a following observed
        valuation period.
    n_periods : int, default=3
        Number of most recent eligible valuation periods to backtest when
        ``valuation_periods`` is not supplied.
    horizon : int, default=1
        Number of reporting periods between the training valuation and the
        observed valuation used for comparison.

    Attributes
    ----------
    results_ : DataFrame
        Cell-level forecast results with actual, predicted, and error fields.
    summary_ : DataFrame
        Aggregated backtest results by valuation period.
    valuation_periods_ : DatetimeIndex
        Historical valuation dates used for the backtest.

    Examples
    --------
    Backtest the two most recent annual valuation periods.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        result = cl.Backtest(n_periods=2).fit(cl.load_sample("raa"))
        print(result.summary_.loc[:, ["observations", "error"]].round(2))

    .. testoutput::

       observations    error
    0             8  3639.56
    1             9 -7129.31
    """

    def __init__(
        self,
        estimator=None,
        valuation_periods: str | list[str] | None = None,
        n_periods: int = 3,
        horizon: int = 1,
    ):
        self.estimator = estimator
        self.valuation_periods = valuation_periods
        self.n_periods = n_periods
        self.horizon = horizon

    @staticmethod
    def _to_long(triangle, value_name: str) -> pd.DataFrame:
        """Convert a Triangle to a consistent long-form DataFrame."""
        frame = triangle.to_frame(keepdims=True).reset_index()
        id_vars = triangle.key_labels + ["origin", "development"]
        value_columns = [column for column in triangle.columns if column in frame]
        return frame.melt(
            id_vars=id_vars,
            value_vars=value_columns,
            var_name="column",
            value_name=value_name,
        )

    def _get_valuation_periods(self, X) -> pd.DatetimeIndex:
        available = X.valuation[X.valuation <= X.valuation_date]
        available = pd.DatetimeIndex(available.drop_duplicates().sort_values())
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        eligible = available[:-self.horizon]
        if len(eligible) == 0:
            raise ValueError(
                "The triangle does not have enough observed valuation periods "
                "for the selected horizon."
            )

        if self.valuation_periods is None:
            if not isinstance(self.n_periods, int) or self.n_periods < 1:
                raise ValueError("n_periods must be a positive integer.")
            return eligible[-self.n_periods :]

        requested = [self.valuation_periods]
        if not isinstance(self.valuation_periods, str):
            requested = self.valuation_periods
        requested = pd.PeriodIndex(requested, freq=X.development_grain)
        eligible_periods = pd.PeriodIndex(eligible, freq=X.development_grain)
        missing = requested[~requested.isin(eligible_periods)]
        if len(missing):
            values = ", ".join(missing.astype(str))
            raise ValueError(
                "valuation_periods must be observed periods with a following "
                f"valuation period at the selected horizon. Invalid periods: {values}."
            )
        return pd.DatetimeIndex(
            [eligible[eligible_periods.get_loc(period)] for period in requested]
        )

    @staticmethod
    def _full_triangle(fitted):
        """Return a fitted estimator's full triangle, including pipeline support."""
        if hasattr(fitted, "full_triangle_"):
            return fitted.full_triangle_
        if hasattr(fitted, "named_steps"):
            for step in reversed(fitted.named_steps.values()):
                if hasattr(step, "full_triangle_"):
                    return step.full_triangle_
        raise ValueError(
            "estimator must expose full_triangle_ after fitting to run a backtest."
        )

    def fit(self, X, y=None, sample_weight=None):
        """Fit backtests at the selected historical valuation periods."""
        obj = X.incr_to_cum()
        available = obj.valuation[obj.valuation <= obj.valuation_date]
        available = pd.DatetimeIndex(available.drop_duplicates().sort_values())
        periods = self._get_valuation_periods(obj)
        estimator = Chainladder() if self.estimator is None else self.estimator

        results = []
        self.models_ = {}
        for valuation in periods:
            target_valuation = available[available.get_loc(valuation) + self.horizon]
            train = obj[obj.valuation <= valuation]
            fitted = clone(estimator)
            if sample_weight is None:
                fitted.fit(train)
            else:
                weight = sample_weight[sample_weight.valuation <= valuation]
                fitted.fit(train, sample_weight=weight)
            self.models_[valuation] = fitted

            predicted = self._to_long(self._full_triangle(fitted), "predicted")
            actual = self._to_long(
                obj[obj.valuation == target_valuation], "actual"
            )
            keys = obj.key_labels + ["origin", "development", "column"]
            result = actual.merge(predicted, on=keys, how="inner")
            result = result[np.isfinite(result["actual"]) & np.isfinite(result["predicted"])]
            if result.empty:
                raise ValueError(
                    "No projected cells aligned with the next observed valuation diagonal."
                )
            result["valuation"] = valuation
            result["target_valuation"] = target_valuation
            result["horizon"] = self.horizon
            result["error"] = result["actual"] - result["predicted"]
            result["absolute_error"] = result["error"].abs()
            result["absolute_percentage_error"] = np.where(
                result["actual"] == 0,
                np.nan,
                result["absolute_error"] / result["actual"].abs(),
            )
            results.append(result)

        self.valuation_periods_ = periods
        self.results_ = pd.concat(results, ignore_index=True)
        self.summary_ = (
            self.results_
            .groupby(["valuation", "target_valuation", "horizon"], as_index=False)
            .agg(
                observations=("actual", "size"),
                actual=("actual", "sum"),
                predicted=("predicted", "sum"),
                error=("error", "sum"),
                absolute_error=("absolute_error", "sum"),
                mean_absolute_percentage_error=("absolute_percentage_error", "mean"),
            )
        )
        self.summary_["absolute_percentage_error"] = np.where(
            self.summary_["actual"] == 0,
            np.nan,
            self.summary_["absolute_error"] / self.summary_["actual"].abs(),
        )
        return self
