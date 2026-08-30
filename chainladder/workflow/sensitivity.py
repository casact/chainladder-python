# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Scenario-based reserve sensitivity analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from chainladder.core.io import EstimatorIO


class SensitivityAnalysis(BaseEstimator, EstimatorIO):
    """Measure ultimate and IBNR movement across reserving scenarios.

    Parameters
    ----------
    estimator : estimator
        A reserving estimator or workflow supporting ``set_params`` and ``fit``.
    scenarios : dict
        Mapping from scenario name to parameter overrides. The first scenario is
        the base scenario used for changes unless ``base_scenario`` is supplied.
    base_scenario : str, optional
        Scenario used as the comparison baseline.

    Attributes
    ----------
    results_ : DataFrame
        Ultimate, IBNR, and changes from the base scenario for each case.
    models_ : dict
        Fitted estimator for each scenario.
    """

    def __init__(self, estimator, scenarios: dict, base_scenario: str | None = None):
        self.estimator = estimator
        self.scenarios = scenarios
        self.base_scenario = base_scenario

    @staticmethod
    def _fitted_estimator(fitted):
        """Return the object carrying reserving attributes, including pipelines."""
        if hasattr(fitted, "ultimate_") and hasattr(fitted, "ibnr_"):
            return fitted
        if hasattr(fitted, "named_steps"):
            for step in reversed(fitted.named_steps.values()):
                if hasattr(step, "ultimate_") and hasattr(step, "ibnr_"):
                    return step
        raise ValueError("estimator must expose ultimate_ and ibnr_ after fitting.")

    @staticmethod
    def _total(triangle) -> float:
        """Sum finite reserve values without changing the input backend."""
        return float(np.nansum(triangle.set_backend("numpy").values))

    def fit(self, X, y=None, sample_weight=None):
        """Fit one reserve model per scenario and compare their totals."""
        if not isinstance(self.scenarios, dict) or not self.scenarios:
            raise ValueError("scenarios must be a non-empty dictionary.")
        if self.base_scenario is not None and self.base_scenario not in self.scenarios:
            raise ValueError("base_scenario must be present in scenarios.")
        self.models_ = {}
        records = []
        for name, parameters in self.scenarios.items():
            if not isinstance(parameters, dict):
                raise TypeError("Each scenario must map to a dictionary of parameters.")
            model = clone(self.estimator).set_params(**parameters)
            if sample_weight is None:
                model.fit(X)
            else:
                model.fit(X, sample_weight=sample_weight)
            reserve_model = self._fitted_estimator(model)
            self.models_[name] = model
            records.append(
                {
                    "scenario": name,
                    "parameters": parameters,
                    "ultimate": self._total(reserve_model.ultimate_),
                    "ibnr": self._total(reserve_model.ibnr_),
                }
            )
        self.results_ = pd.DataFrame(records)
        base_name = self.base_scenario or next(iter(self.scenarios))
        self.base_scenario_ = base_name
        base = self.results_.set_index("scenario").loc[base_name]
        self.results_["ultimate_change"] = self.results_["ultimate"] - base["ultimate"]
        self.results_["ibnr_change"] = self.results_["ibnr"] - base["ibnr"]
        self.results_["ultimate_change_pct"] = np.where(
            base["ultimate"] == 0,
            np.nan,
            self.results_["ultimate_change"] / abs(base["ultimate"]),
        )
        self.results_["ibnr_change_pct"] = np.where(
            base["ibnr"] == 0,
            np.nan,
            self.results_["ibnr_change"] / abs(base["ibnr"]),
        )
        return self
