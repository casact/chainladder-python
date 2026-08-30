# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Portable reserving audit reports."""
from __future__ import annotations

from html import escape

import pandas as pd
from sklearn.base import BaseEstimator

from chainladder.core.io import EstimatorIO


class ReservingReport(BaseEstimator, EstimatorIO):
    """Compile reserving outputs and assumptions into a self-contained HTML report.

    Parameters
    ----------
    title : str, default="Reserving report"
        Heading displayed in the report.
    assumptions : dict, optional
        Named assumptions such as valuation date, method, yield curve, or risk
        adjustment approach.
    """

    def __init__(self, title: str = "Reserving report", assumptions: dict | None = None):
        self.title = title
        self.assumptions = assumptions

    def fit(self, X, y=None):
        """Compile a mapping of section names to report DataFrames."""
        if not isinstance(X, dict) or not X:
            raise ValueError("X must be a non-empty dictionary of report DataFrames.")
        if any(not isinstance(value, pd.DataFrame) for value in X.values()):
            raise TypeError("Every report section must be a pandas DataFrame.")
        self.sections_ = {str(name): frame.copy() for name, frame in X.items()}
        assumptions = self.assumptions or {}
        if not isinstance(assumptions, dict):
            raise TypeError("assumptions must be a dictionary.")
        self.assumptions_ = assumptions.copy()
        self.html_ = self.to_html()
        return self

    def to_html(self) -> str:
        """Render the fitted report as portable HTML."""
        if not hasattr(self, "sections_"):
            raise ValueError("Fit the report before rendering it.")
        assumptions = "".join(
            f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>"
            for key, value in self.assumptions_.items()
        )
        sections = "".join(
            f"<section><h2>{escape(name)}</h2>{frame.to_html(index=False, border=0)}</section>"
            for name, frame in self.sections_.items()
        )
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>"
            f"{escape(self.title)}</title><style>body{{font-family:system-ui;margin:2rem;}}"
            "table{border-collapse:collapse;}th,td{padding:.4rem;border-bottom:1px solid #ddd;}"
            "dt{font-weight:600;}dd{margin:0 0 .5rem;}</style></head><body>"
            f"<h1>{escape(self.title)}</h1><dl>{assumptions}</dl>{sections}</body></html>"
        )
