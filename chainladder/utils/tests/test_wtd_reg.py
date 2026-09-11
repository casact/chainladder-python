from __future__ import annotations

import numpy as np

from chainladder.utils.weighted_regression import WeightedRegression


class TestOLS:
    """Test the OLS calculations"""

    def test_missing_data(self) -> None:
        """Check that having nan in X and/or y still results in the right OLS coefficients."""
        X_full = np.array([[[[1.0], [2.0], [3.0], [4.0], [5.0]]]])
        y_full = np.array([[[[1.0], [2.0], [3.0], [4.0], [5.0]]]])
        X_missing = np.array([[[[1.0], [np.nan], [3.0], [4.0], [5.0]]]])
        y_missing = np.array([[[[1.0], [2.0], [np.nan], [4.0], [5.0]]]])
        w = np.array([[[[1.0], [1.0], [1.0], [1.0], [1.0]]]])
        assert np.all(
            WeightedRegression()
            .fit(X_full, y_full, w, "regression")
            .slope_.flatten()
            == [1.0]
        )
        assert np.all(
            WeightedRegression()
            .fit(X_full, y_missing, w, "regression")
            .slope_.flatten()
            == [1.0]
        )
        assert np.all(
            WeightedRegression()
            .fit(X_missing, y_full, w, "regression")
            .slope_.flatten()
            == [1.0]
        )
        assert np.all(
            WeightedRegression()
            .fit(X_missing, y_missing, w, "regression")
            .slope_.flatten()
            == [1.0]
        )
