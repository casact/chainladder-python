from __future__ import annotations

import numpy as np

from chainladder.utils.weighted_regression import WeightedRegression

class TestOLS:
    """Test the OLS calculations"""

    def test_missing_data(self) -> None:
        """Check that having nan in X and/or y still results in the right OLS coefficients."""
        X_full = np.array([[[[1.], [2.], [3.], [4.], [5.]]]])
        y_full = np.array([[[[1.], [2.], [3.], [4.], [5.]]]])
        X_missing = np.array([[[[1.], [np.nan], [3.], [4.], [5.]]]])
        y_missing = np.array([[[[1.], [2.], [np.nan], [4.], [5.]]]])
        w = np.array([[[[1.], [1.], [1.], [1.], [1.]]]])
        assert np.all(
            WeightedRegression().fit(X_full, y_full, w, "regression").slope_.flatten()
            == [1.]
        )
        assert np.all(
            WeightedRegression().fit(X_full, y_missing, w, "regression").slope_.flatten()
            == [1.]
        )
        assert np.all(
            WeightedRegression().fit(X_missing, y_full, w, "regression").slope_.flatten()
            == [1.]
        )
        assert np.all(
            WeightedRegression().fit(X_missing, y_missing, w, "regression").slope_.flatten()
            == [1.]
        )
