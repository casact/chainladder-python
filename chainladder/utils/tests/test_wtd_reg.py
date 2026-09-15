from __future__ import annotations

import numpy as np
from chainladder.utils.sparse import sp
from chainladder.utils.weighted_regression import WeightedRegression


class TestOLS:
    """Test the OLS calculations"""

    def test_missing_data(self) -> None:
        """Check that having nan in X and/or y still results in the right OLS coefficients."""
        data = [
            {
                "module": np,
                "X": [
                    np.array([[[[1.0], [2.0], [3.0], [4.0], [5.0]]]]),
                    np.array([[[[1.0], [np.nan], [3.0], [4.0], [5.0]]]]),
                ],
                "y": [
                    np.array([[[[1.0], [2.0], [3.0], [4.0], [5.0]]]]),
                    np.array([[[[1.0], [2.0], [np.nan], [4.0], [5.0]]]]),
                ],
                "w": np.array([[[[1.0], [1.0], [1.0], [1.0], [1.0]]]]),
                "slope": np.array([[[[1.0]]]]),
            }
        ]
        data.append({
            "module": sp,
            "X": [sp.COO.from_numpy(i, fill_value=np.nan) for i in data[0]["X"]],
            "y": [sp.COO.from_numpy(i, fill_value=np.nan) for i in data[0]["y"]],
            "w": sp.COO.from_numpy(data[0]["w"]),
            "slope": sp.COO.from_numpy(data[0]["slope"]),
        })
        for i in data:
            for x in i["X"]:
                for y in i["y"]:
                    assert i["module"].all(
                        WeightedRegression(xp=i["module"])
                        .fit(x, y, i["w"], "regression")
                        .slope_
                        == i["slope"]
                    )
