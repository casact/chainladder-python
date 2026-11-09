from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from chainladder.core import pandas as triangle_pandas
from chainladder.core.pandas import TriangleGroupBy, TrianglePandas

if TYPE_CHECKING:
    from chainladder import Triangle


AGGREGATIONS = [
    "sum",
    "mean",
    "median",
    "max",
    "min",
    "prod",
    "var",
    "std",
    "cumsum",
    "quantile",
]
PASSTHRU_METHODS = [
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_json",
    "to_html",
    "to_dict",
    "unstack",
    "pivot",
    "drop_duplicates",
    "describe",
    "melt",
    "pct_change",
]
PASSTHRU_METHODS_COVERED_BY_TEST_TRIANGLE = {
    "to_csv",
    "to_html",
    "describe",
    "drop_duplicates",
    "melt",
    "unstack",
}
PASSTHRU_METHODS_NOT_COVERED_BY_TEST_TRIANGLE = [
    method
    for method in PASSTHRU_METHODS
    if method not in PASSTHRU_METHODS_COVERED_BY_TEST_TRIANGLE
]


def test_pandas_methods_are_defined_on_classes():
    assert (
        set(AGGREGATIONS + ["diff"] + PASSTHRU_METHODS)
        <= TrianglePandas.__dict__.keys()
    )
    assert set(AGGREGATIONS) <= TriangleGroupBy.__dict__.keys()


@pytest.mark.parametrize("method", ["cumsum", "diff"])
def test_triangle_aggregation_dispatch(raa, monkeypatch, method):
    aggregate = Mock()
    monkeypatch.setattr(triangle_pandas, "_triangle_aggregate", aggregate)

    result = getattr(raa, method)("origin", "arg", keepdims=True, auto_sparse=False)

    backend_method = method if method == "diff" else "nan" + method
    aggregate.assert_called_once_with(
        raa, backend_method, "origin", "arg", keepdims=True, auto_sparse=False
    )
    assert result is aggregate.return_value


@pytest.mark.parametrize("method", AGGREGATIONS)
def test_groupby_aggregation_dispatch(clrd, monkeypatch, method):
    aggregate = Mock()
    monkeypatch.setattr(triangle_pandas, "_groupby_aggregate", aggregate)
    grouped = clrd.groupby("LOB")

    result = getattr(grouped, method)("arg", auto_sparse=False)

    aggregate.assert_called_once_with(grouped, method, "arg", auto_sparse=False)
    assert result is aggregate.return_value


@pytest.mark.parametrize("method", PASSTHRU_METHODS_NOT_COVERED_BY_TEST_TRIANGLE)
def test_dataframe_passthru_dispatch(raa, monkeypatch, method):
    frame = raa.to_frame()
    monkeypatch.setattr(raa, "to_frame", lambda: frame)
    passthru = Mock()
    monkeypatch.setattr(pd.DataFrame, method, passthru)

    result = getattr(TrianglePandas, method)(raa, "arg", option="value")

    passthru.assert_called_once_with(frame, "arg", option="value")
    assert result is passthru.return_value


@pytest.mark.parametrize("method", [name for name in AGGREGATIONS if name != "cumsum"])
@pytest.mark.parametrize("axis", ["index", "columns", "origin", "development"])
def test_triangle_aggregation_values(clrd, method, axis):
    triangle = clrd.iloc[:3, :2].set_backend("numpy")
    before = triangle.copy()
    axis_number = triangle._get_axis(axis)
    kwargs = {"q": 0.5} if method == "quantile" else {}
    with np.errstate(invalid="ignore"):
        expected = getattr(np, "nan" + method)(
            triangle.values, axis=axis_number, keepdims=True, **kwargs
        )
    expected[expected == 0] = np.nan

    result = getattr(triangle, method)(
        axis=axis, keepdims=True, auto_sparse=False, **kwargs
    )

    assert result.shape == expected.shape
    np.testing.assert_allclose(result.values, expected, equal_nan=True)
    assert triangle.equals(before)


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_aggregation_singleton_result(raa: Triangle, backend, keepdims):
    triangle = raa.iloc[:, :, :1, :1].set_backend(backend)
    result = triangle.sum(keepdims=keepdims)

    if keepdims:
        assert result.shape == (1, 1, 1, 1)
        assert result.index.equals(triangle.index)
        assert result.values[0, 0, 0, 0] == triangle.values[0, 0, 0, 0]
    else:
        assert isinstance(result, np.floating)
        assert result == triangle.values[0, 0, 0, 0]
