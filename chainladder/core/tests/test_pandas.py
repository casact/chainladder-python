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
# Covered by test_triangle.test_add_df_passthru.
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
    """
    Define pandas-style methods directly on the classes, not attached at runtime.

    This lets static type checkers such as Pyright/Pylance and ty see them.
    """
    assert (
        set(AGGREGATIONS + ["diff"] + PASSTHRU_METHODS)
        <= TrianglePandas.__dict__.keys()
    )
    assert set(AGGREGATIONS) <= TriangleGroupBy.__dict__.keys()


@pytest.mark.parametrize("method", AGGREGATIONS + ["diff"])
def test_triangle_aggregation_dispatch(raa, monkeypatch, method):
    """
    Delegate each Triangle method to ``_triangle_aggregate`` unchanged.

    Aggregations pass the ``nan``-prefixed NumPy function name (e.g. ``nansum``);
    ``diff`` passes ``diff``. Arguments and the return value pass through as is.
    """
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
    """
    Delegate each TriangleGroupBy aggregation to ``_groupby_aggregate``.

    The method name, arguments and return value pass through unchanged.
    """
    aggregate = Mock()
    monkeypatch.setattr(triangle_pandas, "_groupby_aggregate", aggregate)
    grouped = clrd.groupby("LOB")

    result = getattr(grouped, method)("arg", auto_sparse=False)

    aggregate.assert_called_once_with(grouped, method, "arg", auto_sparse=False)
    assert result is aggregate.return_value


@pytest.mark.parametrize("method", PASSTHRU_METHODS_NOT_COVERED_BY_TEST_TRIANGLE)
def test_dataframe_passthru_dispatch(raa, monkeypatch, method):
    """
    Call the same-named DataFrame method on ``to_frame()``.

    Arguments and the return value pass through unchanged. Only methods not
    already covered by ``test_triangle.test_add_df_passthru`` are checked.
    """
    frame = raa.to_frame()
    monkeypatch.setattr(raa, "to_frame", lambda: frame)
    passthru = Mock()
    monkeypatch.setattr(pd.DataFrame, method, passthru)

    result = getattr(TrianglePandas, method)(raa, "arg", option="value")

    passthru.assert_called_once_with(frame, "arg", option="value")
    assert result is passthru.return_value


@pytest.mark.parametrize("method", [name for name in AGGREGATIONS if name != "cumsum"])
@pytest.mark.parametrize("axis", ["index", "columns", "origin", "development"])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_triangle_aggregation_values(clrd, method, axis):
    """
    Match the NumPy ``nan*`` reduction along each axis, with zeros as NaN.

    The result keeps the input's backend and the ``keepdims`` shape, no
    RuntimeWarning is raised, and the input triangle is not modified.
    """
    if clrd.array_backend == "sparse" and method in {"var", "std"}:
        pytest.skip("The sparse backend does not implement nanvar or nanstd")
    triangle = clrd.iloc[11:13, :2, :5, :5]
    before = triangle.copy()
    axis_number = triangle._get_axis(axis)
    kwargs = {"q": 0.5} if method == "quantile" else {}
    expected = getattr(np, "nan" + method)(
        triangle.set_backend("numpy").values,
        axis=axis_number,
        keepdims=True,
        **kwargs,
    )
    expected[expected == 0] = np.nan

    result = getattr(triangle, method)(
        axis=axis, keepdims=True, auto_sparse=False, **kwargs
    )

    assert result.shape == expected.shape
    assert result.array_backend == triangle.array_backend
    np.testing.assert_allclose(
        result.set_backend("numpy").values, expected, equal_nan=True
    )
    assert triangle.equals(before)


@pytest.mark.parametrize("axis", range(4))
def test_aggregation_default_axis_first_non_singleton(clrd, axis):
    """With ``axis=None``, ``sum`` reduces the first axis of length > 1."""
    indexer = [slice(11, 13), slice(0, 2), slice(0, 5), slice(0, 5)]
    for preceding_axis in range(axis):
        start = indexer[preceding_axis].start
        indexer[preceding_axis] = slice(start, start + 1)
    triangle = clrd.iloc[tuple(indexer)]

    result = triangle.sum(axis=None, keepdims=True, auto_sparse=False)
    expected = triangle.sum(axis=axis, keepdims=True, auto_sparse=False)

    assert result.equals(expected)
    assert result.shape[axis] == 1


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_aggregation_default_axis_singleton_result(raa: Triangle, backend, keepdims):
    """
    With ``axis=None`` and all axes of length 1, ``sum`` returns the single value.

    It is a NumPy scalar, or with ``keepdims=True`` a (1, 1, 1, 1) triangle
    keeping the index. Checked on the numpy and sparse backends.
    """
    triangle = raa.iloc[:, :, :1, :1].set_backend(backend)
    result = triangle.sum(axis=None, keepdims=keepdims)

    if keepdims:
        assert result.shape == (1, 1, 1, 1)
        assert result.index.equals(triangle.index)
        assert result.values[0, 0, 0, 0] == triangle.values[0, 0, 0, 0]
    else:
        assert isinstance(result, np.floating)
        assert result == triangle.values[0, 0, 0, 0]
