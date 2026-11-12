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
        set(AGGREGATIONS + ["cumsum", "diff"] + PASSTHRU_METHODS)
        <= TrianglePandas.__dict__.keys()
    )
    assert set(AGGREGATIONS + ["cumsum"]) <= TriangleGroupBy.__dict__.keys()


@pytest.mark.parametrize("method", AGGREGATIONS)
def test_triangle_aggregation_dispatch(raa, monkeypatch, method):
    """
    Delegate each Triangle method to ``_triangle_aggregate`` unchanged.

    Aggregations pass the ``nan``-prefixed NumPy function name (e.g. ``nansum``).
    Arguments and the return value pass through as is.
    """
    aggregate = Mock()
    monkeypatch.setattr(triangle_pandas, "_triangle_aggregate", aggregate)

    result = getattr(raa, method)("origin", "arg", keepdims=True, auto_sparse=False)

    backend_method = "nan" + method
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


@pytest.mark.parametrize("method", AGGREGATIONS)
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
    triangle = clrd.iloc[13:15, :2, :5, :5]
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


def _nan_cumsum(values: np.ndarray, axis: int) -> np.ndarray:
    """Cumulative sum skipping NaN that leaves NaN cells as NaN, like pandas."""
    result = np.where(np.isnan(values), np.nan, np.nancumsum(values, axis=axis))
    result[result == 0] = np.nan
    return result


def _nan_diff(values: np.ndarray, axis: int) -> np.ndarray:
    """First difference keeping the shape with NaN in the first slot, like pandas."""
    result = np.full_like(values, np.nan)
    head = [slice(None)] * values.ndim
    tail = [slice(None)] * values.ndim
    head[axis] = slice(1, None)
    tail[axis] = slice(None, -1)
    result[tuple(head)] = values[tuple(head)] - values[tuple(tail)]
    result[result == 0] = np.nan
    return result


def _assert_same_labels(result: Triangle, triangle: Triangle) -> None:
    assert result.index.equals(triangle.index)
    assert result.columns.equals(triangle.columns)
    result_origin, expected_origin = result.origin, triangle.origin
    if isinstance(expected_origin, pd.Series):
        assert isinstance(result_origin, pd.Series)
        pd.testing.assert_series_equal(result_origin, expected_origin)
    else:
        assert isinstance(result_origin, pd.PeriodIndex)
        pd.testing.assert_index_equal(result_origin, expected_origin)
    assert result.development.equals(triangle.development)


@pytest.mark.parametrize(
    ("method", "expected_func"), [("cumsum", _nan_cumsum), ("diff", _nan_diff)]
)
@pytest.mark.parametrize("axis", ["index", "columns", "origin", "development"])
def test_triangle_shape_preserving_values(clrd, method, expected_func, axis):
    """
    Apply ``cumsum`` and ``diff`` along each axis without collapsing it.

    The result is a Triangle with the input's shape, labels and backend; empty
    cells stay NaN and the input triangle is not modified.
    """
    triangle = clrd.iloc[13:16, :2, -5:, :5]
    before = triangle.copy()
    values = triangle.set_backend("numpy").values
    assert np.isnan(values).any()
    expected = expected_func(values, triangle._get_axis(axis))

    result = getattr(triangle, method)(axis=axis, auto_sparse=False)

    assert result.shape == triangle.shape
    assert result.array_backend == triangle.array_backend
    _assert_same_labels(result, triangle)
    np.testing.assert_allclose(
        result.set_backend("numpy").values, expected, equal_nan=True
    )
    assert triangle.equals(before)


def test_cumsum_development_inverts_cum_to_incr(raa):
    """Cumulating incremental amounts along development restores the triangle."""
    result = raa.cum_to_incr().cumsum(axis="development")

    assert result.shape == raa.shape
    np.testing.assert_allclose(
        result.set_backend("numpy").values,
        raa.set_backend("numpy").values,
        equal_nan=True,
    )


def test_diff_development_matches_cum_to_incr(raa):
    """Differencing along development gives the increments after the first age."""
    result = raa.diff(axis="development").set_backend("numpy").values
    incremental = raa.cum_to_incr().set_backend("numpy").values

    assert np.isnan(result[..., 0]).all()
    np.testing.assert_allclose(result[..., 1:], incremental[..., 1:], equal_nan=True)


@pytest.mark.parametrize("method", ["cumsum", "diff"])
@pytest.mark.parametrize("axis", range(4))
def test_shape_preserving_default_axis_first_non_singleton(clrd, method, axis):
    """With ``axis=None``, use the first non-singleton axis, like the reductions."""
    indexer = [slice(11, 13), slice(0, 2), slice(0, 5), slice(0, 5)]
    for preceding_axis in range(axis):
        start = indexer[preceding_axis].start
        indexer[preceding_axis] = slice(start, start + 1)
    triangle = clrd.iloc[tuple(indexer)]
    result = getattr(triangle, method)(auto_sparse=False)
    expected = getattr(triangle, method)(axis=axis, auto_sparse=False)

    assert result.equals(expected)


@pytest.mark.parametrize("method", ["cumsum", "diff"])
def test_shape_preserving_singleton(raa, method):
    """
    A (1, 1, 1, 1) Triangle stays a Triangle instead of becoming a scalar.

    ``cumsum`` keeps its value and ``diff`` is NaN.
    """
    triangle = raa.iloc[:, :, :1, :1]
    result = getattr(triangle, method)(auto_sparse=False)

    assert result.shape == triangle.shape
    assert result.array_backend == triangle.array_backend
    _assert_same_labels(result, triangle)
    values = result.set_backend("numpy").values
    if method == "diff":
        assert np.isnan(values).all()
    else:
        np.testing.assert_allclose(values, triangle.set_backend("numpy").values)


@pytest.mark.parametrize("method", ["cumsum", "diff"])
def test_shape_preserving_negative_axis(raa, method):
    """Negative axis numbers count back from the development axis."""
    result = getattr(raa, method)(axis=-1, auto_sparse=False)
    expected = getattr(raa, method)(axis="development", auto_sparse=False)
    assert result.equals(expected)


@pytest.mark.parametrize("method", ["cumsum", "diff"])
def test_shape_preserving_empty_triangle(raa, method):
    """A Triangle with no index rows is returned empty, on the same backend."""
    triangle = raa.iloc[:0, :, :, :]

    result = getattr(triangle, method)(auto_sparse=False)

    assert result.shape == triangle.shape == (0, 1, 10, 10)
    assert result.array_backend == triangle.array_backend


@pytest.mark.parametrize("method", ["cumsum", "diff"])
def test_shape_preserving_auto_sparse(raa, method):
    """
    By default a small sparse result is converted to numpy, like the reductions.

    ``auto_sparse=False`` keeps it sparse, with the same values.
    """
    triangle = raa.set_backend("sparse")

    default = getattr(triangle, method)(axis="development")
    kept = getattr(triangle, method)(axis="development", auto_sparse=False)

    assert default.array_backend == "numpy"
    assert kept.array_backend == "sparse"
    assert default.equals(kept)


@pytest.mark.parametrize("skipna", [True, False])
def test_cumsum_internal_missing_cells(raa, skipna):
    """
    Continue the total after a missing cell only with ``skipna=True``.

    The missing cell stays missing either way, and a zero total is stored as NaN.
    """
    triangle = raa.iloc[:, :, :1, :5]
    triangle.values = triangle.get_array_module().array(
        np.array([2.0, np.nan, 3.0, -5.0, 4.0]).reshape(triangle.shape)
    )
    before = triangle.copy()
    result = triangle.cumsum(axis="development", skipna=skipna, auto_sparse=False)
    expected = [2.0, np.nan, 5.0, np.nan, 4.0] if skipna else [2.0] + [np.nan] * 4

    np.testing.assert_allclose(
        result.set_backend("numpy").values.ravel(), expected, equal_nan=True
    )
    assert triangle.equals(before)


@pytest.mark.parametrize("axis", range(4))
@pytest.mark.parametrize("periods", [0, 2, -1, -2, 10, 12])
def test_diff_periods(clrd, axis, periods):
    """Match pandas ``diff`` for each axis and ``periods``, with zeros as NaN."""
    triangle = clrd.iloc[13:16, :2, -5:, :5]
    values = np.moveaxis(triangle.set_backend("numpy").values, axis, 0)
    expected = (
        pd
        .DataFrame(values.reshape(values.shape[0], -1))
        .diff(periods=periods)
        .to_numpy()
        .reshape(values.shape)
    )
    expected = np.moveaxis(expected, 0, axis)
    expected[expected == 0] = np.nan

    result = triangle.diff(axis=axis, periods=periods, auto_sparse=False)

    assert result.shape == triangle.shape
    assert result.array_backend == triangle.array_backend
    _assert_same_labels(result, triangle)
    np.testing.assert_allclose(
        result.set_backend("numpy").values, expected, equal_nan=True
    )


def test_diff_periods_must_be_integer(raa):
    """A non-integer ``periods`` raises instead of being truncated."""
    with pytest.raises(TypeError):
        raa.diff(axis="development", periods=1.5)


def test_sparse_transforms_do_not_densify(raa, monkeypatch):
    """``cumsum`` and ``diff`` on a sparse Triangle never call ``todense``."""
    triangle = raa.set_backend("sparse")

    def reject_densification(*_args, **_kwargs):
        pytest.fail("Transform unexpectedly densified its sparse input")

    with monkeypatch.context() as patch:
        patch.setattr(type(triangle.values), "todense", reject_densification)
        cumulative = triangle.cumsum(axis="development", auto_sparse=False)
        difference = triangle.diff(axis="development", auto_sparse=False)

    assert cumulative.array_backend == difference.array_backend == "sparse"


def test_groupby_cumsum(clrd):
    """
    Cumulate within each group and keep every row in the original order.

    Like pandas ``groupby(...).cumsum()``, the result has the input's shape and
    index, and each row is the running total of the earlier rows in its group.
    """
    triangle = clrd.iloc[:12, :2, -4:, :4]
    before = triangle.copy()
    assert not triangle.index["LOB"].is_monotonic_increasing
    values = triangle.set_backend("numpy").values
    expected = np.empty_like(values)
    for _, rows in triangle.index.groupby("LOB").indices.items():
        expected[rows] = _nan_cumsum(values[rows], axis=0)

    result = triangle.groupby("LOB").cumsum(auto_sparse=False)

    assert result.shape == triangle.shape
    assert result.array_backend == triangle.array_backend
    _assert_same_labels(result, triangle)
    np.testing.assert_allclose(
        result.set_backend("numpy").values, expected, equal_nan=True
    )
    assert triangle.equals(before)


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_groupby_cumsum_other_axes(clrd, axis):
    """Group along the columns, origin and development axes as well."""
    triangle = clrd.iloc[13:16, :4, -4:, :4]
    groups = ["a", "b", "a", "b"]
    values = triangle.set_backend("numpy").values
    expected = np.empty_like(values)
    for positions in ([0, 2], [1, 3]):
        indexer = [slice(None)] * axis + [positions]
        expected[tuple(indexer)] = _nan_cumsum(values[tuple(indexer)], axis)

    result = triangle.groupby(groups, axis=axis).cumsum(auto_sparse=False)

    assert result.shape == triangle.shape
    assert result.array_backend == triangle.array_backend
    _assert_same_labels(result, triangle)
    np.testing.assert_allclose(
        result.set_backend("numpy").values, expected, equal_nan=True
    )


@pytest.mark.parametrize("all_missing", [True, False])
def test_groupby_cumsum_missing_keys(clrd, all_missing):
    """Labels with a missing group key are NaN; the other groups still cumulate."""
    triangle = clrd.iloc[:12, :2, -4:, :4]
    labels = triangle.index.copy()
    labels.loc[labels.index if all_missing else [0], "LOB"] = None
    triangle.index = labels
    values = triangle.set_backend("numpy").values
    expected = np.full_like(values, np.nan)
    for rows in labels.groupby("LOB").indices.values():
        expected[rows] = _nan_cumsum(values[rows], axis=0)

    result = triangle.groupby("LOB").cumsum(auto_sparse=False)

    assert result.shape == triangle.shape
    assert result.array_backend == triangle.array_backend
    _assert_same_labels(result, triangle)
    np.testing.assert_allclose(
        result.set_backend("numpy").values, expected, equal_nan=True
    )


def test_groupby_cumsum_auto_sparse(clrd):
    """
    By default a small sparse result is converted to numpy, like the reductions.

    ``auto_sparse=False`` keeps it sparse, with the same values.
    """
    triangle = clrd.iloc[:12, :2, -4:, :4].set_backend("sparse")

    default = triangle.groupby("LOB").cumsum()
    kept = triangle.groupby("LOB").cumsum(auto_sparse=False)

    assert default.array_backend == "numpy"
    assert kept.array_backend == "sparse"
    assert default.equals(kept)


@pytest.mark.parametrize("skipna", [True, False])
def test_groupby_cumsum_skipna(raa, skipna):
    """``skipna`` applies within each group and does not carry across groups."""
    triangle = raa.iloc[:, :, :1, :4]
    triangle.values = triangle.get_array_module().array(
        np.array([2.0, np.nan, 3.0, 4.0]).reshape(triangle.shape)
    )
    expected = [2.0, np.nan, 5.0, 4.0] if skipna else [2.0, np.nan, np.nan, 4.0]

    result = triangle.groupby(["a", "a", "a", "b"], axis=3).cumsum(
        skipna=skipna, auto_sparse=False
    )

    np.testing.assert_allclose(
        result.set_backend("numpy").values.ravel(), expected, equal_nan=True
    )
