from __future__ import annotations

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_dropna(clrd):
    assert clrd.shape == clrd.dropna().shape
    a = clrd[clrd["LOB"] == "wkcomp"].iloc[-5]["CumPaidLoss"].dropna().shape
    assert a == (1, 1, 2, 2)


def test_dropna_latest_diagonal(raa: Triangle) -> None:
    """
    dropna() on a single-development-period triangle (shape[-1] == 1), where first origin period is nan.
    First origin period should be eliminated.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    t = raa.copy()

    def set_first_origin_nan(values) -> None:
        """
        Sets the values for the first origin period to nan.
        """
        values[:, :, 0, :] = np.nan

    if t.array_backend == "sparse":
        # Sparse COO arrays don't support in-place item assignment.
        with pytest.raises(TypeError, match="sparse backend"):
            set_first_origin_nan(t.values)
        return
    set_first_origin_nan(t.values)
    result = t.latest_diagonal.dropna()
    assert result.shape == (1, 1, 9, 1)
    assert result.origin.min().year == 1982


def _ffill_source_triangle():
    """Triangle from #1030 - a mix of leading, interior, and not-yet-valued NaNs."""
    df = pd.DataFrame({
        "origin": [1985, 1985, 1985, 1985, 1986, 1986, 1986, 1987, 1987, 1988],
        "development": [1985, 1986, 1987, 1988, 1986, 1987, 1988, 1987, 1988, 1988],
        "paid": [500, np.nan, 700, np.nan, np.nan, 1000, 1100, 1200, 1300, np.nan],
    })
    return cl.Triangle(
        data=df,
        origin="origin",
        development="development",
        columns="paid",
        cumulative=True,
    )


def test_ffill_development_axis() -> None:
    """
    Interior NaNs fill forward from the last valid value; a leading NaN
    (1986 at age 12) and not-yet-valued cells (1986 at 48, 1987 at 36/48)
    stay NaN - ffill never writes into a cell that hasn't been valued yet.
    """
    tri = _ffill_source_triangle()
    frame = tri.ffill().to_frame(origin_as_datetime=False)
    assert frame.loc["1985", 24] == 500.0
    assert frame.loc["1985", 48] == 700.0
    assert pd.isna(frame.loc["1986", 12])
    assert pd.isna(frame.loc["1986", 48])
    assert pd.isna(frame.loc["1987", 36])
    assert pd.isna(frame.loc["1987", 48])


def test_ffill_origin_axis() -> None:
    """Same triangle, filled down the origin axis instead."""
    tri = _ffill_source_triangle()
    frame = tri.ffill(axis="origin").to_frame(origin_as_datetime=False)
    assert frame.loc["1986", 12] == 500.0
    assert frame.loc["1987", 24] == 1300.0
    assert frame.loc["1988", 12] == 1200.0
    assert pd.isna(frame.loc["1985", 24])
    assert pd.isna(frame.loc["1988", 24])
    assert pd.isna(frame.loc["1988", 36])


def test_ffill_does_not_mutate_original() -> None:
    """ffill returns a new Triangle; the source is untouched."""
    tri = _ffill_source_triangle()
    before = tri.to_frame(origin_as_datetime=False).copy()
    tri.ffill()
    pd.testing.assert_frame_equal(
        before, tri.to_frame(origin_as_datetime=False), check_dtype=False
    )


def test_ffill_zero_input_is_missing_and_fills() -> None:
    """
    A 0 in the input becomes NaN on construction (the package treats 0 as
    missing everywhere), so ffill carries it forward like any other gap.
    """
    df = pd.DataFrame({
        "origin": [1985, 1985, 1985, 1986, 1986],
        "development": [1985, 1986, 1987, 1986, 1987],
        "paid": [500.0, 0.0, 700.0, 300.0, 400.0],
    })
    tri = cl.Triangle(
        data=df,
        origin="origin",
        development="development",
        columns="paid",
        cumulative=True,
    )
    assert pd.isna(tri.to_frame(origin_as_datetime=False).loc["1985", 24])
    frame = tri.ffill().to_frame(origin_as_datetime=False)
    assert frame.loc["1985", 24] == 500.0
    assert frame.loc["1985", 36] == 700.0


def test_ffill_invalid_axis_raises(raa: Triangle) -> None:
    """ffill() only supports the origin and development axes."""
    with pytest.raises(
        AttributeError,
        match="ffill is only supported for the origin and development axes",
    ):
        raa.ffill(axis="columns")

    with pytest.raises(
        AttributeError,
        match="ffill is only supported for the origin and development axes",
    ):
        raa.ffill(axis=0)


def test_head(clrd: Triangle) -> None:
    """
    Triangle.head(n) returns a Triangle limited to the first n rows of the index axis.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    assert clrd.head(3).shape[0] == 3
    assert list(clrd.head(3).index["LOB"]) == ["othliab", "ppauto", "comauto"]


def test_tail(clrd: Triangle) -> None:
    """
    Triangle.tail(n) returns a Triangle limited to the last n rows of the index axis.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    assert clrd.tail(3).shape[0] == 3
    assert list(clrd.tail(3).index["LOB"]) == ["wkcomp", "comauto", "wkcomp"]


def test_add_df_passthru(raa: Triangle) -> None:
    """
    Check equivalent behavior between Triangle and DataFrame passed-thru methods.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    frame = raa.to_frame()

    # String serialization
    assert raa.to_csv() == frame.to_csv()
    assert raa.to_html() == frame.to_html()

    # DataFrame-returning methods
    assert raa.describe().equals(frame.describe())
    assert raa.drop_duplicates().equals(frame.drop_duplicates())
    assert raa.melt().equals(frame.melt())
    assert raa.unstack().equals(frame.unstack())


def test_xs(clrd: Triangle, genins: Triangle) -> None:
    """
    Tests the xs method

    Parameters
    ----------
    clrd : Triangle
        The clrd sample dataset Triangle.

    genins : Triangle
        The genins sample dataset Triangle.

    Returns
    -------
    None
    """
    # when slicing with .loc on the first term in the index, Triangle will drop the term
    assert clrd.xs("Adriatic Ins Co") == clrd.loc["Adriatic Ins Co"]
    assert clrd.xs("Adriatic Ins Co").index.equals(clrd.loc["Adriatic Ins Co"].index)
    # when slicing with .loc on the all term in the index, Triangle will not drop any term
    assert (
        clrd.xs(("Agway Ins Co", "comauto"), drop_level=False)
        == clrd.loc["Agway Ins Co", "comauto"]
    )
    assert clrd.xs(("Agway Ins Co", "comauto"), drop_level=False).index.equals(
        clrd.loc["Agway Ins Co", "comauto"].index
    )
    # when all index terms are included in xs and drop_level is True, the default 'Total' index value is provided
    assert clrd.xs(("Agway Ins Co", "comauto"), drop_level=True).index.equals(
        genins.index
    )
    # when slicing with .loc on the second or subsequent terms in the index, Triangle will not drop the term
    assert (
        clrd.xs("comauto", level=1, drop_level=False)
        == clrd.loc[clrd["LOB"] == "comauto"]
    )
    assert clrd.xs("comauto", level=1, drop_level=False).index.equals(
        clrd.loc[clrd["LOB"] == "comauto"].index
    )
    # level works with either integer index or name of the index column
    assert clrd.xs("comauto", level=1) == clrd.xs("comauto", level="LOB")
    assert clrd.xs("comauto", level=1).index.equals(
        clrd.xs("comauto", level="LOB").index
    )


def test_intersection(clrd: Triangle, genins: Triangle) -> None:
    """
    Intersecting two Triangles with different index labels returns empty Triangle

    Parameters
    ----------
    clrd : Triangle
        The clrd sample dataset Triangle.

    genins : Triangle
        The genins sample dataset Triangle.

    Returns
    -------
    None
    """
    assert clrd.intersection(genins)._dimensionality == "empty"
