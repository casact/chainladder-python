from __future__ import annotations

import numpy as np
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle

def test_slice_by_boolean(clrd : Triangle) -> None:
    assert (
        clrd[clrd["LOB"] == "ppauto"].loc["Wolverine Mut Ins Co"]["CumPaidLoss"]
        == clrd.loc["Wolverine Mut Ins Co"].loc["ppauto"]["CumPaidLoss"]
    )


def test_slice_by_loc(clrd):
    assert clrd.loc["Aegis Grp"].loc["comauto"].index.iloc[0, 0] == "comauto"


def test_slice_origin(raa: Triangle) -> None:
    """
    Slice the Triangle on the origin. Check the shape and year boundary.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture

    Returns
    -------
    None
    """
    assert raa[raa.origin > "1985"].shape == (1, 1, 5, 10)
    assert raa.loc[..., raa.origin <= "1985", :].origin.max().year == 1985


def test_slice_development(raa: Triangle) -> None:
    """
    Slice the Triangle on the development axis. Check the shape and development periods.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture

    Returns
    -------
    None

    """
    assert raa[raa.development < 72].shape == (1, 1, 10, 5)
    assert raa.loc[..., 24:].development.min() == 24


def test_slice_by_loc_iloc(clrd):
    assert clrd.groupby("LOB").sum().loc["comauto"].index.iloc[0, 0] == "comauto"
    assert len(clrd.loc[clrd.index.iloc[150]]) == 1


def test_slicers_honor_order(clrd):
    clrd = clrd.groupby("LOB").sum()
    assert clrd.iloc[[1, 0], :].iloc[0, 1] == clrd.iloc[1, 1]  # row
    assert clrd.iloc[[1, 0], [1, 0]].iloc[0, 0] == clrd.iloc[1, 1]  # col
    assert clrd.loc[:, ["CumPaidLoss", "IncurLoss"]].iloc[0, 0] == clrd.iloc[0, 1]
    assert (
        clrd.loc[["ppauto", "medmal"], ["CumPaidLoss", "IncurLoss"]].iloc[0, 0]
        == clrd.iloc[3]["CumPaidLoss"]
    )
    assert (
        clrd.loc[clrd["LOB"] == "comauto", ["CumPaidLoss", "IncurLoss"]]
        == clrd[clrd["LOB"] == "comauto"].iloc[:, [1, 0]]
    )
    assert clrd.groupby("LOB").sum() == clrd


def test_backends(clrd):
    clrd = clrd[["CumPaidLoss", "EarnedPremDIR"]]
    a = clrd.iloc[1, 0].set_backend("sparse").dropna()
    b = clrd.iloc[1, 0].dropna()
    assert a == b


def test_union_columns(clrd):
    assert clrd.iloc[:, :3] + clrd.iloc[:, 3:] == clrd


def test_4loc(clrd):
    clrd = clrd.groupby("LOB").sum()
    assert (
        clrd.iloc[:3, :2, 0, 0]
        == clrd[clrd.origin == clrd.origin.min()][
            clrd.development == clrd.development.min()
        ].loc["comauto":"othliab", :"CumPaidLoss", :, :]
    )
    assert (
        clrd.iloc[:3, :2, 0:1, -1]
        == clrd[clrd.development == clrd.development.max()].loc[
            "comauto":"othliab", :"CumPaidLoss", "1988", :
        ]
    )


def test_loc_ellipsis(clrd):
    assert (
        clrd.loc["Aegis Grp"]
        == clrd.loc["Adriatic Ins Co":"Aegis Grp"].loc["Aegis Grp"]
    )
    assert clrd.loc["Aegis Grp", ..., :] == clrd.loc["Aegis Grp"]
    assert clrd.loc[..., 24:] == clrd.loc[..., :, 24:]
    assert clrd.loc[:, ..., 24:] == clrd.loc[..., :, 24:]
    assert clrd.loc[:, "CumPaidLoss"] == clrd.loc[:, "CumPaidLoss", ...]
    assert clrd.loc[..., "CumPaidLoss", :, :] == clrd.loc[:, "CumPaidLoss", :, :]


def test_missing_first_lag(raa):
    x = raa.copy()
    x.values[:, :, :, 0] = 0
    x = x.sum(0)
    assert x.link_ratio.shape == (1, 1, 9, 9)


def test_reverse_slice_integrity(clrd):
    assert clrd.iloc[::-1, ::-1].shape == clrd.shape
    assert np.all(clrd.iloc[:, ::-1].columns.values == clrd.columns[::-1])
    assert clrd.iloc[clrd.index.index[::-1]] + clrd == 2 * clrd


def test_loc_tuple(clrd):
    assert len(clrd.loc[("Adriatic Ins Co", "othliab")]) == 1
    assert clrd.loc[clrd.index] == clrd


def test_at_iat(raa):
    raa1 = raa.copy()
    raa2 = raa.copy()
    _= raa1.at["Total", "values", "1985", 120]
    raa1.at["Total", "values", "1985", 120] = 5
    raa1.at["Total", "values", "1985", 12] = 5
    raa2.iat[0, 0, 4, -1] = 5
    _= raa2.iat[0, 0, 4, -1]
    raa2.iat[-1, -1, 4, 0] = 5
    assert raa1 == raa2


def test_at_iat_exceptions(raa):
    with pytest.raises(ValueError):
        _= raa.iat[0, 0, 4, :]
    with pytest.raises(ValueError):
        _= raa.at["Total", "values", "1985", 0:2]


def test_other_key_unsupported_iterable_raises(raa: Triangle) -> None:
    """
    Pass a nonsense key to raa.loc[], should raise an error.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None

    """
    with pytest.raises(AttributeError, match="Unable to slice"):
        _= raa.loc[:, (0, 1)]


def test_at_setitem_triangle_value(raa: Triangle) -> None:
    """
    Use Triangle.at to set a single value via a TriangleSlicer.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None

    """
    tri = raa.copy()
    tri.at["Total", "values", "1981", 12] = tri.iloc[0, 0, 1:2, 0:1]
    assert tri.at["Total", "values", "1981", 12] == 106.0


@pytest.mark.xfail
def test_sparse_at_iat(prism):
    prism.iloc[0, 0, 0, 0] = 1.0


def test_empty_index_raises(raa: Triangle) -> None:
    """
    Pass an empty list to Triangle.iloc and raise an empty Triangle error.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="Slice returns empty Triangle"):
        _= raa.iloc[[], :]


def test_get_idx_fancy_origin_raises(raa: Triangle) -> None:
    """
    Attempt fancy indexing on origin axis, raise an error.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="Fancy indexing on origin/development is not supported"):
        _= raa.iloc[0, 0, [0, 1, 5], :]


def test_get_idx_fancy_development_raises(raa: Triangle) -> None:
    """
    Attempt fancy indexing on development axis, raise an error.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="Fancy indexing on origin/development is not supported"):
        _= raa.iloc[0, 0, :, [0, 1, 5]]


def test_get_idx_non_contiguous_index_and_columns(clrd: Triangle) -> None:
    """
    Pass lists of non-contiguous index expressions of the index and column axes to Triangle.iloc. Check the values
    of the index and columns returned.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set fixture.

    Returns
    -------
    None

    """
    result = clrd.iloc[[0, 1, 5], [0, 1, 5], :, :]
    expected_index = [
        ['Adriatic Ins Co', 'othliab'],
        ['Adriatic Ins Co', 'ppauto'],
        ['Agency Ins Co Of MD Inc', 'ppauto'],
    ]
    assert result.index.values.tolist() == expected_index
    assert result.columns.tolist() == ['IncurLoss', 'CumPaidLoss', 'EarnedPremNet']


def test_sparse_at_iat1(prism):
    t = prism.copy()
    t.iat[0, 0, 0, 0] = 5.0
    t.at[12138, "reportedCount", "2008-01", 1] = 0
    assert t == prism


def test_sparse_column_assignment(prism):
    t = prism.copy()
    out = t["Paid"]
    t["Paid2"] = t["Paid"]  # New from physical
    t["Paid2"] = lambda x: x["Paid"]  # Existing from virtual
    t["Paid"] = t["Paid2"]  # Existing from physical
    t["Paid3"] = lambda x: x["Paid"]  # New from virtual
    assert out == t["Paid"]
    assert t.shape == (34244, 6, 120, 120)
