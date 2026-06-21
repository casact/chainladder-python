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
    assert raa1.at["Total", "values", "1985", 72] == 26180.0
    raa1.at["Total", "values", "1985", 120] = 5
    raa1.at["Total", "values", "1985", 12] = 5
    raa2.iat[0, 0, 4, -1] = 5
    assert raa2.iat[0, 0, 4, -1] == 5
    raa2.iat[-1, -1, 4, 0] = 5
    assert raa1 == raa2


def test_at_iat_exceptions(raa):
    with pytest.raises(ValueError):
        _= raa.iat[0, 0, 4, :]
    with pytest.raises(ValueError):
        _= raa.at["Total", "values", "1985", 0:2]


def test_at_check_index_full_axis_slice_raises(raa: Triangle) -> None:
    """
    Triange.at[] requires all 4 axis values. Raise an error otherwise.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="Invalid Index in At slicer"):
        _= raa.at["Total", "values"]


def test_at_check_index_full_axis_slice_on_non_unit_axis_raises(raa: Triangle) -> None:
    """
    Calling .at[] with `slice(None, None, None)` to specify the full origin axis.
    Raise an error because this would correspond to multiple data elements.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="Invalid Index in At slicer"):
        _= raa.at["Total", "values", slice(None, None, None), 12]


def test_at_check_index_full_axis_slice_on_unit_axis(raa: Triangle) -> None:
    """
    Calling .at[] with `slice(None, None, None)` to specify the full columns axis.
    Works since raa only has 1 column.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    assert raa.at["Total", slice(None, None, None), "1985", 12] == raa.at["Total", "values", "1985", 12]


def test_at_requires_all_axes(raa: Triangle) -> None:
    """
    raa.at[] requires all 4 axes (index, columns, origin, development) to be
    specified explicitly, mirroring pandas.DataFrame.at[]'s requirement that
    both axes be given as scalar labels. Ellipsis is not allowed.
    """
    with pytest.raises(ValueError, match="Invalid Index in At slicer"):
        _= raa.at[..., "1985", 12]


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


def test_loc_setitem_triangle_value(clrd: Triangle) -> None:
    """
    Assign a Triangle to a .loc[] slice.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set fixture.

    Returns
    -------
    None

    """
    tri = clrd.copy()
    sub = tri.loc["Aegis Grp", "comauto"].copy()
    if tri.array_backend == "sparse":
        with pytest.raises(ValueError, match="Setting values with sparse backend requires .at or .iat"):
            tri.loc["Aegis Grp", "comauto"] = sub * 2
    else:
        tri.loc["Aegis Grp", "comauto"] = sub * 2
        assert tri.loc["Aegis Grp", "comauto"] == sub * 2



def test_invalid_iloc_sparse_assignment(prism) -> None:
    """
    Assignment via Triangle.iloc[] does not work on sparse backend.

    Parameters
    ----------
    prism: Triangle
        The prism sample data set Triangle.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="Setting values with sparse backend requires .at or .iat"):
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


def test_sparse_iat_setitem_triangleslicer_value(prism: Triangle) -> None:
    """
    Iat.[] = value accepts a Triangle (TriangleSlicer) as the assigned value
    when the backend is sparse, if the slicer resolves to a single value.

    Parameters
    ----------
    prism: Triangle
        The prism sample data set fixture, which uses the sparse backend.

    Returns
    -------
    None
    """
    t = prism.copy()
    src = t.iloc[0:1, 0:1, 0:1, 1:2]
    t.iat[2, 1, 0, 0] = src
    assert t.iat[2, 1, 0, 0] == src.iat[0, 0, 0, 0]


def test_virtual_columns_pop_on_overwrite(raa: Triangle) -> None:
    """
    Assigning a non-callable value to a column previously reserved for a
    lazy virtual column removes that column from VirtualColumns.columns.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    t = raa.copy()
    t["vcol"] = lambda x: x["values"] * 2
    assert "vcol" in t.virtual_columns.columns
    t["vcol"] = t["values"] * 3
    assert "vcol" not in t.virtual_columns.columns


def test_setitem_virtual_column_numpy_backend(raa: Triangle) -> None:
    """
    Assign a callable (virtual column) to a new key on a numpy-backed Triangle.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    tri = raa.copy()
    assert tri.array_backend == "numpy"
    tri["double"] = lambda x: x["values"] * 2
    assert "double" in tri.columns
    assert tri["double"] == tri["values"] * 2


def test_setitem_value_backend_conversion(raa: Triangle) -> None:
    """
    Assign a Triangle value with a different array_backend than the target.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    tri = raa.copy()
    value = (tri["values"] * 2).set_backend("sparse")
    assert tri.array_backend != value.array_backend
    tri["values"] = value
    assert tri.array_backend == raa.array_backend
    assert tri["values"] == raa["values"] * 2


def test_setitem_existing_column_triangle_value(raa: Triangle) -> None:
    """
    Reassign an existing column to a Triangle value on a non-sparse backend.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    tri = raa.copy()
    assert tri.array_backend != "sparse"
    value = tri["values"] * 2
    tri["values"] = value
    assert tri["values"] == raa["values"] * 2


def test_setitem_existing_column_array_value(raa: Triangle) -> None:
    """
    Reassign an existing column to a raw array value on a non-sparse backend.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """
    tri = raa.copy()
    assert tri.array_backend != "sparse"
    value = (tri["values"] * 3).values
    assert not isinstance(value, type(tri))
    tri["values"] = value
    assert tri["values"] == raa["values"] * 3


def test_sparse_column_assignment(prism):
    t = prism.copy()
    out = t["Paid"]
    t["Paid2"] = t["Paid"]  # New from physical
    t["Paid2"] = lambda x: x["Paid"]  # Existing from virtual
    t["Paid"] = t["Paid2"]  # Existing from physical
    t["Paid3"] = lambda x: x["Paid"]  # New from virtual
    assert out == t["Paid"]
    assert t.shape == (34244, 6, 120, 120)


def test_setitem_new_column_misaligned_triangle(raa: Triangle) -> None:
    """
    Assigning a misaligned Triangle (fewer origin periods) to a new column.
    Aligns the new column onto the existing origin grid, filling missing
    origins with NaN.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set fixture.

    Returns
    -------
    None
    """

    tri = raa.copy()
    misaligned = raa[raa.origin > "1985"]
    tri["misaligned"] = misaligned
    # Check the shape, new column should be added.
    assert tri.shape == (1, 2, 10, 10)
    new_col = tri["misaligned"]
    # Origin periods 1985 and prior should be nan.
    assert np.isnan(new_col.values[0, 0, :5, :]).all()
    # Origin periods 1986 and beyond should match.
    assert np.allclose(
        new_col.values[0, 0, 5:, :], misaligned.values[0, 0, :, :], equal_nan=True
    )
