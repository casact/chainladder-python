from __future__ import annotations
import numpy as np
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle

from chainladder.utils.sparse import COO


def test_arithmetic_ndarray_other_backend(raa: Triangle) -> None:
    """
    When the left side of an arithmetic operator has a sparse backend, and the right side is a numpy ndarray,
    The result should have a sparse backend.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    sparse_raa = raa.set_backend("sparse")
    other = np.ones(sparse_raa.shape)
    result = sparse_raa + other
    assert result.array_backend == "sparse"
    assert result.set_backend("numpy") == raa + 1


def test_arithmetic_coo_other_backend(raa: Triangle) -> None:
    """
    When the left side of an arithmetic operator has a numpy backend, and the right side is a sparse COO,
    The result should have a sparse backend.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    numpy_raa = raa.set_backend("numpy")
    other = COO.from_numpy(np.ones(numpy_raa.shape))
    result = numpy_raa + other
    assert result.array_backend == "sparse"
    assert result.set_backend("numpy") == raa + 1


def test_arithmetic_grain_mismatch_raises(raa: Triangle, qtr: Triangle) -> None:
    """
    Add two triangles with different grain. Raise an error.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    qtr: Triangle
        The quarterly sample data set Triangle.

    Returns
    -------
    None
    """
    with pytest.raises(
        ValueError,
        match="Triangle arithmetic requires both triangles to be the same grain.",
    ):
        raa + qtr


def test_arithmetic_union(raa):
    assert raa.shape == (raa - raa[raa.valuation < "1987"]).shape
    assert raa[raa.valuation<'1986'] + raa[raa.valuation>='1986'] == raa


def test_arithmetic_across_keys(qtr):
    assert (qtr.sum(1) - qtr.iloc[:, 0]) == qtr.iloc[:, 1]


def test_arithmetic_1(raa):
    x = raa
    assert -(((x / x) + 0) * x) == -(+x)
    assert 1 - (x / x) ==  0 * x * 0


def test_rtruediv(raa):
    xp = raa.get_array_module()
    assert xp.nansum(abs(((1 / raa) * raa).values[0, 0] - raa.nan_triangle)) < 0.00001


def test_vector_division(raa):
    raa.latest_diagonal / raa


def test_multiindex_broadcast(clrd):
    clrd = clrd["CumPaidLoss"]
    clrd / clrd.groupby("LOB").sum()


def test_index_broadcasting(clrd):
    """ Basic broadcasting where b is a subset of a """
    assert ((clrd / clrd.sum()) - ((1 / clrd.sum()) * clrd)).sum().sum().sum() < 1e-4

def test_index_broadcasting2(clrd):
    """ b.key_labels are a subset of a.key_labels and b is missing some elements """
    a = clrd['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum().iloc[:-1]
    c = a + b
    assert (a.index == c.index).all().all()


def test_index_broadcasting3(clrd):
    """ b.key_labels are a subset of a.key_labels and a is missing some elements """
    a = clrd[~clrd['LOB'].isin(['wkcomp', 'medmal'])]['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum()
    c = a + b
    assert (a.index == c[~c['LOB'].isin(['wkcomp', 'medmal'])].index).all().all()
    assert len(c) - len(a) == 2


def test_index_broadcasting4(clrd):
    """ b should broadcast to a if b only has one index element """
    a = clrd['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum().iloc[0]
    c = a + b
    assert (a.index == c.index).all().all()


def test_index_broadacsting4(clrd):
    """ If one triangle has key_labels that are not a subset of the other, then fail """
    a = clrd['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum()
    idx = b.index
    idx['New Field'] = 'New'
    b.index = idx
    with pytest.raises(ValueError, match="Index broadcasting is ambiguous"):
        a + b

def test_index_broadcasting5(clrd):
    """ If a and b have shared key labels but no matching levels, then they will stack """
    a = clrd['CumPaidLoss'].iloc[:300]
    b = clrd['CumPaidLoss'].iloc[300:]
    c = a + b
    assert c.sort_index() == clrd['CumPaidLoss'].sort_index()


def test_index_broadacsting6(clrd):
    a = clrd['CumPaidLoss'].iloc[:100]
    b = clrd['CumPaidLoss'].iloc[50:150]
    c = clrd['CumPaidLoss'].iloc[50:100]
    d = a + b - c
    assert d.sort_index() == clrd['CumPaidLoss'].iloc[:150].sort_index()


def test_index_broadcasting_ambiguous(clrd: Triangle) -> None:
    """
    Attempt to add two triangles with inconsistent indexes. Raise a ValueError.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set fixture.

    Returns
    -------
    None
    """
    a = clrd['CumPaidLoss'].groupby('GRNAME').sum()
    b = clrd['CumPaidLoss'].groupby('LOB').sum()
    with pytest.raises(ValueError, match="Index broadcasting is ambiguous"):
        _= a + b


def test_prep_columns_reindexes_superset(clrd: Triangle) -> None:
    """
    When one triangle's columns are a strict superset of the other's, the subset
    triangle is reindexed with the missing columns filled as zero. Test both
    directions: x as superset and y as superset.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set fixture.

    Returns
    -------
    None
    """
    x = clrd[['CumPaidLoss', 'EarnedPremNet', 'IncurLoss']]
    y = clrd[['CumPaidLoss', 'IncurLoss']]
    for result in [x + y, y + x]:
        assert set(result.columns) == {'CumPaidLoss', 'EarnedPremNet', 'IncurLoss'}
        assert result[['CumPaidLoss', 'IncurLoss']] == clrd[['CumPaidLoss', 'IncurLoss']] * 2
        assert result['EarnedPremNet'] == clrd['EarnedPremNet']