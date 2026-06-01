import numpy as np

from chainladder.utils.sparse import (
    array,
    floor,
    COO,
    where
)


def test_array_from_list_default_fill_value() -> None:
    """
    Tests chainladder.utils.sparse.array() when no fill value is provided.
    Checks whether the default nan is filled.

    Returns
    -------
    None

    """
    result: COO = array([1.0, 2.0, 3.0])
    assert isinstance(result, COO)
    assert np.isnan(result.fill_value)


def test_array_from_list_explicit_fill_value() -> None:
    """
    Tests chainladder.utils.sparse.array() when a fill value of 0 is provided.
    Checks whether the 0 is filled.

    Returns
    -------

    """
    result: COO = array([1, 2, 3], fill_value=0)
    assert isinstance(result, COO)
    assert result.fill_value == 0


def test_array_from_coo_default_fill_value() -> None:
    """
    Tests chainladder.utils.sparse.array() when initializing from a sparse array with a default fill value.

    Returns
    -------
    None

    """
    coo = COO.from_numpy(np.array([1.0, 2.0, 3.0]))
    result: COO = array(coo)
    assert isinstance(result, COO)
    assert np.isnan(result.fill_value)


def test_array_from_coo_explicit_fill_value() -> None:
    """
    Tests chainladder.utils.sparse.array() when initializing from a sparse array with an explicit fill value.

    Returns
    -------
    None

    """
    coo = COO.from_numpy(np.array([1, 2, 3]))
    result: COO = array(coo, fill_value=0)
    assert isinstance(result, COO)
    assert result.fill_value == 0


def test_where_selects_from_two_arrays() -> None:
    """
    Tests element-wise where across sparse arrays. Calls np.where on each element triplet
    (cond[i], a[i], b[i]) - returning a[i] where the condition is True and b[i] where it's False.

    Returns
    -------
    None
    """
    a: COO = array([1.0, 2.0, 3.0])
    b: COO = array([10.0, 20.0, 30.0])
    cond: COO = array([True, False, True])
    result: COO = where(cond, a, b)
    assert isinstance(result, COO)
    np.testing.assert_array_equal(result.todense(), [1.0, 20.0, 3.0])


def test_floor_rounds_down() -> None:
    """
    Checks floor function rounding down with positive and negative floats.

    Returns
    -------
    None
    """
    a: COO = array([1.2, 2.7, -0.3])
    result: COO = floor(a)
    np.testing.assert_array_equal(result.todense(), [1.0, 2.0, -1.0])


def test_floor_returns_copy() -> None:
    """
    Checks that floor returns a copy and does not mutate its input,
    mirroring np.floor (see issue #740).

    Returns
    -------
    None
    """
    a = array([1.2, 2.7, -0.3])
    result: COO = floor(a)
    assert result is not a
    np.testing.assert_array_equal(result.todense(), [1.0, 2.0, -1.0])
    np.testing.assert_array_equal(a.todense(), [1.2, 2.7, -0.3])
