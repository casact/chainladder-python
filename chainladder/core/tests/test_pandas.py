from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


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