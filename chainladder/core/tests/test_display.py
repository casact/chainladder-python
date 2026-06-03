from __future__ import annotations

import chainladder as cl
import importlib
import numpy as np
import pandas as pd
import pytest
import sys


from chainladder.core.display import TriangleDisplay
from lxml import etree, html as lxml_html
from unittest import mock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def check_html(html: str) -> None:
    """
    Parse the HTML and raise an assertion error if it is malformed.

    Parameters
    ----------
    html: str
        The HTML string.

    Returns
    -------
    None

    """

    parser = etree.HTMLParser()
    lxml_html.fromstring(html, parser=parser)

    # Raise assertion error if one is detected. If so, print the error log as a list.
    assert len(parser.error_log) == 0, list(parser.error_log)

def test_check_html() -> None:
    """
    Make sure check_html does its job on a malformed string.

    Parameters
    ----------
    html: str

    Returns
    -------
    None

    """
    with pytest.raises(AssertionError):
        check_html("<b><i>text</b></i>")


def test_dimensionality_empty(empty_triangle: Triangle) -> None:
    """
    Inspect the dimensionality of an empty triangle.

    Parameters
    ----------
    empty_triangle: Triangle
        An empty triangle.

    Returns
    -------
    None

    """

    assert empty_triangle._dimensionality == "empty"


def test_empty_attribute_empty(empty_triangle: Triangle) -> None:
    assert empty_triangle.empty is True


def test_empty_attribute_multi(clrd: Triangle) -> None:
    assert clrd.empty is False


def test_dimensionality_attribute_empty(empty_triangle: Triangle) -> None:
    assert empty_triangle.dimensionality == "empty"


def test_dimensionality_attribute_single(raa: Triangle) -> None:
    assert raa.dimensionality == "single"


def test_dimensionality_attribute_multi(clrd: Triangle) -> None:
    assert clrd.dimensionality == "multi"


def test_dimensionality_multi(clrd: Triangle) -> None:
    """
    Inspect dimensionality of a multidimensional triangle.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    assert clrd._dimensionality == "multi"


def test_repr_empty(empty_triangle: Triangle) -> None:
    """
    Inspect the repr of an empty triangle.

    Parameters
    ----------
    empty_triangle: Triangle
        An empty triangle.

    Returns
    -------
    None

    """

    assert repr(empty_triangle) == "Empty Triangle."


def test_repr_multi(clrd: Triangle) -> None:
    """
    Inspect the repr of a multidimensional triangle.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    assert "Triangle Summary" in repr(clrd)


def test_repr_html_empty(empty_triangle: Triangle):
    """
    Inspect the HTML representation of an empty triangle.

    Parameters
    ----------
    empty_triangle: Triangle
        An empty triangle.

    Returns
    -------
    None

    """

    assert empty_triangle._repr_html_() == "Empty Triangle."


def test_repr_html_single(raa):
    """
    Inspect the HTML representation of a single-dimensional triangle.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    html_str: str = raa._repr_html_()
    assert "<table" in html_str
    check_html(html=html_str)

def test_repr_html_multi(clrd: Triangle) -> None:
    """
    Inspect the HTML representation of a multidimensional triangle.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    html_str = clrd._repr_html_()
    assert "Triangle Summary" in html_str
    assert "<table" in html_str
    check_html(html=html_str)

def test_get_format_str_all_nan() -> None:
    """
    Extract the format string from a DataFrame when data are all nan.

    Returns
    -------
    None

    """
    data = pd.DataFrame([[np.nan, np.nan]])
    assert TriangleDisplay._get_format_str(data) == ""


def test_get_format_str_small() -> None:
    """
    Extract the format string from a DataFrame when mean of data is less than 10.

    Returns
    -------
    None

    """
    data = pd.DataFrame([[1.0, 2.0]])
    assert TriangleDisplay._get_format_str(data) == "{0:,.4f}"


def test_get_format_str_medium() -> None:
    """
    Extract the format string from a DataFrame when mean of data is less than 1000.

    Returns
    -------
    None

    """
    data = pd.DataFrame([[100.0, 200.0]])
    assert TriangleDisplay._get_format_str(data) == "{0:,.2f}"


def test_repr_format_semi_annual(prism: Triangle) -> None:
    """
    When origin has semiannual grain, "H1" and "H2" should appear in the index.

    Parameters
    ----------
    prism: Triangle
        The prism sample data set.

    Returns
    -------
    None

    """
    prism = prism.sum()[["reportedCount"]]
    semi = prism.grain("OSDM")
    df = semi._repr_format()
    assert any("H1" in str(i) or "H2" in str(i) for i in df.index)


def test_heatmap_multi_raises(clrd: Triangle) -> None:
    """
    Heatmap only works on a single-dimension triangle. Raise a ValueError if multidimensional.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="heatmap"):
        clrd.heatmap()


def test_heatmap_no_ipython(raa: Triangle) -> None:
    """
    Raise ImportError when calling heatmap when IPython is not installed.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    import chainladder.core.display as display_mod

    blocked = {
        "IPython": None,
        "IPython.core": None,
        "IPython.core.display": None
    }
    with mock.patch.dict(sys.modules, blocked):
        importlib.reload(display_mod)
        with pytest.raises(ImportError, match=r"heatmap\(\) requires IPython\."):
            raa.heatmap()

    importlib.reload(display_mod)


def test_display_import_fallback_when_ipython_missing() -> None:
    """
    Set the variables of HTML and IPython in the display module to None when IPython is not installed.

    Returns
    -------
    None

    """
    import chainladder.core.display as display_mod

    blocked = {
        "IPython": None,
        "IPython.core": None,
        "IPython.core.display": None
    }
    with mock.patch.dict(sys.modules, blocked):
        importlib.reload(display_mod)
        assert display_mod.HTML is None
        assert display_mod.IPython is None

    importlib.reload(display_mod)


def test_heatmap_render(raa):
    """The heatmap method should render correctly given the sample."""
    try:
        raa.heatmap()

    except:
        assert False


def test_to_frame(raa):
    try:
        cl.Chainladder().fit(raa).cdf_.to_frame()
        cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=False)
        cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=True)
        cl.Chainladder().fit(raa).ultimate_.to_frame()
        cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=False)
        cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=True)

    except:
        assert False


def test_labels(xyz):
    assert (
        xyz.valuation_date.strftime("%Y-%m-%d %H:%M:%S.%f")
        == "2008-12-31 23:59:59.999999"
    )
    assert xyz.origin_grain == "Y"
    assert xyz.development_grain == "Y"
    assert xyz.shape == (1, 5, 11, 11)
    assert xyz.index_label == ["Total"]
    assert xyz.columns_label == ["Incurred", "Paid", "Reported", "Closed", "Premium"]
    assert xyz.origin_label == ["AccidentYear"]
