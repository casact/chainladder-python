from __future__ import annotations

import chainladder as cl
import importlib
import numpy as np
import pandas as pd
import pytest
import sys


from collections.abc import Iterable
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


def test_get_format_str_reads_from_options() -> None:
    """
    Options for displaying small/medium/large triangles in Jupyter should
    be passed TriangleDisplay.

    Returns
    -------
    None

    """
    small = pd.DataFrame([[1.0, 2.0]])
    medium = pd.DataFrame([[100.0, 200.0]])
    large = pd.DataFrame([[10000.0, 20000.0]])
    try:
        cl.options.set_option("display.html.auto_format_small", "{0:.1f}")
        cl.options.set_option("display.html.auto_format_medium", "{0:.2f}")
        cl.options.set_option("display.html.auto_format_large", "{0:.3f}")
        assert TriangleDisplay._get_format_str(small) == "{0:.1f}"
        assert TriangleDisplay._get_format_str(medium) == "{0:.2f}"
        assert TriangleDisplay._get_format_str(large) == "{0:.3f}"
    finally:
        cl.options.reset_option("display.html.auto_format_small")
        cl.options.reset_option("display.html.auto_format_medium")
        cl.options.reset_option("display.html.auto_format_large")


def test_get_format_str_reads_thresholds_from_options() -> None:
    """
    Changing the threshold at which a triangle is considered small/medium/large
    should update the corresponding format string if the size classification of the
    Triangle changes.

    Returns
    -------
    None

    """
    data = pd.DataFrame([[500.0, 600.0]])
    assert TriangleDisplay._get_format_str(data) == "{0:,.2f}"
    try:
        # Triangle moves from medium to small.
        cl.options.set_option("display.html.auto_format_small_threshold", 1000)
        assert TriangleDisplay._get_format_str(data) == "{0:,.4f}"
    finally:
        cl.options.reset_option("display.html.auto_format_small_threshold")
    assert TriangleDisplay._get_format_str(data) == "{0:,.2f}"

    try:
        # Triangle moves from medium to large.
        cl.options.set_option("display.html.auto_format_medium_threshold", 100)
        assert TriangleDisplay._get_format_str(data) == "{:,.0f}"
    finally:
        cl.options.reset_option("display.html.auto_format_medium_threshold")
    assert TriangleDisplay._get_format_str(data) == "{0:,.2f}"


def test_display_option_default_is_none() -> None:
    """
    Without a user override, both display.value_format and
    display.pattern_format should resolve to None, signaling that each
    display pathway should fall back to its own default behavior.

    Returns
    -------
    None

    """
    assert TriangleDisplay._display_option(is_pattern=False) is None
    assert TriangleDisplay._display_option(is_pattern=True) is None


def test_normalize_format_wraps_format_string() -> None:
    """
    A format-string value_format should be normalized into a callable
    that correctly formats a provided value.

    Returns
    -------
    None

    """
    formatter = TriangleDisplay._normalize_format("{:,.0f}")
    assert formatter(1234.5) == "1,234"


def test_normalize_format_passes_through_callable() -> None:
    """
    A callable value_format should be returned unchanged.

    Returns
    -------
    None

    """

    def swiss_format(x):
        return f"{x:,.0f}".replace(",", "'")

    formatter = TriangleDisplay._normalize_format(swiss_format)
    assert formatter is swiss_format
    assert formatter(1234.5) == "1'234"


def test_value_format_override_affects_repr_and_html(raa: Triangle) -> None:
    """
    Setting display.value_format should override both the console and
    HTML/Jupyter default formatting for value triangles.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    try:
        cl.options.set_option("display.value_format", "{:,.0f}")
        assert "5,012" in repr(raa)
        assert "5,012" in raa._repr_html_()
    finally:
        cl.options.reset_option("display.value_format")
    # Back to the (unchanged) per-pathway defaults.
    assert "5012.0" in repr(raa)
    assert "5,012" not in repr(raa)


def test_value_format_override_affects_heatmap(raa: Triangle) -> None:
    """
    Setting display.value_format with a callable should apply to
    heatmap() as well.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """

    def swiss_format(x):
        return f"{x:,.0f}".replace(",", "'")

    try:
        cl.options.set_option("display.value_format", swiss_format)
        html_str = raa.heatmap().data
        assert "5'012" in html_str
    finally:
        cl.options.reset_option("display.value_format")


def test_pattern_format_independent_of_value_format(raa: Triangle) -> None:
    """
    display.pattern_format should only affect pattern triangles (e.g.
    link_ratio), leaving value-triangle formatting untouched, and vice
    versa.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    try:
        cl.options.set_option("display.pattern_format", "{:.4f}".format)
        assert "1.6498" in repr(raa.link_ratio)
        # Value triangle repr is unaffected by the pattern-only override.
        assert "5012.0" in repr(raa)
    finally:
        cl.options.reset_option("display.pattern_format")


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


def test_triangle_not_iterable(raa: Triangle) -> None:
    """
    A Triangle is a 4-D container, not a 1-D sequence, so it must not be
    iterable (GH #142). This prevents pandas from misclassifying it as a
    sequence and iterating it while formatting a DataFrame cell.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    assert not isinstance(raa, Iterable)
    with pytest.raises(TypeError):
        iter(raa)


def test_triangle_in_dataframe_cell_display(clrd: Triangle) -> None:
    """
    Storing a Triangle in a DataFrame cell and displaying the DataFrame must
    not crash (GH #142).

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    df = pd.DataFrame(data=[["clrd", clrd]], columns=["name", "cl_triangle"])
    # Both the text and HTML representations previously raised because pandas
    # iterated the Triangle stored in the cell.
    assert isinstance(repr(df), str)
    assert isinstance(df._repr_html_(), str)


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

    blocked = {"IPython": None, "IPython.core": None, "IPython.core.display": None}
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

    blocked = {"IPython": None, "IPython.core": None, "IPython.core.display": None}
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
