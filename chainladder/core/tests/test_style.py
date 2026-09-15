# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.io.formats.style import Styler as PandasStyler

import chainladder as cl
from chainladder.core.style import Styler


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


def test_is_public() -> None:
    """
    Check that Styler is importable from the top-level chainladder package.

    Returns
    -------
    None

    """
    assert cl.Styler is Styler


def test_is_pandas_styler_subclass() -> None:
    """
    Check that Styler extends pandas' own Styler rather than replacing it.

    Returns
    -------
    None

    """
    assert issubclass(Styler, PandasStyler)


def test_wraps_a_dataframe(df: pd.DataFrame) -> None:
    """
    Check that Styler wraps a DataFrame the same way pandas' Styler does.

    Parameters
    ----------
    df: pd.DataFrame
        A simple two-column DataFrame fixture.

    Returns
    -------
    None

    """
    styler = Styler(df)
    assert styler.data is df


def test_chained_methods_preserve_subclass(df: pd.DataFrame) -> None:
    """
    Check that chaining a pandas Styler method still returns a chainladder Styler,
    not a plain pandas Styler.

    Parameters
    ----------
    df: pd.DataFrame
        A simple two-column DataFrame fixture.

    Returns
    -------
    None

    """
    result = (
        Styler(df)
        .format(precision=1)
        .hide(  # pyright: ignore[reportAttributeAccessIssue]
            axis="columns", subset=["a"]
        )
    )
    assert isinstance(result, Styler)


def test_renders_identically_to_pandas_styler(df: pd.DataFrame) -> None:
    """
    Check that Styler produces the same HTML as pandas' own Styler.

    Parameters
    ----------
    df: pd.DataFrame
        A simple two-column DataFrame fixture.

    Returns
    -------
    None

    """
    cl_html = Styler(df, uuid="fixed").format(precision=2).to_html()
    pd_html = PandasStyler(df, uuid="fixed").format(precision=2).to_html()
    assert cl_html == pd_html


def test_triangle_style_is_a_property(raa) -> None:
    """
    Check that Triangle.style is a property, matching pandas.DataFrame.style,
    rather than a method that must be called.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    assert isinstance(type(raa).style, property)


def test_triangle_style_returns_styler(raa) -> None:
    """
    Check that Triangle.style returns a chainladder Styler.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    assert isinstance(raa.style, Styler)


def test_triangle_style_wraps_to_frame(raa) -> None:
    """
    Check that Triangle.style wraps the same data as
    Triangle.to_frame(origin_as_datetime=False).

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    pd.testing.assert_frame_equal(
        raa.style.data, raa.to_frame(origin_as_datetime=False)
    )


def test_triangle_style_is_fresh_each_access(raa) -> None:
    """
    Check that each access to Triangle.style returns a new Styler, matching
    pandas.DataFrame.style's own behavior -- so unrelated uses don't share state
    or a uuid.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    assert raa.style is not raa.style


def test_highlight_lower_triangle_styles_exactly_the_lower_triangle(raa) -> None:
    """
    Check that every styled cell -- and only those cells -- correspond to a
    NaN in the Triangle's own nan_triangle mask.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    styler = raa.style.highlight_lower_triangle(color="lightgray")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}

    expected = {
        (r, c)
        for r, row in enumerate(np.isnan(raa.nan_triangle))
        for c, is_lower in enumerate(row)
        if is_lower
    }
    assert styled == expected
    assert all(
        v == [("background-color", "lightgray")] for v in styler.ctx.values() if v
    )


def test_highlight_lower_triangle_props_overrides_color(raa) -> None:
    """
    Supplying a custom props overrides the color parameter.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    styler = raa.style.highlight_lower_triangle(
        color="lightgray", props="background-color: blue; opacity: 50%;"
    )
    assert "opacity: 50%" in styler.to_html()
    assert "lightgray" not in styler.to_html()


def test_highlight_lower_triangle_text_color(raa) -> None:
    """
    Check that text_color sets the text color alongside the background.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    styler = raa.style.highlight_lower_triangle(color="#DDEBF7", text_color="#1F4E78")
    styler._compute()
    styled = [v for v in styler.ctx.values() if v]
    assert len(styled) > 0
    assert all(
        v == [("background-color", "#DDEBF7"), ("color", "#1F4E78")] for v in styled
    )


def test_highlight_lower_triangle_text_color_ignored_without_default_props(
    raa,
) -> None:
    """
    Check that text_color is ignored when props overrides color, matching
    color's own documented behavior.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    styler = raa.style.highlight_lower_triangle(
        text_color="#1F4E78", props="background-color: blue;"
    )
    assert "1F4E78" not in styler.to_html()


def test_highlight_lower_triangle_returns_styler(raa) -> None:
    """
    Check that the subclass is preserved through the chained call.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    assert isinstance(raa.style.highlight_lower_triangle(), Styler)


def test_highlight_lower_triangle_requires_a_triangle(df: pd.DataFrame) -> None:
    """
    Check that calling it on a Styler not built from Triangle.style raises a
    clear error, since there's no way to know which cells are the lower triangle.

    Parameters
    ----------
    df: pd.DataFrame
        A simple two-column DataFrame fixture.

    Returns
    -------
    None

    """
    with pytest.raises(
        ValueError, match="requires a Styler created from Triangle.style"
    ):
        Styler(df).highlight_lower_triangle()


def test_highlight_lower_triangle_predicted_cells_not_highlighted_by_default(
    raa,
) -> None:
    """
    A fully-predicted Triangle (e.g. full_triangle_) has no NaN cells left,
    and once it carries an ultimate column its own nan_triangle collapses to
    all-observed -- so without an explicit valuation_date, nothing gets
    highlighted.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    full = cl.Chainladder().fit(raa).full_triangle_
    styler = full.style.highlight_lower_triangle(color="lightgray")
    styler._compute()
    assert not any(styler.ctx.values())


def test_highlight_lower_triangle_with_valuation_date_highlights_predicted_cells(
    raa,
) -> None:
    """
    Passing the original valuation_date recovers which cells were originally
    the lower (unobserved) triangle, even though they've since been filled in
    with predicted values.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    full = cl.Chainladder().fit(raa).full_triangle_
    styler = full.style.highlight_lower_triangle(
        color="lightgray", valuation_date=raa.valuation_date
    )
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}

    val_array = np.array(full.valuation).reshape(full.shape[-2:], order="F")
    expected = {
        (r, c)
        for r, row in enumerate(val_array > raa.valuation_date)
        for c, is_lower in enumerate(row)
        if is_lower
    }
    assert styled == expected
    assert len(styled) > 0
    assert all(
        v == [("background-color", "lightgray")] for v in styler.ctx.values() if v
    )


def test_highlight_lower_triangle_requires_matching_shape(clrd) -> None:
    """
    Check that a Triangle whose to_frame() isn't a single origin-by-development
    grid (multiple index labels and columns, here) raises a clear error rather
    than silently misaligning the mask.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="only supports a single \\(2-D\\) Triangle"):
        clrd.style.highlight_lower_triangle()
