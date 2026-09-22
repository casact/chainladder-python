# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from datetime import datetime

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


@pytest.mark.parametrize(
    "obj", [np.array([1]), [1], None, "raa", 3, pd.Series([1])], ids=type
)
def test_rejects_anything_that_is_not_a_triangle(obj) -> None:
    """
    Check that the constructor rejects every non-Triangle, not just frames, so a
    Styler always has the Triangle its methods rely on.

    Parameters
    ----------
    obj: object
        A value that is not a Triangle.

    Returns
    -------
    None

    """
    with pytest.raises(TypeError, match="must be created from a Triangle"):
        Styler(obj)


def test_rejection_names_the_type_it_received(df: pd.DataFrame) -> None:
    """
    Check that the error names the offending type, and points DataFrame users at
    pandas' own styler.

    Parameters
    ----------
    df: pd.DataFrame
        A simple two-column DataFrame fixture.

    Returns
    -------
    None

    """
    with pytest.raises(TypeError, match="not a DataFrame. Use DataFrame.style"):
        Styler(df)
    with pytest.raises(TypeError, match="not a list.$"):
        Styler([1])


def test_wraps_a_triangles_frame(raa) -> None:
    """
    Check that Styler wraps the Triangle's frame representation, the way pandas'
    Styler wraps the DataFrame it is given.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    pd.testing.assert_frame_equal(
        Styler(raa).data, raa.to_frame(origin_as_datetime=False)
    )


def test_chained_methods_preserve_subclass(raa) -> None:
    """
    Check that chaining a pandas Styler method still returns a chainladder Styler,
    not a plain pandas Styler.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    result = (
        Styler(raa)
        .format(precision=1)
        .hide(  # pyright: ignore[reportAttributeAccessIssue]
            axis="columns", subset=[raa.development[0]]
        )
    )
    assert isinstance(result, Styler)


def test_renders_identically_to_pandas_styler(raa) -> None:
    """
    Check that Styler produces the same HTML as pandas' own Styler wrapping the
    same frame, once pandas is given the default numeric format that Styler
    applies to a Triangle on construction.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    frame = raa.to_frame(origin_as_datetime=False)
    cl_html = Styler(raa, uuid="fixed").to_html()
    pd_html = (
        PandasStyler(frame, uuid="fixed")
        .format(raa._get_format_str(data=frame), na_rep="")
        .to_html()
    )
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


def test_highlight_lower_triangle_predicted_cells_highlighted_by_default(
    raa,
) -> None:
    """
    A fully-predicted Triangle (e.g. full_triangle_) has no NaN cells left and
    carries the ultimate sentinel as its valuation_date, which would select no
    cells. Its latest diagonal is taken from the last origin period instead, so
    the predicted cells are highlighted without an explicit valuation_date, and
    identically to passing the source Triangle's own valuation date.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    full = cl.Chainladder().fit(raa).full_triangle_

    inferred = full.style.highlight_lower_triangle(color="lightgray")
    inferred._compute()
    assert any(inferred.ctx.values())

    explicit = full.style.highlight_lower_triangle(
        color="lightgray",
        valuation_date=raa.valuation_date,
    )
    explicit._compute()
    assert inferred.ctx == explicit.ctx


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


def test_style_rejects_multidimensional_triangle(clrd) -> None:
    """
    Check that a Triangle holding more than a single index and column raises a
    clear error at construction, rather than silently styling the long frame
    that to_frame() flattens it into.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    assert clrd._dimensionality == "multi"
    with pytest.raises(ValueError, match="only supports a single"):
        _ = clrd.style


def test_style_rejects_empty_triangle() -> None:
    """
    Check that styling an empty Triangle raises an error.

    Returns
    -------
    None

    """
    empty = cl.Triangle()
    assert empty._dimensionality == "empty"
    with pytest.raises(ValueError, match="only supports a single"):
        _ = empty.style


def test_style_accepts_single_triangle_selected_from_multidimensional(clrd) -> None:
    """
    Check that the error the multidimensional case raises is escapable by
    selecting a single index and column, as its message advises.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    single = clrd.iloc[0, 0]
    assert single._dimensionality == "single"
    assert isinstance(single.style, Styler)


def test_highlight_lower_triangle_rejects_valuation_triangle(raa) -> None:
    """
    Check that a valuation Triangle is refused outright. It is indexed by
    calendar date, so it holds nothing beyond its own valuation date and has no
    lower triangle to highlight -- its empty corner is the lower left, the
    origin periods that had not begun yet.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="does not support a valuation Triangle"):
        raa.dev_to_val().style.highlight_lower_triangle()


def test_highlight_lower_triangle_rejects_fully_developed_valuation_triangle(
    raa,
) -> None:
    """
    Check that a fully developed Triangle in valuation mode is refused too. It
    carries the ultimate sentinel as its valuation_date, so it reaches the
    branch that infers a cutoff from the last origin period -- an inference that
    does not hold once the axes are calendar dates.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    full = cl.Chainladder().fit(raa).full_triangle_
    with pytest.raises(ValueError, match="does not support a valuation Triangle"):
        full.dev_to_val().style.highlight_lower_triangle()


def test_highlight_diagonal_styles_latest_diagonal_by_default(raa) -> None:
    """
    Check that highlight_diagonal styles the latest diagonal by default.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    styler = raa.style.highlight_diagonal(color="lightyellow")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}

    val_array = np.array(raa.valuation).reshape(raa.shape[-2:], order="F")
    expected = {
        (r, c)
        for r, row in enumerate(val_array == raa.valuation_date)
        for c, is_diag in enumerate(row)
        if is_diag
    }
    assert styled == expected
    assert len(styled) == len(raa.origin)
    assert all(
        v == [("background-color", "lightyellow")] for v in styler.ctx.values() if v
    )


@pytest.mark.parametrize(
    "val_arg",
    [
        "1988",
        "1988-12-31",
        1988,
        datetime(1988, 12, 31),
        pd.Timestamp("1988-12-31"),
    ],
)
def test_highlight_diagonal_explicit_dates(raa, val_arg) -> None:
    """
    Check that highlight_diagonal accepts different date representations for
    historical diagonals.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.
    val_arg: object
        Valuation date argument in different formats.

    Returns
    -------
    None
    """
    styler = raa.style.highlight_diagonal(valuation=val_arg, color="yellow")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}

    target_period = pd.Period("1988", freq=raa.origin.freq)
    val_periods = raa.valuation.to_period(raa.origin.freq).values.reshape(
        raa.shape[-2:], order="F"
    )
    expected = {
        (r, c)
        for r, row in enumerate(val_periods == target_period)
        for c, is_diag in enumerate(row)
        if is_diag
    }
    assert styled == expected
    assert len(styled) == 8


def test_highlight_diagonal_valuation_date_alias(raa) -> None:
    """
    Check that valuation_date keyword argument works as an alias for valuation.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    styler = raa.style.highlight_diagonal(valuation_date="1988")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}
    assert len(styled) == 8


def test_highlight_diagonal_predicted_cells_full_triangle(raa) -> None:
    """
    Check that a fully-predicted Triangle defaults to highlighting the latest
    observed diagonal, and can highlight future diagonals when explicitly requested.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    full = cl.Chainladder().fit(raa).full_triangle_
    inferred = full.style.highlight_diagonal(color="yellow")
    inferred._compute()

    explicit = full.style.highlight_diagonal(
        color="yellow", valuation=raa.valuation_date
    )
    explicit._compute()
    assert inferred.ctx == explicit.ctx

    future = full.style.highlight_diagonal(valuation="1995")
    future._compute()
    styled_future = {k for k, v in future.ctx.items() if v}
    assert len(styled_future) > 0


def test_highlight_diagonal_props_overrides_color(raa) -> None:
    """
    Check that props overrides color.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    styler = raa.style.highlight_diagonal(
        color="lightgray", props="background-color: green; font-weight: bold;"
    )
    assert "font-weight: bold" in styler.to_html()
    assert "lightgray" not in styler.to_html()


def test_highlight_diagonal_text_color(raa) -> None:
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
    styler = raa.style.highlight_diagonal(color="#FFF2CC", text_color="#7F6000")
    styler._compute()
    styled = [v for v in styler.ctx.values() if v]
    assert len(styled) > 0
    assert all(
        v == [("background-color", "#FFF2CC"), ("color", "#7F6000")] for v in styled
    )


def test_highlight_diagonal_chaining_with_lower_triangle(raa) -> None:
    """
    Check that highlight_diagonal can be chained with highlight_lower_triangle.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    styler = raa.style.highlight_diagonal(color="yellow").highlight_lower_triangle(
        color="gray"
    )
    assert isinstance(styler, Styler)
    styler._compute()
    assert any(v == [("background-color", "yellow")] for v in styler.ctx.values())
    assert any(v == [("background-color", "gray")] for v in styler.ctx.values())


def test_highlight_diagonal_invalid_valuation_raises(raa) -> None:
    """
    Check that an invalid or missing valuation date raises ValueError.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="not found in Triangle"):
        raa.style.highlight_diagonal(valuation="1970")

    with pytest.raises(ValueError, match="Invalid valuation date"):
        raa.style.highlight_diagonal(valuation="invalid-valuation-string")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        raa.style.highlight_diagonal(invalid_arg=123)


def test_highlight_diagonal_rejects_valuation_triangle(raa) -> None:
    """
    Check that a valuation Triangle raises ValueError when highlighting diagonal.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="does not support a valuation Triangle"):
        raa.dev_to_val().style.highlight_diagonal()


def test_highlight_diagonal_finer_development_grain() -> None:
    """
    Check that highlight_diagonal highlights exactly a single diagonal when
    development grain is finer than origin grain (e.g. annual origin,
    quarterly development).

    Returns
    -------
    None
    """
    tri = cl.load_sample("quarterly")["paid"]
    styler = tri.style.highlight_diagonal(valuation="1995-03-31", color="yellow")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}
    assert len(styled) == 1
    assert styled == {(0, 0)}

    # Check a valuation date spanning multiple origins
    styler2 = tri.style.highlight_diagonal(valuation="1998-03-31", color="yellow")
    styler2._compute()
    styled2 = {k for k, v in styler2.ctx.items() if v}
    assert len(styled2) == 4
    rows = [r for r, c in styled2]
    assert len(rows) == len(set(rows))
