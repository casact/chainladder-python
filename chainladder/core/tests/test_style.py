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
    styler_alias = raa.style.highlight_diagonal(valuation_date="1988")
    styler_direct = raa.style.highlight_diagonal(valuation="1988")
    styler_alias._compute()
    styler_direct._compute()
    assert styler_alias.ctx == styler_direct.ctx
    assert len([v for v in styler_alias.ctx.values() if v]) == 8


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


def test_apply_from_triangle_styles_selected_cells(raa) -> None:
    """
    Check that apply_from_triangle highlights cells that are not NaN.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    mask_tri = raa.copy()
    mask = np.zeros(raa.shape[-2:], dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask_tri.values = np.where(mask, 1.0, np.nan)[None, None, :, :]

    styler = raa.style.apply_from_triangle(mask_tri, color="yellow")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}
    assert styled == {(0, 0), (1, 1)}


def test_apply_from_triangle_validation(raa) -> None:
    """
    Check input validation on apply_from_triangle.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    with pytest.raises(TypeError, match="mask must be a Triangle instance"):
        raa.style.apply_from_triangle(pd.DataFrame())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="only supports a single"):
        raa.style.apply_from_triangle(cl.load_sample("clrd"))

    mismatched = raa.iloc[:, :, :5, :5]
    with pytest.raises(ValueError, match="only supports a single"):
        raa.style.apply_from_triangle(mismatched)


def test_apply_from_triangle_valuation_triangle(raa) -> None:
    """
    Check that apply_from_triangle works on a valuation Triangle.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    val = raa.dev_to_val()
    mask = val.copy()
    mask_arr = np.zeros(val.shape[-2:], dtype=bool)
    mask_arr[0, :] = True
    mask.values = np.where(mask_arr, 1.0, np.nan)[None, None, :, :]
    styler = val.style.apply_from_triangle(mask, color="yellow")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}
    assert styled == {(0, c) for c in range(val.shape[-1])}


def test_apply_from_triangle_sparse_mask(raa) -> None:
    """
    Check that apply_from_triangle works when mask has a sparse backend.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    sparse_mask = raa[raa.valuation == raa.valuation_date].set_backend("sparse")
    styler = raa.style.apply_from_triangle(sparse_mask, color="yellow")
    styler._compute()
    styled = {k for k, v in styler.ctx.items() if v}
    assert len(styled) == 10

    bool_mask_tri = raa.copy()
    b = np.zeros(raa.shape, dtype=bool)
    b[0, 0, 0, 0] = True
    bool_mask_tri.values = b
    sparse_bool = bool_mask_tri.set_backend("sparse")
    styler_bool = raa.style.apply_from_triangle(sparse_bool, color="yellow")
    styler_bool._compute()
    styled_bool = {k for k, v in styler_bool.ctx.items() if v}
    assert styled_bool == {(0, 0)}


def test_concat_identical_columns(raa) -> None:
    """
    Check that concatenating stylers with identical columns succeeds.
    """
    dev = cl.Development().fit(raa)
    s1 = raa.link_ratio.style
    s2 = dev.ldf_.style
    res = s1.concat(s2)
    assert res is s1
    assert len(s1.concatenated) == 1
    assert "foot0_row0" in s1.to_html()


def test_concat_triangle_and_link_ratio(raa) -> None:
    """
    Check that concatenating a Triangle with its link ratios pads the missing
    column on the link ratios with blanks and preserves caller immutability.
    """
    s1 = raa.style
    lr_styler = raa.link_ratio.style
    orig_cols = list(lr_styler.data.columns)

    res = s1.concat(lr_styler)
    assert res is s1
    assert len(s1.concatenated) == 1

    # Caller's other styler data was not mutated
    assert list(lr_styler.data.columns) == orig_cols

    # Rendered table has triangle columns
    html = s1.to_html()
    for col in s1.data.columns:
        assert f">{col}<" in html


def test_concat_link_ratio_and_cdf(raa) -> None:
    """
    Check that concatenating link ratios with CDF patterns (different column labels
    like '12-24' vs '12-Ult') succeeds due to compatible development periods.
    """
    dev = cl.Development().fit(raa)
    s1 = raa.link_ratio.style.format(precision=3, na_rep="")
    s2 = dev.cdf_.style.format(precision=3, na_rep="")
    res = s1.concat(s2)
    assert res is s1
    assert len(s1.concatenated) == 1
    html = s1.to_html()
    assert "foot0_row0" in html


def test_concat_multiple_chaining(raa) -> None:
    """
    Check that chaining multiple concats (Triangle + link ratios + ldf + cdf) works.
    """
    dev = cl.Development().fit(raa)
    s = (
        raa.style
        .concat(raa.link_ratio.style.format(precision=4, na_rep=""))
        .concat(dev.ldf_.style.format(precision=4, na_rep=""))
        .concat(dev.cdf_.style.format(precision=4, na_rep=""))
    )
    assert len(s.concatenated) == 3
    html = s.to_html()
    assert "foot0_row0" in html
    assert "foot1_row0" in html
    assert "foot2_row0" in html


def test_concat_widens_self_with_tail(raa) -> None:
    """
    Check that concatenating an other styler with additional tail columns
    widens self.data and pads prior rows with blanks.
    """
    pipe = cl.Pipeline([
        ("dev", cl.Development()),
        ("tail", cl.TailConstant(1.05)),
    ]).fit(raa)
    tail_step = pipe.named_steps.tail

    s1 = raa.link_ratio.style.format(precision=4, na_rep="")
    orig_len = len(s1.data.columns)
    s2 = tail_step.ldf_.style.format(precision=4, na_rep="")
    tail_len = len(s2.data.columns)
    assert tail_len > orig_len

    s1.concat(s2)
    assert len(s1.data.columns) == tail_len
    # Check that to_string and to_html render without error
    text = s1.to_string()
    assert len(text) > 0
    html = s1.to_html()
    assert "foot0_row0" in html


def test_concat_widening_updates_prior_concatenated(raa) -> None:
    """
    Check that when a tail styler widens self, all previously concatenated
    stylers are also widened to match the new column structure.
    """
    pipe = cl.Pipeline([
        ("dev", cl.Development()),
        ("tail", cl.TailConstant(1.05)),
    ]).fit(raa)
    tail_step = pipe.named_steps.tail

    dev = cl.Development().fit(raa)
    s1 = raa.link_ratio.style.format(precision=4, na_rep="")
    s2 = dev.ldf_.style.format(precision=4, na_rep="")
    s3 = tail_step.ldf_.style.format(precision=4, na_rep="")

    s = s1.concat(s2).concat(s3)
    assert len(s.concatenated) == 2
    # Verify s2 copy in s.concatenated was widened to match s.data.columns
    assert len(s.concatenated[0].data.columns) == len(s.data.columns)
    assert len(s.concatenated[1].data.columns) == len(s.data.columns)
    assert len(s.to_html()) > 0


def test_concat_preserves_styles(raa) -> None:
    """
    Check that styles applied to self and other (e.g. highlight_diagonal)
    are both preserved in the concatenated HTML output.
    """
    s1 = raa.style.highlight_diagonal(color="#FFF2CC")
    s2 = raa.link_ratio.style.highlight_diagonal(color="#DDEBF7")
    s = s1.concat(s2)
    html = s.to_html()
    assert "#FFF2CC" in html
    assert "#DDEBF7" in html


def test_concat_preserves_lower_triangle_styling(raa) -> None:
    """
    Check that lower triangle highlights are preserved when concatenated.
    """
    s1 = raa.style.highlight_lower_triangle(color="#BDD7EE")
    s2 = raa.link_ratio.style.highlight_lower_triangle(color="#F8CBAD")
    s = s1.concat(s2)
    html = s.to_html()
    assert "#BDD7EE" in html
    assert "#F8CBAD" in html


def test_concat_rejects_raw_triangle(raa) -> None:
    """
    Check that passing a raw Triangle raises TypeError with a helpful hint.
    """
    with pytest.raises(
        TypeError, match="must be of type `Styler`. Use `Triangle.style`"
    ):
        raa.style.concat(raa.link_ratio)


def test_concat_rejects_non_styler(raa) -> None:
    """
    Check that passing an invalid type raises TypeError.
    """
    with pytest.raises(TypeError, match="must be of type `Styler`"):
        raa.style.concat([1, 2, 3])


def test_concat_rejects_mismatched_index_levels(raa) -> None:
    """
    Check that mismatched index levels raises ValueError.
    """
    df_multi = pd.DataFrame(
        [[1] * len(raa.style.data.columns)],
        index=pd.MultiIndex.from_tuples([("A", "B")]),
        columns=raa.style.data.columns,
    )
    with pytest.raises(ValueError, match="number of index levels"):
        raa.style.concat(df_multi.style)


def test_concat_rejects_incompatible_grain(raa) -> None:
    """
    Check that triangles with different development grains raise ValueError.
    """
    quarterly = cl.load_sample("quarterly")["paid"]
    with pytest.raises(ValueError, match="Development grains must match"):
        raa.style.concat(quarterly.style)


def test_concat_rejects_incompatible_lags(raa) -> None:
    """
    Check that triangles with incompatible development lags raise ValueError.
    """
    t1 = raa.copy()
    t2 = raa.copy()
    t2.development = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    with pytest.raises(ValueError, match="Development periods must be compatible"):
        t1.style.concat(t2.style)


def test_concat_rejects_valuation_with_development(raa) -> None:
    """
    Check that mixing valuation and development mode triangles raises ValueError.
    """
    with pytest.raises(
        ValueError, match="valuation Triangle with a development Triangle"
    ):
        raa.dev_to_val().style.concat(raa.style)


def test_concat_output_formats(raa) -> None:
    """
    Check that concatenated Styler supports to_html, to_latex, and to_string.
    """
    s = raa.style.concat(raa.link_ratio.style.format(precision=3, na_rep=""))
    assert len(s.to_html()) > 0
    assert len(s.to_string()) > 0
    assert len(s.to_latex()) > 0


def test_concat_copy_preserves_concatenated(raa) -> None:
    """
    Check that _copy on a concatenated Styler deep-copies the concatenated list.
    """
    s = raa.style.concat(raa.link_ratio.style)
    s_copy = s._copy(deepcopy=True)
    assert isinstance(s_copy, Styler)
    assert len(s_copy.concatenated) == 1
    assert s_copy.concatenated is not s.concatenated
    assert len(s_copy.to_html()) > 0
