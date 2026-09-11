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
    """Check that Styler is importable from the top-level chainladder package."""
    assert cl.Styler is Styler


def test_is_pandas_styler_subclass() -> None:
    """Check that Styler extends pandas' own Styler rather than replacing it."""
    assert issubclass(Styler, PandasStyler)


def test_wraps_a_dataframe(df: pd.DataFrame) -> None:
    """Check that Styler wraps a DataFrame the same way pandas' Styler does."""
    styler = Styler(df)
    assert styler.data is df


def test_chained_methods_preserve_subclass(df: pd.DataFrame) -> None:
    """Check that chaining a pandas Styler method still returns a chainladder Styler,
    not a plain pandas Styler -- the methods mutate self and return self, so
    subclassing doesn't get lost partway through a chain.
    """
    # pandas types .format()/.hide() as returning StylerRenderer (their defining
    # base class) rather than Self, so a type checker loses the subclass after
    # chaining even though the runtime object is still a Styler -- see the
    # isinstance assertion below.
    result = (
        Styler(df)
        .format(precision=1)
        .hide(  # pyright: ignore[reportAttributeAccessIssue]
            axis="columns", subset=["a"]
        )
    )
    assert isinstance(result, Styler)


def test_renders_identically_to_pandas_styler(df: pd.DataFrame) -> None:
    """Check that Styler produces the same HTML as pandas' own Styler, uuid aside --
    confirms the subclass doesn't alter behavior, matching the goal of starting as
    a pure adapter before adding Triangle-specific behavior.
    """
    # Same StylerRenderer-vs-Self typing gap as above, on pandas' own side too.
    cl_html = Styler(df, uuid="fixed").format(precision=2).to_html()  # pyright: ignore[reportAttributeAccessIssue]
    pd_html = PandasStyler(df, uuid="fixed").format(precision=2).to_html()  # pyright: ignore[reportAttributeAccessIssue]
    assert cl_html == pd_html


def test_triangle_style_is_a_property(raa) -> None:
    """Check that Triangle.style is a property, matching pandas.DataFrame.style,
    rather than a method that must be called.
    """
    assert isinstance(type(raa).style, property)


def test_triangle_style_returns_styler(raa) -> None:
    """Check that Triangle.style returns a chainladder Styler."""
    assert isinstance(raa.style, Styler)


def test_triangle_style_wraps_to_frame(raa) -> None:
    """Check that Triangle.style wraps the same data as Triangle.to_frame()."""
    pd.testing.assert_frame_equal(raa.style.data, raa.to_frame())


def test_triangle_style_is_fresh_each_access(raa) -> None:
    """Check that each access to Triangle.style returns a new Styler, matching
    pandas.DataFrame.style's own behavior -- so unrelated uses don't share state
    or a uuid.
    """
    assert raa.style is not raa.style


def test_highlight_lower_triangle_styles_exactly_the_lower_triangle(raa) -> None:
    """Check that every styled cell -- and only those cells -- correspond to a
    NaN in the Triangle's own nan_triangle mask.
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
    """Check that a custom props string is used verbatim instead of color."""
    styler = raa.style.highlight_lower_triangle(
        color="lightgray", props="background-color: blue; opacity: 50%;"
    )
    assert "opacity: 50%" in styler.to_html()
    assert "lightgray" not in styler.to_html()


def test_highlight_lower_triangle_returns_styler(raa) -> None:
    """Check that the subclass is preserved through the chained call."""
    assert isinstance(raa.style.highlight_lower_triangle(), Styler)


def test_highlight_lower_triangle_requires_a_triangle(df: pd.DataFrame) -> None:
    """Check that calling it on a Styler not built from Triangle.style raises a
    clear error, since there's no way to know which cells are the lower triangle.
    """
    with pytest.raises(
        ValueError, match="requires a Styler created from Triangle.style"
    ):
        Styler(df).highlight_lower_triangle()


def test_highlight_lower_triangle_requires_matching_shape(clrd) -> None:
    """Check that a Triangle whose to_frame() isn't a single origin-by-development
    grid (multiple index labels and columns, here) raises a clear error rather
    than silently misaligning the mask.
    """
    with pytest.raises(ValueError, match="only supports a single \\(2-D\\) Triangle"):
        clrd.style.highlight_lower_triangle()
