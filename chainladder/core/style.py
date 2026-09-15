"""
Styler for formatting Triangle output.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler as _PandasStyler

from datetime import date, datetime
from typing import Any, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from chainladder.core.typing import TriangleProtocol

# A value accepted by pandas.Timestamp's constructor.
ValuationDateLike: TypeAlias = int | float | str | date | datetime | pd.Timestamp

del TYPE_CHECKING
del annotations


class Styler(_PandasStyler):
    """
    Styles a Triangle according to the data with HTML and CSS.

    This class provides methods for styling and formatting a Triangle. The
    styled output can be rendered as HTML or LaTeX, and it supports CSS-based styling, allowing
    users to control colors, font styles, and other visual aspects of triangular data. It is particularly
    useful for presenting Triangle objects in a Jupyter Notebook environment or when exporting
    styled triangles for reports.

    Parameters
    ----------
    data: pd.DataFrame | pd.Series
        The data to style, as for :class:`pandas.io.formats.style.Styler`.
    triangle: TriangleProtocol | None
        The Triangle ``data`` was produced from, e.g. via ``Triangle.to_frame()``.
    *args: Any
        Additional positional arguments passed to
        :class:`pandas.io.formats.style.Styler`.
    **kwargs: Any
        Additional keyword arguments passed to
        :class:`pandas.io.formats.style.Styler`.

    Examples
    --------

    .. testcode::

        import pandas as pd
        from chainladder.core.style import Styler

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        Styler(df).format(precision=1)

    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        *args: Any,
        triangle: TriangleProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data, *args, **kwargs)
        self._triangle = triangle

    def highlight_lower_triangle(
        self,
        color: str = "blue",
        props: str | None = None,
        valuation_date: ValuationDateLike | None = None,
        text_color: str | None = None,
    ) -> Styler:
        """
        Highlight the lower triangle -- the cells beyond the Triangle's
        latest diagonal -- with a style.

        Parameters
        ----------
        color: str
            Background color applied to lower-triangle cells. Ignored if
            ``props`` is given. Defaults to "blue".
        props: str | None
            A full CSS properties string to apply instead of ``color`` and
            ``text_color``, e.g. ``"background-color: blue; opacity: 60%;"``.
            Optional.
        valuation_date: ValuationDateLike | None
            The "as of" date used to determine which cells fall beyond the
            latest diagonal, i.e. the lower triangle. If ``None``, defaults to the
            wrapped Triangle's own ``valuation_date``.
        text_color: str | None
            Text color applied to lower-triangle cells. Ignored if ``props``
            is given. Left unstyled (i.e. inherited) if not given.

        Returns
        -------
        Styler

        Examples
        --------

        .. testcode::
            :options: +SKIP

            import chainladder as cl

            cl.load_sample("raa").style.highlight_lower_triangle(color="lightgray")

        A softer, higher-contrast pairing than the default:

        .. testcode::
            :options: +SKIP

            cl.load_sample("raa").style.highlight_lower_triangle(
                color="#BDD7EE", text_color="#1F4E78"
            )

        Highlighting a fully-predicted Triangle requires a
        valuation date, since its cells are no longer ``NaN``.

        .. testcode::
            :options: +SKIP

            raa = cl.load_sample("raa")
            full = cl.Chainladder().fit(raa).full_triangle_
            full.style.highlight_lower_triangle(
                color="lightgray", valuation_date=raa.valuation_date
            )

        """
        if self._triangle is None:
            raise ValueError(
                "highlight_lower_triangle requires a Styler created from "
                "Triangle.style, so it knows which cells are the lower triangle."
            )
        if valuation_date is None:
            cutoff = self._triangle.valuation_date
        else:
            cutoff = pd.Timestamp(valuation_date)
        val_array = np.array(self._triangle.valuation).reshape(
            self._triangle.shape[-2:], order="F"
        )
        nan_triangle = np.where(val_array > cutoff, np.nan, 1)
        if nan_triangle.shape != self.data.shape:
            raise ValueError(
                "highlight_lower_triangle only supports a single (2-D) Triangle."
            )

        if props is None:
            props = f"background-color: {color};"
            if text_color is not None:
                props += f" color: {text_color};"

        def f(_data: pd.DataFrame) -> np.ndarray:
            return np.where(pd.isna(nan_triangle), props, "")

        return self.apply(f, axis=None)  # pyright: ignore[reportReturnType]
