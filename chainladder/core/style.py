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

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core.typing import TriangleProtocol

del TYPE_CHECKING
del annotations


class Styler(_PandasStyler):
    """
    Styles a Triangle according to the data with HTML and CSS.

    This class provides methods for styling and formatting a Triangle. The
    styled output can be rendered as HTML or LaTeX, and it supports CSS-based styling, allowing
    users to control colors, font styles, and other visual aspects of tabular data. It is particularly
    useful for presenting Triangle objects in a Jupyter Notebook environment or when exporting
    styled triangles for reports.

    Parameters
    ----------
    data: pd.DataFrame | pd.Series
        The data to style, as for :class:`pandas.io.formats.style.Styler`.
    triangle: TriangleProtocol | None
        The Triangle ``data`` was produced from, e.g. via ``Triangle.to_frame()``.
        Required by builtins that need to know the Triangle's actuarial
        structure, such as :meth:`highlight_lower_triangle`. Optional, since a
        Styler can still be built directly from an arbitrary DataFrame the same
        way a pandas ``Styler`` can; those Triangle-specific builtins simply
        aren't available in that case.
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
        self, color: str = "blue", props: str | None = None
    ) -> Styler:
        """
        Highlight the lower (future, unobserved) triangle with a style.

        Parameters
        ----------
        color: str
            Background color applied to lower-triangle cells. Ignored if
            ``props`` is given. Defaults to "blue".
        props: str | None
            A full CSS properties string to apply instead of ``color``, e.g.
            ``"background-color: blue; opacity: 60%;"``. Optional.

        Returns
        -------
        Styler

        Examples
        --------

        .. testcode::
            :options: +SKIP

            import chainladder as cl

            cl.load_sample("raa").style.highlight_lower_triangle(color="lightgray")

        """
        if self._triangle is None:
            raise ValueError(
                "highlight_lower_triangle requires a Styler created from "
                "Triangle.style, so it knows which cells are the lower triangle."
            )
        nan_triangle = self._triangle.nan_triangle
        # Densify backends (sparse, cupy, ...) that don't support pd.isna() directly.
        nan_triangle = (
            nan_triangle.todense()  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(nan_triangle, "todense")
            else np.asarray(nan_triangle)
        )
        if nan_triangle.shape != self.data.shape:
            raise ValueError(
                "highlight_lower_triangle only supports a single (2-D) Triangle."
            )

        def f(_data: pd.DataFrame, props: str) -> np.ndarray:
            return np.where(pd.isna(nan_triangle), props, "")

        if props is None:
            props = f"background-color: {color};"
        return self.apply(f, axis=None, props=props)  # pyright: ignore[reportReturnType]
