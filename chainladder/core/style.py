"""
Styler for formatting Triangle output.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import copy as copy_module

from functools import partial

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler as _PandasStyler

from datetime import date, datetime
from typing import Any, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from chainladder import Triangle

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
    triangle: Triangle
        The Triangle to style.
    *args: Any
        Additional positional arguments passed to
        :class:`pandas.io.formats.style.Styler`.
    **kwargs: Any
        Additional keyword arguments passed to
        :class:`pandas.io.formats.style.Styler`.

    Raises
    ------
    TypeError
        If ``triangle`` is anything other than a Triangle.
    ValueError
        If ``triangle`` is multidimensional, i.e. holds more than a single index
        and column.

    Examples
    --------

    .. testcode::

        import chainladder as cl
        from chainladder.core.style import Styler

        raa = cl.load_sample("raa")
        print(Styler(raa.link_ratio).format(precision=1).to_string(), end="")

    .. testoutput::

         12-24 24-36 36-48 48-60 60-72 72-84 84-96 96-108 108-120
        1981 1.6 1.3 1.1 1.1 1.2 1.1 1.0 1.0 1.0
        1982 40.4 1.3 2.0 1.3 1.1 1.0 1.0 1.0 nan
        1983 2.6 1.5 1.2 1.2 1.2 1.0 1.0 nan nan
        1984 2.0 1.4 1.3 1.1 1.1 1.0 nan nan nan
        1985 8.8 1.7 1.4 1.2 1.0 nan nan nan nan
        1986 4.3 1.8 1.1 1.2 nan nan nan nan nan
        1987 7.2 2.7 1.1 nan nan nan nan nan nan
        1988 5.1 1.9 nan nan nan nan nan nan nan
        1989 1.7 nan nan nan nan nan nan nan nan

    Please see: :doc:`Triangle Visualization </user_guide/style>` for more
    examples.

    """

    def __init__(
        self,
        triangle: Triangle,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Imported here rather than at module scope: Triangle's own mixins import
        # this module, so a top-level import would be circular.
        from chainladder.core.triangle import Triangle

        if not isinstance(triangle, Triangle):
            # The frame case is the likely mistake, so point it somewhere useful.
            hint = (
                " Use DataFrame.style for a DataFrame."
                if isinstance(triangle, (pd.DataFrame, pd.Series))
                else ""
            )
            raise TypeError(
                "Styler must be created from a Triangle, not a "
                f"{type(triangle).__name__}.{hint}"
            )
        if triangle._dimensionality == "multi":
            raise ValueError(
                "Styler only supports a single Triangle. Select one "
                "index and column first, e.g. triangle.iloc[0, 0]."
            )
        data = triangle._repr_format(origin_as_datetime=False)
        super().__init__(data, *args, **kwargs)
        self._triangle = triangle
        self.format(triangle._get_format_str(data=data), na_rep="")

    def _copy(
        self,
        deepcopy: bool = False,
    ) -> Styler:
        """
        Copy the Styler. Overrides the parent Pandas Styler to be able to work on Triangles instead of a DataFrame.

        Parameters
        ----------
        deepcopy: bool
            If ``True``, deep-copy every attribute carried over from the calling Styler.

        Returns
        -------
        Styler
            A new copy of the ``Styler``.
        """
        styler = object.__new__(type(self))
        _PandasStyler.__init__(styler, self.data)
        styler._triangle = self._triangle
        for key, value in self.__dict__.items():
            if key in (
                "data",
                "index",
                "columns",
                "_triangle",
            ):
                continue
            setattr(
                styler,
                key,
                copy_module.deepcopy(value) if deepcopy else value,
            )
        return styler

    @staticmethod
    def _mask_style(
        _data: pd.DataFrame,
        mask: np.ndarray,
        props: str,
    ) -> np.ndarray:
        """
        Apply ``props`` to the cells ``mask`` selects, and leave every other
        cell unstyled.

        Parameters
        ----------
        _data: pd.DataFrame
            The styled DataFrame, as passed in by ``Styler.apply``. Unused --
            ``mask`` and ``props`` already carry everything needed to style --
            but required by ``apply``'s calling convention.
        mask: np.ndarray
            A 2-D boolean array, shaped like ``_data``, selecting the cells to
            style.
        props: str
            A CSS properties string applied to the selected cells, e.g.
            ``"background-color: blue;"``.

        Returns
        -------
        np.ndarray
            An array shaped like ``mask``, holding ``props`` at the cells it
            selects and ``""`` (unstyled) everywhere else.

        Example
        -------
        >>> mask_array = np.array([[False, True]])
        >>> Styler._mask_style(pd.DataFrame(), mask_array, "background-color: red;")
        array([['', 'background-color: red;']], dtype='<U22')
        """
        return np.where(mask, props, "")

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

        .. code-block:: python

            import chainladder as cl

            cl.load_sample("raa").style.highlight_lower_triangle(color="lightgray")

        A softer, higher-contrast pairing than the default:

        .. code-block:: python

            cl.load_sample("raa").style.highlight_lower_triangle(
                color="#BDD7EE", text_color="#1F4E78"
            )

        Highlighting a fully-predicted Triangle requires a
        valuation date, since its cells are no longer ``NaN``.

        .. code-block:: python

            raa = cl.load_sample("raa")
            full = cl.Chainladder().fit(raa).full_triangle_
            full.style.highlight_lower_triangle(
                color="lightgray",
                valuation_date=raa.valuation_date,
            )

        Please see: :doc:`Triangle Visualization </user_guide/style>` for more
        examples.
        """
        # Find which cells are beyond the valuation date. Does this by filling
        # a triangle's cells with their valuation dates and then comparing them to
        # the triangle's overall valuation date.
        if valuation_date is None:
            cutoff = self._triangle.valuation_date
        else:
            cutoff = pd.Timestamp(valuation_date)
        val_array = np.array(self._triangle.valuation).reshape(
            self._triangle.shape[-2:],
            order="F",
        )
        mask = val_array > cutoff
        if mask.shape != self.data.shape:
            raise ValueError(
                "highlight_lower_triangle only supports a single (2-D) Triangle."
            )

        if props is None:
            props = f"background-color: {color};"
            if text_color is not None:
                props += f" color: {text_color};"

        return self.apply(  # pyright: ignore[reportReturnType]
            partial(
                self._mask_style,
                mask=mask,
                props=props,
            ),
            axis=None,
        )
