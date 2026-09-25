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

from chainladder import options
from chainladder.core.triangle import Triangle

from datetime import (
    date,
    datetime,
)
from typing import (
    Any,
    Literal,
    TypeAlias,
)

# A value accepted by pandas.Timestamp's constructor.
ValuationDateLike: TypeAlias = int | float | str | date | datetime | pd.Timestamp

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
        and column, or if it is empty, i.e. holds no values to style.

    See Also
    --------
    Triangle.style : Returns a Styler for the Triangle.

    Examples
    --------

    .. testcode::

        import chainladder as cl

        raa = cl.load_sample("raa")
        raa.link_ratio.style.format(precision=1)

    Please see: :doc:`Triangle Visualization </user_guide/style>` for more
    examples.
    """

    def __init__(
        self,
        triangle: Triangle,
        *args: Any,
        **kwargs: Any,
    ) -> None:

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
        if triangle._dimensionality in ["multi", "empty"]:
            raise ValueError("Styler only supports a single Triangle.")
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

    def apply_from_triangle(
        self,
        mask: Triangle,
        color: str = "blue",
        text_color: str | None = None,
        props: str | None = None,
    ) -> Styler:
        """
        Apply CSS styles to cells where the supplied mask Triangle has
        non-missing (or truthy) values.

        Parameters
        ----------
        mask: Triangle
            A single (2-D) Triangle matching the shape of the styled Triangle.
            Cells where ``mask`` is not NaN (or True if boolean) will be styled.
        color: str
            Background color applied to selected cells. Ignored if ``props``
            is given. Defaults to "blue".
        text_color: str | None
            Text color applied to selected cells. Ignored if ``props``
            is given. Left unstyled (inherited) if not given.
        props: str | None
            A full CSS properties string to apply instead of ``color`` and
            ``text_color``. Optional.

        Returns
        -------
        Styler

        Raises
        ------
        TypeError
            If ``mask`` is not a Triangle instance.
        ValueError
            If the wrapped Triangle or ``mask`` is a valuation Triangle, or
            if ``mask`` does not match the 2-D shape of the styled Triangle.

        Examples
        --------
        .. code-block:: python

            import chainladder as cl

            raa = cl.load_sample("raa")
            # Highlight latest diagonal via valuation slicing
            raa.style.apply_from_triangle(
                raa[raa.valuation == raa.valuation_date], color="#FFE599"
            )
        """
        if not isinstance(mask, Triangle):
            raise TypeError("mask must be a Triangle instance.")

        if self._triangle.is_val_tri or mask.is_val_tri:
            raise ValueError(
                "apply_from_triangle does not support a valuation Triangle."
            )

        if (
            mask._dimensionality in ["multi", "empty"]
            or mask.shape[-2:] != self.data.shape
        ):
            raise ValueError(
                "apply_from_triangle only supports a single (2-D) Triangle "
                "matching the shape of the styled Triangle."
            )

        vals = mask.values
        if hasattr(vals, "compute"):
            vals = vals.compute()
        if hasattr(vals, "todense"):
            vals = vals.todense()
        elif hasattr(vals, "get"):
            vals = vals.get()

        mask_vals = np.asarray(vals).reshape(mask.shape[-2:])
        if np.issubdtype(mask_vals.dtype, np.bool_):
            bool_mask = mask_vals
        else:
            bool_mask = ~np.isnan(mask_vals)

        if props is None:
            props = f"background-color: {color};"
            if text_color is not None:
                props += f" color: {text_color};"

        return self.apply(  # pyright: ignore[reportReturnType]
            partial(
                self._mask_style,
                mask=bool_mask,
                props=props,
            ),
            axis=None,
        )

    def highlight_lower_triangle(
        self,
        color: str = "blue",
        text_color: str | None = None,
        props: str | None = None,
        valuation_date: ValuationDateLike | None = None,
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
            The "as of" date used to determine the diagonal beyond which the
            cells are highlighted. If ``None``, defaults to the
            lastest diagonal.
        text_color: str | None
            Text color applied to lower-triangle cells. Ignored if ``props``
            is given. Left unstyled (i.e. inherited) if not given.

        Returns
        -------
        Styler

        Raises
        ------
        ValueError
            If the wrapped Triangle is a valuation Triangle.

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

        A fully-predicted Triangle needs no valuation date, even though its
        cells are no longer ``NaN``.

        .. code-block:: python

            raa = cl.load_sample("raa")
            full = cl.Chainladder().fit(raa).full_triangle_
            full.style.highlight_lower_triangle(color="lightgray")
        """
        if self._triangle.is_val_tri:
            raise ValueError(
                "highlight_lower_triangle does not support a valuation Triangle."
            )

        # Find which cells are beyond the valuation date. Does this by filling
        # a triangle's cells with their valuation dates and then comparing them to
        # the triangle's overall valuation date.
        val_array = np.array(self._triangle.valuation).reshape(
            self._triangle.shape[-2:],
            order="F",
        )
        if valuation_date is not None:
            cutoff = pd.Timestamp(valuation_date)
        elif self._triangle.valuation_date >= pd.Timestamp(options.ULT_VAL):
            cutoff = pd.Timestamp(val_array[-1, 0])
        else:
            cutoff = self._triangle.valuation_date
        mask = val_array > cutoff

        mask_tri = self._triangle.copy()
        mask_tri.values = np.where(mask, 1.0, np.nan)[None, None, :, :]
        return self.apply_from_triangle(
            mask_tri,
            color=color,
            text_color=text_color,
            props=props,
        )

    def highlight_diagonal(
        self,
        color: str = "#FFE599",
        text_color: str | None = None,
        props: str | None = None,
        valuation: ValuationDateLike | Literal["latest"] = "latest",
        **kwargs: Any,
    ) -> Styler:
        """
        Highlight a diagonal -- the cells corresponding to a specific
        valuation date -- with a style.

        Parameters
        ----------
        color: str
            Background color applied to diagonal cells. Ignored if
            ``props`` is given. Defaults to "#FFE599".
        text_color: str | None
            Text color applied to diagonal cells. Ignored if ``props``
            is given. Left unstyled (i.e. inherited) if not given.
        props: str | None
            A full CSS properties string to apply instead of ``color`` and
            ``text_color``, e.g. ``"background-color: blue; opacity: 60%;"``.
            Optional.
        valuation: ValuationDateLike | Literal["latest"]
            The valuation date of the diagonal to highlight. Can be ``"latest"``
            to highlight the most recent observed diagonal, or a specific date
            (such as a year, date string, or datetime). Defaults to ``"latest"``.
        **kwargs: Any
            Additional keyword arguments. Supports ``valuation_date`` as an alias
            for ``valuation``.

        Returns
        -------
        Styler

        Raises
        ------
        ValueError
            If the wrapped Triangle is a valuation Triangle, or if the specified
            valuation date is not present in the Triangle.

        Examples
        --------

        .. code-block:: python

            import chainladder as cl

            cl.load_sample("raa").style.highlight_diagonal(color="lightyellow")

        Highlight a specific historical valuation diagonal:

        .. code-block:: python

            cl.load_sample("raa").style.highlight_diagonal(
                valuation="1988", color="#FFE599"
            )

        Chaining diagonal and lower triangle highlighting:

        .. code-block:: python

            (
                cl
                .load_sample("raa")
                .style.highlight_diagonal(color="#FFF2CC")
                .highlight_lower_triangle(color="#DDEBF7")
            )
        """
        if "valuation_date" in kwargs:
            valuation = kwargs.pop("valuation_date")
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"highlight_diagonal() got an unexpected keyword argument '{unexpected}'"
            )

        if self._triangle.is_val_tri:
            raise ValueError(
                "highlight_diagonal does not support a valuation Triangle."
            )

        val_array = np.array(self._triangle.valuation).reshape(
            self._triangle.shape[-2:],
            order="F",
        )
        if valuation == "latest" or valuation is None:
            if self._triangle.valuation_date >= pd.Timestamp(options.ULT_VAL):
                cutoff = pd.Timestamp(val_array[-1, 0])
            else:
                cutoff = self._triangle.valuation_date
            mask = val_array == cutoff
        else:
            dev_freq = self._triangle.development_grain.replace("S", "2Q")
            try:
                target_period = pd.Period(valuation, freq=dev_freq)
            except (ValueError, TypeError):
                try:
                    target_period = pd.Period(pd.Timestamp(valuation), freq=dev_freq)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid valuation date: '{valuation}'") from e
            val_periods = self._triangle.valuation.to_period(
                freq=dev_freq
            ).values.reshape(self._triangle.shape[-2:], order="F")
            mask = val_periods == target_period

        if not mask.any():
            raise ValueError(f"Valuation '{valuation}' not found in Triangle.")

        mask_tri = self._triangle.copy()
        mask_tri.values = np.where(mask, 1.0, np.nan)[None, None, :, :]
        return self.apply_from_triangle(
            mask_tri,
            color=color,
            text_color=text_color,
            props=props,
        )
