# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
import re

from typing import TYPE_CHECKING, Any

try:
    from IPython.core.display import HTML
    import IPython.display
except ImportError:
    HTML = None
    IPython = None

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        IndexSlice,
        Series
    )

class TriangleDisplay:

    def __repr__(self) -> str | DataFrame:

        # If values hasn't been defined yet, return an empty triangle.
        if self._dimensionality == 'empty':
            return "Empty Triangle."

        # For triangles with a single segment, containing a single triangle, return the
        # DataFrame of the values.
        elif self._dimensionality == 'single':
            data: DataFrame = self._repr_format()
            return data.to_string()

        # For multidimensional triangles, return a summary.
        else:
            return self._summary_frame().__repr__()

    def _summary_frame(self) -> DataFrame:
        """
        Returns summary information about the triangle. Used in the case of multidimensional triangles.

        Returns
        -------

        DataFrame
        """
        return pd.Series(
            data=[
                self.valuation_date.strftime("%Y-%m"),
                "O" + self.origin_grain + "D" + self.development_grain,
                self.shape,
                self.key_labels,
                list(self.vdims),
            ],
            index=["Valuation:", "Grain:", "Shape:", "Index:", "Columns:"],
            name="Triangle Summary",
        ).to_frame()

    def _repr_html_(self) -> str:
        """
        Jupyter/Ipython HTML representation.

        Returns
        -------
        str
        """

        # Case empty triangle.
        if self._dimensionality == 'empty':
            return "Empty Triangle."

        # Case single-dimensional triangle.
        elif self._dimensionality == 'single':
            data = self._repr_format()
            fmt_str = self._get_format_str(data=data)
            default = (
                data.to_html(
                    max_rows=pd.options.display.max_rows,
                    max_cols=pd.options.display.max_columns,
                    float_format=fmt_str.format,
                )
                .replace("nan", "")
                .replace("NaN", "")
            )
            return default
        # Case multidimensional triangle.
        else:
            return self._summary_frame().to_html(
                max_rows=pd.options.display.max_rows,
                max_cols=pd.options.display.max_columns,
            )

    @staticmethod
    def _get_format_str(data: DataFrame) -> str:
        """
        Returns a numerical format string based on the magnitude of the mean absolute value of the values in the
        supplied DataFrame.

        Returns
        -------
        str
        """
        if np.all(np.isnan(data)):
            return ""
        elif np.nanmean(abs(data)) < 10:
            return "{0:,.4f}"
        elif np.nanmean(abs(data)) < 1000:
            return "{0:,.2f}"
        else:
            return "{:,.0f}"

    def _repr_format(
            self,
            origin_as_datetime: bool = False
    ) -> DataFrame:
        """
        Prepare triangle values for printing as a DataFrame. Mainly used with single-dimensional triangles.

        Returns
        -------
        DataFrame
        """
        out: np.ndarray = self.compute().set_backend("numpy").values[0, 0]
        if origin_as_datetime and not self.is_pattern:
            origin: Series = self.origin.to_timestamp(how="s")
        else:
            origin = self.origin.copy()
        origin.name = None

        if self.origin_grain == "S" and not origin_as_datetime:
            origin_formatted = [""] * len(origin)
            for origin_index in range(len(origin)):
                origin_formatted[origin_index] = (
                    origin.astype("str")[origin_index]
                    .replace("Q1", "H1")
                    .replace("Q3", "H2")
                )
            origin = origin_formatted
        development = self.development.copy()
        development.name = None
        return pd.DataFrame(out, index=origin, columns=development)

    def heatmap(
            self,
            cmap: str = "coolwarm",
            low: float = 0,
            high: float = 0,
            axis: int | str = 0,
            subset: IndexSlice=None
    ) -> Any:
        """
        Color the background in a gradient according to the data in each
        column (optionally row). Requires matplotlib.

        Parameters
        ----------

        cmap : str or colormap
            matplotlib colormap
        low, high : float
            compress the range by these values.
        axis : int or str
            The axis along which to apply heatmap
        subset : IndexSlice
            a valid slice for data to limit the style application to

        Returns
        -------
            Ipython.display.HTML

        """
        if self._dimensionality == 'single':
            data = self._repr_format()
            fmt_str = self._get_format_str(data)

            axis = self._get_axis(axis)

            raw_rank = data.rank(axis=axis)
            shape_size = data.shape[axis]
            rank_size = data.rank(axis=axis).max(axis=axis)
            gmap = (raw_rank - 1).div(rank_size - 1, axis=not axis) * (
                shape_size - 1
            ) + 1
            gmap = gmap.replace(np.nan, (shape_size + 1) / 2)
            if pd.__version__ >= "1.3":
                default_output = (
                    data.style.format(fmt_str)
                    .background_gradient(
                        cmap=cmap,
                        low=low,
                        high=high,
                        axis=None,
                        subset=subset,
                        gmap=gmap,
                    )
                    .to_html()
                )
            else:
                default_output = (
                    data.style.format(fmt_str)
                    .background_gradient(
                        cmap=cmap,
                        low=low,
                        high=high,
                        axis=axis,
                    )
                    .render()
                )
            output_xnan = re.sub("<td.*nan.*td>", "<td></td>", default_output)
        else:
            raise ValueError("heatmap() only works with a single triangle.")
        if HTML:
            return HTML(output_xnan)
        elif HTML is None:
            raise ImportError("heatmap requires IPython.")

    @property
    def _dimensionality(self) -> str:
        """
        Determine whether the triangle is empty, single-dimensional, or multidimensional. Used for conditional
        branching in displaying the triangle.

        Returns
        -------
        str
        """
        try:
             self.values
        except AttributeError:
            return 'empty'

        if (self.values.shape[0], self.values.shape[1]) == (1, 1):
            return 'single'

        else :
            return 'multi'