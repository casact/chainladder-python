# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
import warnings
import re

try:
    from IPython.core.display import HTML
except:
    HTML = None


class TriangleDisplay:
    def __repr__(self):
        if (self.values.shape[0], self.values.shape[1]) == (1, 1):
            data = self._repr_format()
            return data.to_string()
        else:
            return self._summary_frame().__repr__()

    def _summary_frame(self):
        return pd.Series(
            [
                self.valuation_date.strftime("%Y-%m"),
                "O" + self.origin_grain + "D" + self.development_grain,
                self.shape,
                self.key_labels,
                list(self.vdims),
            ],
            index=["Valuation:", "Grain:", "Shape:", "Index:", "Columns:"],
            name="Triangle Summary",
        ).to_frame()

    def _repr_html_(self):
        """ Jupyter/Ipython HTML representation """
        if (self.values.shape[0], self.values.shape[1]) == (1, 1):
            data = self._repr_format()
            fmt_str = self._get_format_str(data)
            if len(self.ddims) > 1 and type(self.ddims[0]) is int:
                data.columns = [["Development Lag"] * len(self.ddims), self.ddims]
            default = data.to_html(
                max_rows=pd.options.display.max_rows,
                max_cols=pd.options.display.max_columns,
                float_format=fmt_str.format,
            ).replace("nan", "").replace("NaN", "")
            return default
        else:
            return self._summary_frame().to_html(
                max_rows=pd.options.display.max_rows,
                max_cols=pd.options.display.max_columns,
            )

    def _get_format_str(self, data):
        if np.all(np.isnan(data)):
            return ""
        elif np.nanmean(abs(data)) < 10:
            return "{0:,.4f}"
        elif np.nanmean(abs(data)) < 1000:
            return "{0:,.2f}"
        else:
            return "{:,.0f}"

    def _repr_format(self, origin_as_datetime=False):
        out = self.compute().set_backend("numpy").values[0, 0]
        origin = self.origin.copy()
        origin.name = None
        development = self.development.copy()
        development.name = None
        return pd.DataFrame(out, index=origin, columns=development)

    def heatmap(self, cmap="Reds", low=0, high=0, axis=0, subset=None):
        """ Color the background in a gradient according to the data in each
        column (optionally row). Requires matplotlib

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
        if (self.values.shape[0], self.values.shape[1]) == (1, 1):
            data = self._repr_format()
            fmt_str = self._get_format_str(data)
            if len(self.ddims) > 1 and type(self.ddims[0]) is int:
                data.columns = [["Development Lag"] * len(self.ddims), self.ddims]
            warnings.filterwarnings("ignore")
            axis = self._get_axis(axis)
            default = (
                data.style.format(fmt_str)
                .background_gradient(
                    cmap=cmap, low=low, high=high, axis=axis, subset=subset
                )
                .render()
            )
            default = re.sub("<td.*nan.*td>", "<td></td>", default)
            warnings.filterwarnings("default")
            return HTML(default)
        elif HTML is None:
            raise ImportError("heatmap requires IPython")
        else:
            raise ValueError("heatmap only works with single triangles")
