"""
Mirror pandas API onto the Triangle class.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import operator
import warnings

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from chainladder import (
    __dt64_dtype__,
    _deprecated_drop_argument,
    _warn_dask_parallel_deprecated,
)
from chainladder.utils.utility_functions import concat, num_to_nan

try:
    import dask.bag as db
except ImportError:
    db = None

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType
    from typing import Any, Literal

    from numpy import ndarray
    from pandas import DataFrame, Series
    from pandas._typing import IndexLabel

    from chainladder import Triangle
    from chainladder.core.typing import BackendArray, TriangleProtocol
    from chainladder.utils.sparse import COO

    _TrianglePandasBase = TriangleProtocol
else:
    _TrianglePandasBase = object


class TriangleGroupBy:
    def __init__(self, obj: Triangle, by, axis=0, **kwargs):
        self.obj = obj.copy()
        self.axis = self.obj._get_axis(axis)
        self.by = by
        if kwargs.get("groups", None):
            self.groups = kwargs["groups"]
        elif self.axis == 0:
            self.groups = obj.index.groupby(by)
        elif axis == 1:
            self.groups = pd.DataFrame(obj.columns).groupby(by)
        else:
            self.groups = pd.DataFrame(by).groupby(by)

    def __getitem__(self, key):
        return TriangleGroupBy(
            obj=self.obj.__getitem__(key),
            by=self.by,
            axis=self.axis,
            groups=self.groups,
        )

    def sum(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``sum`` functionality."""
        return _groupby_aggregate(self, "sum", *args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``mean`` functionality."""
        return _groupby_aggregate(self, "mean", *args, **kwargs)

    def median(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``median`` functionality."""
        return _groupby_aggregate(self, "median", *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``max`` functionality."""
        return _groupby_aggregate(self, "max", *args, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``min`` functionality."""
        return _groupby_aggregate(self, "min", *args, **kwargs)

    def prod(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``prod`` functionality."""
        return _groupby_aggregate(self, "prod", *args, **kwargs)

    def var(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``var`` functionality."""
        return _groupby_aggregate(self, "var", *args, **kwargs)

    def std(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``std`` functionality."""
        return _groupby_aggregate(self, "std", *args, **kwargs)

    def cumsum(self, *, auto_sparse: bool = True) -> Triangle:
        """
        Cumulate within each group, keeping every label in its original order.

        Unlike the other groupby methods, this does not reduce each group to a
        single label. Like pandas ``groupby(...).cumsum()``, the result has the
        shape and labels of the grouped Triangle.

        Parameters
        ----------
        auto_sparse : bool, default True
            Whether the result's array backend may be switched based on its size
            and density. ``False`` keeps the input's backend.

        Returns
        -------
        Triangle
            Running totals within each group, following
            :meth:`~chainladder.Triangle.cumsum`. Labels with a missing group
            key are NaN.
        """
        obj = self.obj.copy()
        xp = obj.get_array_module()
        values = []
        positions = []
        for rows in self.groups.indices.values():
            indexer = [slice(None)] * self.axis + [rows]
            group = obj.iloc[tuple(indexer)].cumsum(self.axis, auto_sparse=False)
            values.append(group.values)
            positions.extend(rows)

        # Rows excluded by groupby (e.g. missing keys) have no running total.
        missing = np.setdiff1d(np.arange(obj.shape[self.axis]), positions)
        if len(missing):
            shape = list(obj.shape)
            shape[self.axis] = len(missing)
            values.append(xp.full(tuple(shape), xp.nan))
            positions.extend(missing)
        if values:
            indexer = [slice(None)] * self.axis + [np.argsort(positions)]
            obj.values = xp.concatenate(values, axis=self.axis)[tuple(indexer)]
        if auto_sparse:
            obj = obj._auto_sparse()
        return obj

    def quantile(self, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``quantile`` functionality."""
        return _groupby_aggregate(self, "quantile", *args, **kwargs)


class TrianglePandas(_TrianglePandasBase):
    if TYPE_CHECKING:
        # Implemented by Common.set_backend; declared here so it takes precedence
        # over the unimplemented TriangleProtocol stub in the type-checking MRO.
        def set_backend(
            self,
            backend: str,
            inplace: bool = False,
            deep: bool = False,
            _warn: bool = True,
            **kwargs: Any,
        ) -> Triangle: ...

    def sum(self, axis: str | int | None = None, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``sum`` functionality."""
        return _triangle_aggregate(self, "nansum", axis, *args, **kwargs)

    def mean(
        self, axis: str | int | None = None, *args: Any, **kwargs: Any
    ) -> Triangle:
        """Refer to pandas for ``mean`` functionality."""
        return _triangle_aggregate(self, "nanmean", axis, *args, **kwargs)

    def median(
        self, axis: str | int | None = None, *args: Any, **kwargs: Any
    ) -> Triangle:
        """Refer to pandas for ``median`` functionality."""
        return _triangle_aggregate(self, "nanmedian", axis, *args, **kwargs)

    def max(self, axis: str | int | None = None, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``max`` functionality."""
        return _triangle_aggregate(self, "nanmax", axis, *args, **kwargs)

    def min(self, axis: str | int | None = None, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``min`` functionality."""
        return _triangle_aggregate(self, "nanmin", axis, *args, **kwargs)

    def prod(
        self, axis: str | int | None = None, *args: Any, **kwargs: Any
    ) -> Triangle:
        """Refer to pandas for ``prod`` functionality."""
        return _triangle_aggregate(self, "nanprod", axis, *args, **kwargs)

    def var(self, axis: str | int | None = None, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``var`` functionality."""
        return _triangle_aggregate(self, "nanvar", axis, *args, **kwargs)

    def std(self, axis: str | int | None = None, *args: Any, **kwargs: Any) -> Triangle:
        """Refer to pandas for ``std`` functionality."""
        return _triangle_aggregate(self, "nanstd", axis, *args, **kwargs)

    def cumsum(
        self,
        axis: str | int | None = None,
        *,
        auto_sparse: bool = True,
    ) -> Triangle:
        """
        Cumulate values along an axis, keeping the shape and labels.

        Triangles store zeros as NaN, so a NaN inside the triangle counts as
        zero and the running total carries through it, as in
        :meth:`~chainladder.Triangle.incr_to_cum`. Only cells outside the
        triangle, which are valued after ``valuation_date``, stay NaN.

        Only the values change. Flags such as ``is_cumulative`` are left as they
        are; use :meth:`~chainladder.Triangle.incr_to_cum` to convert an
        incremental Triangle.

        Parameters
        ----------
        axis : str | int | None
            Axis to cumulate along. ``None`` uses the first axis whose length is
            not 1, or the index if every axis has length 1.
        auto_sparse : bool, default True
            Whether the result's array backend may be switched based on its size
            and density. ``False`` keeps the input's backend.

        Returns
        -------
        Triangle
            Running totals with the input's shape and labels. Zero totals are
            stored as NaN.
        """
        obj = self.copy()
        axis = (
            next((i for i, size in enumerate(obj.shape) if size != 1), 0)
            if axis is None
            else self._get_axis(axis)
        )
        xp = obj.get_array_module()
        if obj.array_backend == "sparse":
            # sparse has no cumsum: add one slice at a time, which stays sparse.
            prefixes = []
            running = None
            for stop in range(1, obj.shape[axis] + 1):
                indexer = [slice(None)] * axis + [slice(stop - 1, stop)]
                increment = xp.nan_to_num(obj.values[tuple(indexer)])
                running = increment if running is None else running + increment
                prefixes.append(running)
            values = xp.concatenate(prefixes, axis=axis) if prefixes else obj.values
        else:
            values = xp.nancumsum(obj.values, axis)
        obj.values = cast(
            "BackendArray", num_to_nan(values * obj.nan_triangle[None, None])
        )
        if auto_sparse:
            obj = obj._auto_sparse()
        return obj

    def quantile(
        self, axis: str | int | None = None, *args: Any, **kwargs: Any
    ) -> Triangle:
        """Refer to pandas for ``quantile`` functionality."""
        return _triangle_aggregate(self, "nanquantile", axis, *args, **kwargs)

    def diff(
        self,
        axis: str | int | None = None,
        periods: int = 1,
        *,
        auto_sparse: bool = True,
    ) -> Triangle:
        """
        Difference values along an axis, keeping the shape and labels.

        Each cell has the cell ``periods`` positions away subtracted from it,
        like pandas ``diff``. Unlike pandas, ``axis`` is the first argument.

        Triangles store zeros as NaN, so a NaN inside the triangle counts as
        zero, as in :meth:`~chainladder.Triangle.cum_to_incr`. A difference is
        NaN only when either of its two cells is outside the triangle, that is,
        valued after ``valuation_date``, or when there is no cell to subtract.

        Only the values change. Flags such as ``is_cumulative`` are left as they
        are; use :meth:`~chainladder.Triangle.cum_to_incr` to convert a
        cumulative Triangle.

        Parameters
        ----------
        axis : str | int | None
            Axis to difference along. ``None`` uses the first axis whose length
            is not 1, or the index if every axis has length 1.
        periods : int, default 1
            Distance to the cell that is subtracted. Positive values subtract an
            earlier cell, negative values a later one.
        auto_sparse : bool, default True
            Whether the result's array backend may be switched based on its size
            and density. ``False`` keeps the input's backend.

        Returns
        -------
        Triangle
            Differences with the input's shape and labels. Zero differences are
            stored as NaN.
        """
        periods = operator.index(periods)
        obj = self.copy()
        axis = (
            next((i for i, size in enumerate(obj.shape) if size != 1), 0)
            if axis is None
            else self._get_axis(axis)
        )
        xp = obj.get_array_module()
        if periods == 0 or abs(periods) >= obj.shape[axis]:
            # Every difference is zero, or no cell has one to subtract.
            obj.values = xp.full(obj.shape, xp.nan)
        else:
            current = [slice(None)] * 4
            previous = [slice(None)] * 4
            current[axis] = (
                slice(periods, None) if periods > 0 else slice(None, periods)
            )
            previous[axis] = (
                slice(None, -periods) if periods > 0 else slice(-periods, None)
            )
            values = xp.where(xp.isnan(obj.values), 0, obj.values)
            values = values[tuple(current)] - values[tuple(previous)]
            mask = obj.nan_triangle[None, None]
            if axis in (2, 3):
                mask = mask[tuple(current)] * mask[tuple(previous)]
            values = values * mask
            shape = list(obj.shape)
            shape[axis] = abs(periods)
            padding = xp.full(tuple(shape), xp.nan)
            parts = [padding, values] if periods > 0 else [values, padding]
            obj.values = cast(
                "BackendArray", num_to_nan(xp.concatenate(parts, axis=axis))
            )
        if auto_sparse:
            obj = obj._auto_sparse()
        return obj

    def to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """Refer to pandas for ``to_clipboard`` functionality."""
        return self._df_passthru("to_clipboard", *args, **kwargs)

    def to_csv(self, *args: Any, **kwargs: Any) -> str | None:
        """Refer to pandas for ``to_csv`` functionality."""
        return self._df_passthru("to_csv", *args, **kwargs)

    def to_excel(self, *args: Any, **kwargs: Any) -> None:
        """Refer to pandas for ``to_excel`` functionality."""
        return self._df_passthru("to_excel", *args, **kwargs)

    def to_json(self, *args: Any, **kwargs: Any) -> str | None:
        """Refer to pandas for ``to_json`` functionality."""
        return self._df_passthru("to_json", *args, **kwargs)

    def to_html(self, *args: Any, **kwargs: Any) -> str | None:
        """Refer to pandas for ``to_html`` functionality."""
        return self._df_passthru("to_html", *args, **kwargs)

    def to_dict(self, *args: Any, **kwargs: Any) -> Mapping | list[Mapping]:
        """Refer to pandas for ``to_dict`` functionality."""
        return self._df_passthru("to_dict", *args, **kwargs)

    def unstack(self, *args: Any, **kwargs: Any) -> DataFrame | Series:
        """Refer to pandas for ``unstack`` functionality."""
        return self._df_passthru("unstack", *args, **kwargs)

    def pivot(self, *args: Any, **kwargs: Any) -> DataFrame:
        """Refer to pandas for ``pivot`` functionality."""
        return self._df_passthru("pivot", *args, **kwargs)

    def drop_duplicates(self, *args: Any, **kwargs: Any) -> DataFrame | None:
        """Refer to pandas for ``drop_duplicates`` functionality."""
        return self._df_passthru("drop_duplicates", *args, **kwargs)

    def describe(self, *args: Any, **kwargs: Any) -> DataFrame:
        """Refer to pandas for ``describe`` functionality."""
        return self._df_passthru("describe", *args, **kwargs)

    def melt(self, *args: Any, **kwargs: Any) -> DataFrame:
        """Refer to pandas for ``melt`` functionality."""
        return self._df_passthru("melt", *args, **kwargs)

    def pct_change(self, *args: Any, **kwargs: Any) -> DataFrame:
        """Refer to pandas for ``pct_change`` functionality."""
        return self._df_passthru("pct_change", *args, **kwargs)

    def _df_passthru(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return getattr(pd.DataFrame, method)(self.to_frame(), *args, **kwargs)

    def to_frame(
        self,
        origin_as_datetime: bool = True,
        keepdims: bool = False,
        implicit_axis: bool = False,
    ) -> DataFrame | Series:
        """
        Converts a triangle to a pandas.DataFrame.

        Parameters
        ----------
        origin_as_datetime : bool (default = True)
            When all dimensions are returned, whether the origin vector
            should be converted from PeriodIndex into a datetime dtype.
        keepdims : bool (default = False)
            Converted DataFrame will keep all dimensions intact and maintain a consistent
            format regardless of whether any dimensions are of length 1.

            Ignored when 3 or more dimensions (index, column, origin, and development)
            have lengths greater than 1
        implicit_axis : bool (default = False)
            When implicit_axis is True, this denotes whether to include the implicit
            valuation axis in addition to the origin and development.

        Returns
        -------
        DataFrame or Series representation of the Triangle.
        """

        # Identify the axes that increase the dimensionality of the triangle, i.e., those whose length is > 1.
        axes: list[int] = [num for num, item in enumerate(self.shape) if item > 1]

        # Long format.
        if keepdims:
            is_val_tri: bool = self.is_val_tri
            obj: Triangle = self.val_to_dev().set_backend("sparse")
            values: COO = cast("COO", obj.values)
            out: DataFrame = pd.DataFrame(obj.index.iloc[values.coords[0]])
            out["columns"] = obj.columns[values.coords[1]]
            missing_cols: list = list(set(self.columns) - set(out["columns"]))
            if origin_as_datetime:
                out["origin"] = obj.odims[values.coords[2]]
            else:
                out["origin"] = obj.origin[values.coords[2]]
            out["development"] = obj.ddims[values.coords[3]]
            out["values"] = values.data
            out: DataFrame = pd.pivot_table(
                out, index=obj.key_labels + ["origin", "development"], columns="columns"
            )
            out: DataFrame = out.reset_index().set_index(obj.key_labels)
            out.columns = ["origin", "development"] + list(
                out.columns.get_level_values(1)[2:]
            )

            valuation_series = pd.DataFrame(
                obj.valuation.values.reshape(obj.shape[-2:], order="F"),
                index=obj.odims if origin_as_datetime else obj.origin,
                columns=obj.ddims,
            ).unstack()
            valuation_series.name = "valuation"
            valuation: DataFrame = valuation_series.reset_index().rename(
                columns={"level_0": "development", "level_1": "origin"}
            )
            val_dict: dict = dict(
                zip(
                    list(zip(valuation["origin"], valuation["development"])),
                    valuation["valuation"],
                )
            )
            if len(out) > 0:
                out["valuation"] = out.apply(
                    lambda x: val_dict[(x["origin"], x["development"])], axis=1
                )
            else:
                out["valuation"] = self.valuation_date
            col_order: list = list(self.columns)
            if implicit_axis:
                col_order: list = ["origin", "development", "valuation"] + col_order
            else:
                if is_val_tri:
                    col_order: list = ["origin", "valuation"] + col_order
                else:
                    col_order: list = ["origin", "development"] + col_order
            for col in set(missing_cols) - self.virtual_columns.columns.keys():
                out[col] = np.nan
            # Create physical columns out of virtual ones.
            for col in set(missing_cols).intersection(
                self.virtual_columns.columns.keys()
            ):
                # Fill na to enable floating-point computation.
                out[col] = out.fillna(0).apply(self.virtual_columns.columns[col], 1)
                # Coerce 0 to np.nan.
                out.loc[out[col] == 0, col] = np.nan

            return out[col_order]

        # keepdims = False
        else:
            # Case when there is a single triangle, for a single segment.
            if self.shape[:2] == (1, 1):
                return self._repr_format(origin_as_datetime)
            # Case when triangle is multidimensional but is of unusual shape, such as a collection of latest diagonals.
            elif len(axes) in [1, 2]:
                tri: ndarray = np.squeeze(self.set_backend("numpy").values)
                axes_lookup: dict = {
                    0: self.index.values,
                    1: self.columns.values,
                    2: self.origin,
                    3: self.development,
                }

                # Set the index to be key dimension if the key dimension is greater than length 1.
                if axes[0] == 0:
                    idx = self.index.set_index(self.key_labels).index
                # Otherwise, find the axis that is greater than length 0 and set that to be the index.
                else:
                    idx = axes_lookup[axes[0]]

                if len(axes) == 1:
                    return pd.Series(tri, index=idx).fillna(0)
                # Case len(axes) == 2.
                else:
                    return pd.DataFrame(
                        tri, index=idx, columns=axes_lookup[axes[1]]
                    ).fillna(0)
            # Multidimensional triangles, return DataFrame in long form.
            else:
                return self.to_frame(
                    origin_as_datetime=origin_as_datetime,
                    keepdims=True,
                    implicit_axis=implicit_axis,
                )

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """
        Passthrough of pandas functionality. Calls DataFrame.plot() after the
        Triangle is transformed into a pandas DataFrame.

        Parameters
        ----------
        *args: Any
            Positional arguments passed to ``pandas.DataFrame.plot``.
        **kwargs: Any
            Keyword arguments passed to ``pandas.DataFrame.plot``, e.g.
            ``kind``, ``ax``, ``title``, ``subplots``. See the pandas
            documentation for the full list of supported parameters.

        Returns
        -------
        None
        """
        return self.to_frame(origin_as_datetime=False).plot(*args, **kwargs)

    def hvplot(self, *args: Any, **kwargs: Any) -> Any:
        """
        Passthrough of pandas functionality. Generate an interactive plot
        of a Triangle after it has been transformed into a DataFrame().

        Parameters
        ----------
        *args: Any
            Positional arguments passed to ``pandas.DataFrame.hvplot``.
        **kwargs: Any
            Keyword arguments passed to ``pandas.DataFrame.hvplot``.
        Returns
        -------
        Any
        """
        df = self.to_frame(origin_as_datetime=True)
        if isinstance(df.index, pd.PeriodIndex) and len(df.columns) > 1:
            df.index = df.index.to_timestamp(how="s")
        return df.hvplot(*args, **kwargs)

    @staticmethod
    def _get_axis(
        axis: Literal["index", "columns", "origin", "development"] | int | None,
    ) -> int:
        """
        Returns the integer representation of the requested axis.

        Parameters
        ----------
        axis: Literal['index', 'columns', 'origin', 'development'] | int | None
            String or integer representation of the requested axis. If
            supplied as a string, returns the integer representation. If
            supplied as an integer, returns the same integer.

        Returns
        -------
        int
            The integer representation of the requested axis
        """

        ax = {
            **{0: 0, 1: 1, 2: 2, 3: 3},
            **{-1: 3, -2: 2, -3: 1, -4: 0},
            **{"index": 0, "columns": 1, "origin": 2, "development": 3},
        }

        try:
            return ax[axis]
        except KeyError:
            if axis is None:
                return 0
            else:
                raise ValueError(
                    "Invalid axis specified. Please specify the correct string or "
                    "integer representation of the desired axis."
                )

    def dropna(self) -> Triangle:
        """
        Method that removes origin/development vectors from edge of a
        triangle that are all missing values. Does not work on the interior
        of a triangle, i.e., when a period has all NaNs but is not the first or last period
        of the dimension.

        This may come in handy for a
        new line of business that doesn't have origins/developments of an
        existing line in the same triangle.

        Returns
        -------
        Triangle

        Examples
        --------

        In a single-dimension case, an origin period will be dropped if it contains all NaN.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            import numpy as np
            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1986, 1986, 1987],
                    'valuation': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [np.nan, np.nan, np.nan, 500, 600, 500],
                },
                origin='origin',
                valuation='valuation',
                columns=['paid'],
                cumulative=True
            )
            print(tri)

        .. testoutput::

                     12     24  36
            1985    NaN    NaN NaN
            1986  500.0  600.0 NaN
            1987  500.0    NaN NaN

        .. testcode::

            print(tri.dropna())

        .. testoutput::

                     12     24
            1986  500.0  600.0
            1987  500.0    NaN

        If the development period has all NaNs, it will be dropped.

        .. testcode::

            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1986, 1986, 1987],
                    'valuation': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [np.nan, 500, 600, np.nan, 600, np.nan],
                },
                origin='origin',
                valuation='valuation',
                columns=['paid'],
                cumulative=True
            )
            print(tri)

        .. testoutput::

                  12     24     36
            1985 NaN  500.0  600.0
            1986 NaN  600.0    NaN
            1987 NaN    NaN    NaN

        .. testcode::

            print(tri.dropna())

        .. testoutput::

                     24     36
            1985  500.0  600.0
            1986  600.0    NaN

        If both the earliest origin and development periods are all NaN, both will be dropped.

        .. testcode::

            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1986, 1986, 1987],
                    'valuation': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [np.nan, np.nan, np.nan, np.nan, 600, np.nan],
                },
                origin='origin',
                valuation='valuation',
                columns=['paid'],
                cumulative=True
            )
            print(tri)

        .. testoutput::

                  12     24  36
            1985 NaN    NaN NaN
            1986 NaN  600.0 NaN
            1987 NaN    NaN NaN

        .. testcode::

            print(tri.dropna())

        .. testoutput::

                     24
            1986  600.0

        If a period in the middle of the Triangle is all NaN, `Triangle.dropna()` will have no effect.

        .. testcode::

            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1986, 1986, 1987],
                    'valuation': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [500, np.nan, 700, 500, np.nan, 500],
                },
                origin='origin',
                valuation='valuation',
                columns=['paid'],
                cumulative=True
            )
            print(tri)

        .. testoutput::

                     12  24     36
            1985  500.0 NaN  700.0
            1986  500.0 NaN    NaN
            1987  500.0 NaN    NaN

        .. testcode::

            print(tri.dropna())

        .. testoutput::

                     12  24     36
            1985  500.0 NaN  700.0
            1986  500.0 NaN    NaN
            1987  500.0 NaN    NaN

        If the last period has a NaN, it will be dropped.

        .. testcode::

            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1986, 1986, 1987],
                    'valuation': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [500, 600, np.nan, 500, 600, 500],
                },
                origin='origin',
                valuation='valuation',
                columns=['paid'],
                cumulative=True
            )
            print(tri)

        .. testoutput::

                     12     24  36
            1985  500.0  600.0 NaN
            1986  500.0  600.0 NaN
            1987  500.0    NaN NaN

        .. testcode::

            print(tri.dropna())

        .. testoutput::

                     12     24
            1985  500.0  600.0
            1986  500.0  600.0
            1987  500.0    NaN

        In the case of a multi-dimensional Triangle, periods will only be dropped if their aggregate sum across
        the index and columns results in all NaN for the period.

        .. testcode::

            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1986, 1986, 1987] * 2,
                    'valuation': [1985, 1986, 1987, 1986, 1987, 1987] * 2,
                    'lob': ['abc'] * 6 + ['xyz'] * 6,
                    'paid': [np.nan, np.nan, np.nan, 500, 600, 500] * 2,
                },
                origin='origin',
                valuation='valuation',
                index='lob',
                columns=['paid'],
                cumulative=True
            )
            print(tri.loc['abc'])

        .. testoutput::

                     12     24  36
            1985    NaN    NaN NaN
            1986  500.0  600.0 NaN
            1987  500.0    NaN NaN

        .. testcode::

            print(tri.dropna().sum())

        .. testoutput::

                      12      24
            1986  1000.0  1200.0
            1987  1000.0     NaN
        """

        # Aggregate the triangle across the index and columns.
        obj = self.sum(axis=0).sum(axis=1)
        xp = obj.get_array_module()
        # Check which origins have all NaNs and indicate with a boolean. 0 means that the nth origin is all NaN.
        odim = list((xp.nansum(obj.values[0, 0, :], -1) != 0).astype("int"))
        # Find the first origin period with data.
        min_odim = obj.origin[odim.index(1)]
        # Find the last origin period with data.
        max_odim = obj.origin[::-1][odim[::-1].index(1)]
        # Case when triangle has multiple development periods, e.g., not latest diagonal or ultimate.
        if obj.shape[-1] != 1:
            # Flag the development periods that have data.
            ddim = list((xp.nansum(obj.values[0, 0, :], -2) != 0).astype("int"))
            ddim = obj.development[pd.Series(ddim).astype(bool)]
            # Slice the Triangle by the development periods that have data.
            obj = self[
                (self.development >= ddim.min()) & (self.development <= ddim.max())
            ]
            obj = cast("TriangleProtocol", cast(object, obj))
            # Slice the triangle by the origin periods that have data.
            return obj[(self.origin >= min_odim) & (self.origin <= max_odim)]
        # Case when Triangle has a single development period, e.g., latest diagonal or ultimate.
        obj = self[(self.origin >= min_odim) & (self.origin <= max_odim)]
        return obj

    def fillna(self, value: int | float | ndarray, inplace: bool = False) -> Triangle:
        """
        Fill nan with 'value' by axis.

        Parameters
        ----------
        value: single value or array-like values
            Value(s) to fill across the axis.

        inplace: boolean, default = False
            Whether to modify the triangle object directly (True), or
            return a new modified triangle (False).

        Returns
        -------
        Triangle

        .. deprecated:: 0.11

           The return value of using ``inplace=True`` is deprecated.
           In version 2.0, ``None`` will be returned instead of ``self``.
        """
        if value is None:
            raise TypeError("Must specify a fill value.")
        if inplace:
            xp = self.get_array_module()
            # Create a triangle will the fill value in the original Triangle's NaN positions.
            # Positions corresponding to populated positions in teh original Triangle are set to NaN.
            fill = (xp.nan_to_num(self.values) == 0) * (self * 0 + value)
            self.values = (self + fill).values
            return cast("Triangle", cast(object, self))
        else:
            new_obj = self.copy()
            cast("TriangleProtocol", cast(object, new_obj)).fillna(
                value=value, inplace=True
            )
            return new_obj

    def fillzero(self, inplace: bool = False) -> Triangle:
        """
        Fill nan with 0 by axis. separate function from fillna() because fillna(0) isn't working.

        Parameters
        ----------
        inplace: bool, default = False
            Whether to modify the triangle object directly (True), or
            return a new modified triangle (False).

        Returns
        -------
        Triangle

        .. deprecated:: 0.11

           The return value of using ``inplace=True`` is deprecated.
           In version 2.0, ``None`` will be returned instead of ``self``.
        """
        if inplace:
            xp = self.get_array_module()
            # Fill the NaNs by locating their positions within the triangle.
            self.values = np.where(
                (xp.nan_to_num(self.values) == 0) * (self.nan_triangle == 1),
                self.nan_triangle * 0,
                self.values,
            )
            return cast("Triangle", cast(object, self))
        else:
            new_obj = self.copy()
            cast("TriangleProtocol", cast(object, new_obj)).fillzero(inplace=True)
            return new_obj

    def ffill(self, axis: int | str = 3) -> Triangle:
        """
        Forward-fill missing values along an axis.

        Only cells within the observed triangle (see ``nan_triangle``) are
        filled; a cell that has not yet been valued is left as-is regardless
        of what precedes it along the axis.

        Parameters
        ----------
        axis : {2 or 'origin', 3 or 'development'}, default 3
            Fill direction.

        Returns
        -------
        Triangle

        Examples
        --------

        .. testsetup::

            import numpy as np

        .. testcode::

            import chainladder as cl
            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1985, 1986, 1986, 1986, 1987, 1987, 1988],
                    'valuation': [1985, 1986, 1987, 1988, 1986, 1987, 1988, 1987, 1988, 1988],
                    'paid': [500, np.nan, 700, np.nan, np.nan, 1000, 1100, 1200, 1300, np.nan],
                },
                origin='origin',
                valuation='valuation',
                columns=['paid'],
                cumulative=True,
            )
            print(tri)

        .. testoutput::
            :options: +NORMALIZE_WHITESPACE

                     12      24      36  48
            1985  500.0     NaN   700.0 NaN
            1986     NaN  1000.0  1100.0 NaN
            1987  1200.0  1300.0     NaN NaN
            1988     NaN     NaN     NaN NaN

        Fill along the development axis (the default). ``1986`` at age 12
        stays missing because nothing precedes it, and ``1986`` at age 48 /
        ``1987`` at ages 36-48 stay missing because they have not yet been
        valued.

        .. testcode::

            print(tri.ffill())

        .. testoutput::
            :options: +NORMALIZE_WHITESPACE

                     12      24      36     48
            1985  500.0   500.0   700.0  700.0
            1986     NaN  1000.0  1100.0    NaN
            1987  1200.0  1300.0     NaN    NaN
            1988     NaN     NaN     NaN    NaN

        Fill along the origin axis. ``1985`` at age 24 stays missing because
        nothing precedes it there, and ``1988`` at ages 24-48 stay missing
        because they have not yet been valued.

        .. testcode::

            print(tri.ffill(axis='origin'))

        .. testoutput::
            :options: +NORMALIZE_WHITESPACE

                     12      24      36  48
            1985  500.0     NaN   700.0 NaN
            1986  500.0  1000.0  1100.0 NaN
            1987  1200.0  1300.0     NaN NaN
            1988  1200.0     NaN     NaN NaN
        """
        axis = self._get_axis(axis)
        if axis < 2:
            raise AttributeError(
                "ffill is only supported for the origin and development axes"
            )
        xp = self.get_array_module()
        n = self.shape[axis]
        columns = (
            [self.iloc[..., i : i + 1] for i in range(n)]
            if axis == 3
            else [self.iloc[..., i : i + 1, :] for i in range(n)]
        )
        filled = [columns[0]]
        for current in columns[1:]:
            previous = filled[-1]
            is_missing = xp.isnan(current.values)
            current = current.copy()
            current.values = xp.where(is_missing, previous.values, current.values)
            filled.append(current)
        out = concat(filled, axis=axis)
        # a value can never be carried into a cell that hasn't been valued yet
        out.values = xp.where(xp.isnan(out.nan_triangle), out.nan_triangle, out.values)
        return cast("Triangle", cast(object, out))

    @staticmethod
    def _validate_contiguous_drop(
        axis_series: pd.Series,
        drop_labels: list[Any],
        axis_name: str,
        errors: str,
    ) -> np.ndarray:
        """
        Validate and return boolean keep mask for dropping contiguous edge periods.

        Parameters
        ----------
        axis_series : pd.Series
            The existing labels along the axis.
        drop_labels : list
            Labels requested to be dropped.
        axis_name : str
            Name of the axis ('origin' or 'development') for error messages.
        errors : {'raise', 'ignore'}
            Whether to raise or ignore missing labels.

        Returns
        -------
        np.ndarray
            Boolean mask of labels to keep.
        """
        axis_labels = np.array(axis_series.astype(str))
        str_drop_labels = [str(label) for label in drop_labels]
        missing = [label for label in str_drop_labels if label not in axis_labels]
        if missing and errors == "raise":
            raise KeyError(f"{missing} not found in the {axis_name} axis.")
        keep = ~np.isin(axis_labels, str_drop_labels)
        kept_positions = np.flatnonzero(keep)
        if len(kept_positions) and not np.array_equal(
            kept_positions,
            np.arange(kept_positions[0], kept_positions[-1] + 1),
        ):
            raise ValueError(
                f"Only the first or last {axis_name} periods may be dropped; "
                f"dropping an interior {axis_name} period would leave a gap."
            )
        return keep

    def drop(
        self,
        labels: str | int | list | None = None,
        axis: Literal["index", "columns", "origin", "development"] | int = 1,
        index: str | int | list | None = None,
        columns: str | int | list | None = None,
        origin: str | int | list | None = None,
        development: str | int | list | None = None,
        level: int | str | None = None,
        errors: str = "raise",
    ) -> Triangle:
        """
        Drop specified labels from rows or columns.

        Remove labels by specifying label names and corresponding axis, or by
        specifying directly ``index``, ``columns``, ``origin``, or
        ``development`` names.

        Parameters
        ----------

        labels:  str | int | list | None
            Index or column labels to drop. A single label or list-like.

        axis: {0 or ‘index’, 1 or ‘columns’, 2 or 'origin', 3 or 'development'}, default 1
            The axis to drop ``labels`` from.

        index: str | int | list | None
            Alternative to ``axis=0``. Equivalent to ``labels, axis=0``.

        columns: str | int | list | None
            Alternative to ``axis=1``. Equivalent to ``labels, axis=1``.

        origin: str | int | list | None
            Alternative to ``axis=2``. Equivalent to ``labels, axis=2``.

        development: str | int | list | None
            Alternative to ``axis=3``. Equivalent to ``labels, axis=3``.

        level: int | str | None, default None
            For MultiIndex, level from which the labels will be removed.

        errors: {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are dropped.

        Returns
        -------
        Triangle

        Examples
        --------

        Drop a single column with the ``labels``/``axis`` form or the
        ``columns`` alternative; the two are equivalent.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tri = cl.load_sample('clrd')
            print(tri.columns.tolist())
            print(tri.drop(columns='CumPaidLoss').columns.tolist())

        .. testoutput::

            ['IncurLoss', 'CumPaidLoss', 'BulkLoss', 'EarnedPremDIR', 'EarnedPremCeded', 'EarnedPremNet']
            ['IncurLoss', 'BulkLoss', 'EarnedPremDIR', 'EarnedPremCeded', 'EarnedPremNet']

        A list of labels can be dropped from an axis as well.

        .. testcode::

            print(tri.drop(columns=['CumPaidLoss', 'IncurLoss']).columns.tolist())

        .. testoutput::

            ['BulkLoss', 'EarnedPremDIR', 'EarnedPremCeded', 'EarnedPremNet']

        Origin periods can be dropped with ``origin`` (or ``axis=2``), but only
        the first or last period may be removed so the triangle stays contiguous.

        .. testcode::

            raa = cl.load_sample('raa')
            print(raa.origin.astype(str).tolist()[-3:])
            print(raa.drop(origin='1990').origin.astype(str).tolist()[-3:])

        .. testoutput::

            ['1988', '1989', '1990']
            ['1987', '1988', '1989']

        Dropping an origin period also trims any development periods that are
        left entirely NaN, so the triangle is trimmed rather than left with
        empty development columns.

        .. testcode::

            tri = cl.Triangle(
                data={
                    'origin': [1985, 1985, 1985, 1986, 1986, 1987],
                    'valuation': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [300, 400, 500, 500, 600, 500],
                },
                origin='origin',
                valuation='valuation',
                columns=['paid'],
                cumulative=True
            )
            print(tri)

        .. testoutput::

                     12     24     36
            1985  300.0  400.0  500.0
            1986  500.0  600.0    NaN
            1987  500.0    NaN    NaN

        .. testcode::

            print(tri.drop(origin='1985'))

        .. testoutput::

                     12     24
            1986  500.0  600.0
            1987  500.0    NaN

        Development periods can also be dropped with ``development`` (or ``axis=3``).

        .. testcode::

            raa = cl.load_sample('raa')
            print(raa.development.tolist())
            print(raa.drop(development=120).development.tolist())

        .. testoutput::

            [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
            [12, 24, 36, 48, 60, 72, 84, 96, 108]

        """
        if errors not in ("raise", "ignore"):
            raise ValueError(f"errors must be 'raise' or 'ignore', got '{errors}'")
        alternatives = {0: index, 1: columns, 2: origin, 3: development}
        if any(value is not None for value in alternatives.values()):
            if labels is not None:
                raise ValueError(
                    "Cannot specify both 'labels' and any of 'index', "
                    "'columns', 'origin', or 'development'."
                )
            to_drop = {
                ax: value for ax, value in alternatives.items() if value is not None
            }
        else:
            if labels is None:
                raise ValueError(
                    "Need to specify at least one of 'labels', 'index', "
                    "'columns', 'origin', or 'development'."
                )
            to_drop = {self._get_axis(axis): labels}
        if level is not None and any(ax != 0 for ax in to_drop):
            raise ValueError("level is only supported for axis=0 ('index').")
        result = self
        for ax, ax_labels in to_drop.items():
            ax_labels = (
                list(ax_labels)
                if ax_labels is None or pd.api.types.is_list_like(ax_labels)
                else [ax_labels]
            )
            if ax == 1:
                missing = [item for item in ax_labels if item not in result.columns]
                if missing and errors == "raise":
                    raise KeyError(f"{missing} not found in the columns axis.")
                result = result[
                    [item for item in result.columns if item not in ax_labels]
                ]
            elif ax == 2:
                keep = self._validate_contiguous_drop(
                    result.origin, ax_labels, "origin", errors
                )
                result = result[keep]
                # Trim any development periods that were left entirely NaN by
                # the origin drop, so dropping origins trims the triangle
                # rather than leaving empty development columns behind. This is
                # what makes origin dropping more useful than plain 4D slicing
                # (gh-1055).
                if result.shape[-1] > 1:
                    agg = result.sum(axis=0).sum(axis=1)
                    vals = agg.values[0, 0, :]
                    vals_np = (
                        vals.todense() if hasattr(vals, "todense") else np.asarray(vals)
                    )
                    arr = np.nan_to_num(vals_np)
                    dev_has_data = list((arr.sum(axis=-2) != 0).astype(int))
                    dev_labels = agg.development[pd.Series(dev_has_data).astype(bool)]
                    result = result[
                        (result.development >= dev_labels.min())
                        & (result.development <= dev_labels.max())
                    ]
            elif ax == 3:
                keep = self._validate_contiguous_drop(
                    result.development, ax_labels, "development", errors
                )
                result = result._slice(keep, "ddims")
                if result.is_val_tri:
                    result.valuation_date = min(
                        result.valuation.max(), result.valuation_date
                    )
            else:
                raise NotImplementedError(
                    "Triangle.drop() only implemented for the column, "
                    "origin, and development axes."
                )
        return result

    @property
    def T(self) -> DataFrame:  # noqa: N802
        """
        Converts the Triangle to a Pandas DataFrame and then transposes it.

        Returns
        -------
        DataFrame

        """
        return self.to_frame(origin_as_datetime=False).T

    def groupby(self, by: str | list, axis: Literal[0, 1, 2, 3] = 0) -> TriangleGroupBy:
        """
        Group Triangle by index values.  If the triangle is convertable to a
        DataFrame, then it defaults to pandas groupby functionality.

        Parameters
        ----------
        by: str | list
            The index to group by

        axis: int
            The axis the groupby applies to.

        Returns
        -------
        TriangleGroupBy
            GroupBy object (pandas or Triangle)
        """
        return TriangleGroupBy(cast("Triangle", cast(object, self)), by, axis)

    def append(self, other: Triangle) -> Triangle:
        """
        Append rows of another Triangle to self, returning an updated Triangle.

        Parameters
        ----------
        other : Triangle
            The data to append.

        Returns
        -------
            New Triangle with appended data.
        """

        return concat((self, other), 0)

    def rename(
        self,
        axis: Literal["index", "columns", "origin", "development"] | int,
        value: str | list | dict,
    ) -> Triangle:
        """
        Alter Triangle axes labels.

        Parameters
        ----------
        axis: Literal['index', 'columns', 'origin', 'development'] | int
            Target axis to be relabeled.
        value: str | list | dict
            New label names for the target axis. String and list inputs replace
            all labels on the axis. Dictionary inputs map existing labels to new names.

        Returns
        -------
        Triangle
            Triangle with relabeled axis.
        """

        if isinstance(value, dict):
            if axis == "columns" or axis == 1:
                full_dict = dict(zip(self.columns.values, self.columns.values))
                full_dict.update(value)
                self.columns = self.columns.map(full_dict)
            else:
                raise ValueError(
                    "Invalid value provided to the 'value' parameter. Accepted values for index, origin, and development axes are a str or a list"
                )
        else:
            value = [value] if type(value) is str else value
            if axis == "index" or axis == 0:
                self.index = pd.DataFrame(value, columns=self.index.columns)
            elif axis == "columns" or axis == 1:
                self.columns = value
            elif axis == "origin" or axis == 2:
                self.origin = value
            elif axis == "development" or axis == 3:
                self.development = value
            else:
                raise ValueError(
                    "Invalid value provided to the 'axis' parameter. Accepted values are a string of 'index', "
                    "'columns', 'origin', or 'development', or an integer in the interval [0, 4] specifying the"
                    " axis to be modified."
                )
        return cast("Triangle", cast(object, self))

    @_deprecated_drop_argument(
        "inplace",
        "Copy the Triangle first and use the return value: "
        "`tri = tri.copy()` then `tri = tri.astype(dtype)`.",
    )
    def astype(self, dtype, inplace=True) -> Triangle:
        """
        Copy of the array, cast to a specified type.

        .. warning::
            The ``inplace`` parameter is deprecated and will be removed.
            Copy first and use the return value:

            .. code-block:: python

                tri = tri.copy()
                tri = tri.astype(new_dtype)

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        inplace : bool, default True
            Deprecated and scheduled for removal. When False, the cast is
            applied to a copy and the original Triangle is left untouched.

        Returns
        -------
        Triangle as new datatype
        """
        obj = self.copy() if inplace is False else self
        obj.values = obj.values.astype(dtype)
        return cast("Triangle", obj)

    def head(self, n: int = 5) -> Triangle:
        """
        Return the first ``n`` triangles along the index axis.

        Parameters
        ----------
        n : int, default 5
            Number of triangles to select.

        Returns
        -------
        Triangle
        """
        return self.iloc[:n]

    def tail(self, n: int = 5) -> Triangle:
        """
        Return the last ``n`` triangles along the index axis.

        Parameters
        ----------
        n : int, default 5
            Number of triangles to select.

        Returns
        -------
        Triangle
        """
        return self.iloc[-n:]

    def sort_index(self, *args, **kwargs) -> Triangle:
        """
        Sort Triangle rows by index labels.

        Returns
        -------
        Triangle
        """
        inplace = kwargs.pop("inplace", False)
        sorted_index: DataFrame = cast(
            "DataFrame",
            self.index.copy().sort_values(self.key_labels, *args, **kwargs),
        )
        result = self.iloc[sorted_index.index]
        if inplace:
            self.values = result.values
            self.index = result.index
            return self
        return result

    def exp(self) -> Triangle:
        """
        Return the exponential of each element.

        Returns
        -------
        Triangle
        """
        return self.get_array_module().exp(self)

    def log(self) -> Triangle:
        """
        Return the natural logarithm of each element.

        Returns
        -------
        Triangle
        """
        return self.get_array_module().log(self)

    def minimum(self, other: Triangle | int | float) -> Triangle:
        """
        Element-wise minimum of this Triangle and another operand.


        See :func:`chainladder.minimum` for parameters, usage, and examples.
        """
        return self.get_array_module().minimum(self, other)

    def maximum(self, other: Triangle | int | float) -> Triangle:
        """
        Element-wise maximum of this Triangle and another operand.

        See :func:`chainladder.maximum` for parameters, usage, and examples.
        """
        return self.get_array_module().maximum(self, other)

    def sqrt(self) -> Triangle:
        """
        Return the non-negative square root of each element.

        Returns
        -------
        Triangle
        """
        return self.get_array_module().sqrt(self)

    def round(self, decimals: int = 0) -> Triangle:
        """
        Round each element to the given number of decimal places.

        Uses banker's rounding (round half to even). For example,
        ``(8.5).round(0)`` returns 8, not 9. For conventional rounding,
        add a small epsilon before rounding, e.g. ``(tri + 1e-9).round(0)``.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to.

        Returns
        -------
        Triangle
        """
        return cast("Triangle", cast(object, self.__round__(decimals)))

    def xs(
        self,
        index_key: IndexLabel,
        level: IndexLabel | None = None,
        drop_level: bool = True,
    ) -> Triangle:
        """
        Mimics xs from pandas. key difference is that  this function only slices
        the index, therefore axis is always 0 and not an argument in the function

        Main use case for this function is when slicing beyond the first field in
        the index (such as LOB in the clrd dataset)

        Parameters
        ----------
        index_key: IndexLabel
            Label contained in the index.
        level: IndexLabel | None = None
            Level to take the cross-section on.
        drop_level: bool = True
            If False, returns object with same levels as self.

        Returns
        -------
        Triangle
            Cross-section from the original Triangle corresponding to the selected index levels.
        """
        mi = pd.MultiIndex.from_frame(self.index)

        lvl = 0 if level is None else level
        loc, new_ax = mi.get_loc_level(index_key, level=lvl, drop_level=drop_level)

        # create the tuple of the indexer
        _indexer = [slice(None)] * 2
        _indexer[0] = loc
        indexer = tuple(_indexer)
        result = self.iloc[indexer]
        if new_ax is not None:
            new_ax_df = new_ax.to_frame(index=None)[new_ax.names]
            result.index = new_ax_df
        else:
            result.index = pd.DataFrame(data=["Total"], columns=pd.Index(["Total"]))
        return result


def _triangle_aggregate(
    self: TrianglePandas,
    method: str,
    axis: str | int | None = None,
    *args: Any,
    **kwargs: Any,
) -> Triangle:
    """
    Apply an array aggregation, preserving Triangle dimensions and metadata.

    A singleton result is returned as a scalar unless ``keepdims=True``.
    """
    keepdims = kwargs.get("keepdims", None)
    obj = self.copy()
    auto_sparse: bool = kwargs.pop("auto_sparse", True)
    if axis is None:
        if max(obj.shape) == 1:
            axis = 0
        else:
            axis = min([num for num, _ in enumerate(obj.shape) if _ != 1])
    else:
        axis = self._get_axis(axis)
    xp: ModuleType = obj.get_array_module()
    func: Callable = getattr(xp, method)
    kwargs.update({"keepdims": True})
    # Apply the function on the requested axis.
    if axis in [0, 1] and not obj.is_full:
        with warnings.catch_warnings():
            # the lower right half of an undeveloped triangle will always emit an
            # empty or All-NaN slice warning when aggregating by index or column
            # (depending on `func`; e.g. nanmean warns "empty slice",
            # nanmax warns All-NaN slice
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*empty slice.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*All-NaN slice.*",
            )
            obj.values = func(obj.values, axis=axis, *args, **kwargs)
    else:
        obj.values = func(obj.values, axis=axis, *args, **kwargs)

    # Aggregation function will collapse a dimension, so
    # adjust the dimensions of the original object to match that of the aggregation.
    if axis == 0 and obj.values.shape[axis] == 1 and len(obj.index) > 1:
        obj.index = pd.DataFrame(
            [["(All)"] * len(obj.key_labels)], columns=obj.key_labels
        )
    if axis == 1 and obj.values.shape[axis] == 1 and len(obj.columns) > 1:
        obj.columns = pd.Index([0], name="columns")
    if axis == 2 and obj.values.shape[axis] == 1 and len(obj.odims) > 1:
        obj.odims = obj.odims[0:1]
    # If axis is development, set the ddims to be the valuation date.
    if axis == 3 and obj.values.shape[axis] == 1 and len(obj.ddims) > 1:
        obj.ddims = pd.DatetimeIndex(
            [self.valuation_date], dtype=__dt64_dtype__, freq=None
        )
    obj._set_slicers()
    if auto_sparse:
        obj = obj._auto_sparse()
    obj.values = cast("BackendArray", num_to_nan(obj.values))
    if not keepdims and obj.shape == (1, 1, 1, 1):
        return obj.values[0, 0, 0, 0]
    else:
        return obj


def _groupby_aggregate(
    self: TriangleGroupBy, method: str, *args: Any, **kwargs: Any
) -> Triangle:
    """Aggregate each group and restore the grouped axis labels."""
    from chainladder.utils import concat

    obj = self.obj.copy()
    auto_sparse = kwargs.pop("auto_sparse", True)
    if db and obj.array_backend == "sparse":
        _warn_dask_parallel_deprecated()

        def aggregate(i, obj, axis, v):
            return getattr(obj.iloc.__getitem__(tuple([slice(None)] * axis + [i])), v)(
                axis, auto_sparse=False, keepdims=True
            )

        bag = db.from_sequence(self.groups.indices.values())
        bag = bag.map(aggregate, obj, self.axis, method)
        values = bag.compute(scheduler="threads")
    else:
        values = [
            getattr(
                obj.iloc.__getitem__(tuple([slice(None)] * self.axis + [i])), method
            )(self.axis, auto_sparse=False, keepdims=True)
            for i in self.groups.indices.values()
        ]
    obj = concat(values, axis=self.axis, ignore_index=True)
    group_index = self.groups.first().index
    if self.axis == 0:
        if isinstance(group_index, pd.MultiIndex):
            index = (
                pd
                .DataFrame(
                    np.zeros(len(group_index)),
                    index=group_index,
                    columns=["_"],
                )
                .reset_index()
                .iloc[:, :-1]
            )
            obj.index = index
        else:
            index = pd.DataFrame(group_index)
            obj.index = index
    if self.axis == 1:
        obj.columns = group_index
    if self.axis == 2:
        odims = self.obj._to_datetime(
            pd.Series(self.groups.indices.keys()).to_frame(), [0]
        )
        if odims is None:
            raise ValueError("Group labels could not be converted to origin dates.")
        obj.origin_grain = self.obj._get_grain(odims)
        split = obj.origin_grain.split("-")
        obj.origin_grain = {"2Q": "S"}.get(split[0], split[0])
        obj.odims = odims.values
    obj._set_slicers()
    if auto_sparse:
        obj = obj._auto_sparse()
    return obj
