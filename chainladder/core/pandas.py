"""
Mirror pandas API onto the Triangle class.
"""
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd

from chainladder import (
    __dt64_dtype__,
    _warn_dask_parallel_deprecated,
)
from chainladder.core.typing import TriangleProtocol
from chainladder.utils.utility_functions import num_to_nan
from typing import (
    cast,
    TYPE_CHECKING
)


try:
    import dask.bag as db
except ImportError:
    db = None

if TYPE_CHECKING:
    from chainladder import Triangle
    from chainladder.utils.sparse import COO
    from chainladder.core.typing import BackendArray
    from collections.abc import Callable
    from numpy import ndarray
    from pandas import (
        DataFrame,
        Series
    )
    from types import ModuleType
    from pandas._typing import(
        IndexLabel
    )
    from typing import (
        Any,
        Literal,
        Type
    )



class TriangleGroupBy:
    def __init__(self, obj: Triangle, by, axis=0, **kwargs):
        self.obj = obj.copy()
        self.axis = self.obj._get_axis(axis)
        self.by = by
        if kwargs.get("groups", None):
            self.groups = kwargs.get("groups", None)
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


class TrianglePandas:
    # Stubs to supress type checker warnings. Refer to typing.TriangleProtocol for actual
    # typing. Remove once linters improve.
    if TYPE_CHECKING:
        values: np.ndarray

    def to_frame(
            self: TriangleProtocol,
            origin_as_datetime: bool = True,
            keepdims: bool = False,
            implicit_axis: bool = False,
    ) -> DataFrame | Series:
        """ Converts a triangle to a pandas.DataFrame.

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
            obj.values = cast("COO", obj.values)
            out: DataFrame = pd.DataFrame(obj.index.iloc[obj.values.coords[0]])
            out["columns"] = obj.columns[obj.values.coords[1]]
            missing_cols: list = list(set(self.columns) - set(out['columns']))
            if origin_as_datetime:
                out["origin"] = obj.odims[obj.values.coords[2]]
            else:
                out["origin"] = obj.origin[obj.values.coords[2]]
            out["development"] = obj.ddims[obj.values.coords[3]]
            out["values"] = obj.values.data
            out: DataFrame = pd.pivot_table(
                out, index=obj.key_labels + ["origin", "development"], columns="columns"
            )
            out: DataFrame = out.reset_index().set_index(obj.key_labels)
            out.columns = ["origin", "development"] + list(
                out.columns.get_level_values(1)[2:]
            )

            valuation_series = pd.DataFrame(
                obj.valuation.values.reshape(obj.shape[-2:], order='F'),
                index=obj.odims if origin_as_datetime else obj.origin, 
                columns=obj.ddims
            ).unstack()
            valuation_series.name = 'valuation'
            valuation: DataFrame = valuation_series.reset_index().rename(
                columns={
                    'level_0': 'development',
                    'level_1': 'origin'}
            )
            val_dict: dict = dict(zip(list(zip(
                valuation['origin'], valuation['development'])),
                valuation['valuation']))
            if len(out) > 0:
                out['valuation'] = out.apply(
                    lambda x: val_dict[(x['origin'], x['development'])], axis=1)
            else:
                out['valuation'] = self.valuation_date
            col_order: list = list(self.columns)
            if implicit_axis:
                col_order: list = ['origin', 'development', 'valuation'] + col_order
            else:
                if is_val_tri:
                    col_order: list = ['origin', 'valuation'] + col_order
                else:
                    col_order: list = ['origin', 'development'] + col_order
            for col in set(missing_cols) - self.virtual_columns.columns.keys():
                out[col] = np.nan
            # Create physical columns out of virtual ones.
            for col in set(missing_cols).intersection(self.virtual_columns.columns.keys()):
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
                    0: self.kdims,
                    1: self.vdims,
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
                    implicit_axis=implicit_axis
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
        if type(df.index) == pd.PeriodIndex and len(df.columns) > 1:
            df.index = df.index.to_timestamp(how="s")
        return df.hvplot(*args, **kwargs)

    @staticmethod
    def _get_axis(axis: Literal['index', 'columns', 'origin', 'development'] | int | None) -> int:
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

    def dropna(self: TriangleProtocol) -> Triangle:
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
                    'development': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [np.nan, np.nan, np.nan, 500, 600, 500],
                },
                origin='origin',
                development='development',
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
                    'development': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [np.nan, 500, 600, np.nan, 600, np.nan],
                },
                origin='origin',
                development='development',
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
                    'development': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [np.nan, np.nan, np.nan, np.nan, 600, np.nan],
                },
                origin='origin',
                development='development',
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
                    'development': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [500, np.nan, 700, 500, np.nan, 500],
                },
                origin='origin',
                development='development',
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
                    'development': [1985, 1986, 1987, 1986, 1987, 1987],
                    'paid': [500, 600, np.nan, 500, 600, 500],
                },
                origin='origin',
                development='development',
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
                    'development': [1985, 1986, 1987, 1986, 1987, 1987] * 2,
                    'lob': ['abc'] * 6 + ['xyz'] * 6,
                    'paid': [np.nan, np.nan, np.nan, 500, 600, 500] * 2,
                },
                origin='origin',
                development='development',
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
            ddim = list(
                (xp.nansum(obj.values[0, 0, :], -2) != 0).astype("int"))
            ddim = obj.development[pd.Series(ddim).astype(bool)]
            # Slice the Triangle by the development periods that have data.
            obj = self[
                (self.development >= ddim.min()) & (
                    self.development <= ddim.max())
            ]
            obj = cast("TriangleProtocol", cast(object, obj))
            # Slice the triangle by the origin periods that have data.
            return obj[(self.origin >= min_odim) & (self.origin <= max_odim)]
        # Case when Triangle has a single development period, e.g., latest diagonal or ultimate.
        obj = self[(self.origin >= min_odim) & (self.origin <= max_odim)]
        return obj

    def fillna(self: TriangleProtocol, value: int | float | ndarray, inplace: bool = False) -> Triangle:
        """Fill nan with 'value' by axis.

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
            cast("TriangleProtocol", cast(object, new_obj)).fillna(value=value, inplace=True)
            return new_obj

    def fillzero(self: TriangleProtocol, inplace: bool = False) -> Triangle:
        """Fill nan with 0 by axis. separate function from fillna() because fillna(0) isn't working.

        Parameters
        ----------
        inplace: bool, default = False
            Whether to modify the triangle object directly (True), or
            return a new modified triangle (False).

        Returns
        -------
        Triangle
        """
        if inplace:
            xp = self.get_array_module()
            # Fill the NaNs by locating their positions within the triangle.
            self.values = np.where(
                (xp.nan_to_num(self.values) == 0) * (self.nan_triangle == 1),
                self.nan_triangle * 0, self.values
            )
            return cast("Triangle", cast(object, self))
        else:
            new_obj = self.copy()
            cast("TriangleProtocol", cast(object, new_obj)).fillzero(inplace=True)
            return new_obj

    def drop(self, labels=None, axis=1):
        """Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding axis,
        or by specifying directly index or column names.

        Parameters
        -----------

        label: single label or list-like
            Index or column labels to drop.

        axis: {0 or ‘index’, 1 or ‘columns’}, default 1
            Whether to drop labels from the index (0 or ‘index’)
            or columns (1 or ‘columns’).

        Returns
        -------
        Triangle

        """
        labels = [labels] if type(labels) is str else list(labels)
        if axis == 1:
            return self[[item for item in self.columns if item not in labels]]
        else:
            raise NotImplementedError("drop only inpemented for column axis")

    @property
    def T(self):
        return self.to_frame(origin_as_datetime=False).T

    def groupby(self, by, axis=0, *args, **kwargs):
        """Group Triangle by index values.  If the triangle is convertable to a
        DataFrame, then it defaults to pandas groupby functionality.

        Parameters
        ----------
        by: str or list
            The index to group by

        Returns
        -------
            GroupBy object (pandas or Triangle)
        """
        return TriangleGroupBy(self, by, axis)

    def append(self, other):
        """Append rows of other to the end of caller, returning a new object.

        Parameters
        ----------
        other : Triangle
            The data to append.

        Returns
        -------
            New Triangle with appended data.
        """
        from chainladder.utils.utility_functions import concat

        return concat((self, other), 0)

    def rename(
            self: TriangleProtocol,
            axis: Literal['index', 'columns', 'origin', 'development'] | int,
            value: list | str | dict
    ):
        """Alter axes labels.

        Parameters
        ----------
        axis: str or int
            A value of 0 <= axis <= 4 corresponding to axes 'index',
            'columns', 'origin', 'development' respectively.  Both the
            int and str representation can be used.
        value: list or str or dict
            List of new labels to be assigned to the axis. List must be of
            same length of the specified axis. Can also be a dictionary for renaming columns

        Returns
        -------
            Triangle with relabeled axis.
        """
        
        if type(value) is dict:
            if axis == "columns" or axis == 1:
                full_dict = dict(zip(self.columns.values,self.columns.values))
                full_dict.update(value)
                self.columns = self.columns.map(full_dict)
            else:
                raise ValueError(
                    "Invalid value provided to the 'value' parameter. Accepted values for index, origin, and development axes are a str or a list"    
                )
        else:
            value = [value] if type(value) is str else value
            if axis == "index" or axis == 0:
                self.index = pd.DataFrame(value,columns = self.index.columns)
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
        return self

    def astype(self: TriangleProtocol, dtype, inplace=True):
        """Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, optional
            By default, astype always returns a newly allocated array.

        Returns
        -------
        Triangle as new datatype
        """
        obj = self.copy() if inplace is False else self
        obj.values = obj.values.astype(dtype)
        return obj

    def head(self: TriangleProtocol, n: int=5):
        """Return the first ``n`` triangles along the index axis.

        Parameters
        ----------
        n : int, default 5
            Number of triangles to select.

        Returns
        -------
        Triangle
        """
        return self.iloc[:n]

    def tail(self: TriangleProtocol, n: int=5):
        """Return the last ``n`` triangles along the index axis.

        Parameters
        ----------
        n : int, default 5
            Number of triangles to select.

        Returns
        -------
        Triangle
        """
        return self.iloc[-n:]

    def sort_index(self: TriangleProtocol, *args, **kwargs):
        """Sort Triangle rows by index labels.

        Returns
        -------
        Triangle
        """
        return self.iloc[self.index.sort_values(self.key_labels, *args, **kwargs).index]

    def exp(self: TriangleProtocol):
        """Return the exponential of each element.

        Returns
        -------
        Triangle
        """
        return self.get_array_module().exp(self)

    def log(self: TriangleProtocol):
        """Return the natural logarithm of each element.

        Returns
        -------
        Triangle
        """
        return self.get_array_module().log(self)

    def minimum(self: TriangleProtocol, other):
        """Element-wise minimum of this Triangle and another operand.

        See :func:`chainladder.minimum` for parameters, usage, and examples.
        """
        return self.get_array_module().minimum(self, other)

    def maximum(self: TriangleProtocol, other):
        """Element-wise maximum of this Triangle and another operand.

        See :func:`chainladder.maximum` for parameters, usage, and examples.
        """
        return self.get_array_module().maximum(self, other)

    def sqrt(self: TriangleProtocol):
        """Return the non-negative square root of each element.

        Returns
        -------
        Triangle
        """
        return self.get_array_module().sqrt(self)

    def round(self, decimals=0, *args, **kwargs):
        """Round each element to the given number of decimal places.

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
        return round(self, decimals)

    def xs(
        self: TriangleProtocol,
        index_key:IndexLabel,
        level:IndexLabel | None = None,
        drop_level:bool = True):
        '''
        Mimics xs from pandas. key difference is that  this function only slices 
        the index, therefore axis is always 0 and not an argument in the function
        
        Main use case for this function is when slicing beyond the first field in 
        the index (such as LOB in the clrd dataset)
        '''
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
            result.index = pd.DataFrame(data=['Total'],columns=['Total'])
        return result

def add_triangle_agg_func(
        cls: Type[TrianglePandas],
        k: str,
        v: str
):
    """
    Aggregate Overrides in Triangle
    """

    def agg_func(
            self: Triangle,
            axis: str | int | None = None,
            *args,
            **kwargs
    ) -> Triangle | ndarray:
        """
        Applies the aggregation function specified by k from the outer function.

        Parameters
        ----------

        self: Triangle
            The triangle to which the aggregation function will be applied.

        axis: str | int | None
            The axis of the triangle to which the aggregation function will be applied.

        Returns
        -------

        An aggregated Triangle or ndarray.
        """
        keepdims = kwargs.get("keepdims", None)
        obj = self.copy()
        auto_sparse: bool = kwargs.pop("auto_sparse", True)
        if axis is None:
            if max(obj.shape) == 1:
                axis: int = 0
            else:
                axis: int = min([num for num, _ in enumerate(obj.shape) if _ != 1])
        else:
            axis = self._get_axis(axis)
        xp: ModuleType = obj.get_array_module()
        func: Callable = getattr(xp, v)
        kwargs.update({"keepdims": True})
        # Apply the function on the requested axis.
        obj.values = func(obj.values, axis=axis, *args, **kwargs)

        # Aggregation function will collapse a dimension, so
        # adjust the dimensions of the original object to match that of the aggregation.
        if axis == 0 and obj.values.shape[axis] == 1 and len(obj.kdims) > 1:
            obj.kdims = np.array([["(All)"] * len(obj.key_labels)])
        if axis == 1 and obj.values.shape[axis] == 1 and len(obj.vdims) > 1:
            obj.vdims = np.array([0])
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

    set_method(cls, agg_func, k)


def add_groupby_agg_func(cls, k: str, v: str):
    """Aggregate Overrides in GroupBy"""

    def agg_func(self, *args, **kwargs):
        from chainladder.utils import concat
        xp = self.obj.get_array_module()
        obj = self.obj.copy()
        auto_sparse = kwargs.pop("auto_sparse", True)
        if db and obj.array_backend == "sparse":
            _warn_dask_parallel_deprecated()

            def aggregate(i, obj, axis, v):
                return getattr(
                    obj.iloc.__getitem__(tuple([slice(None)] * axis + [i])), v
                )(axis, auto_sparse=False, keepdims=True)

            bag = db.from_sequence(self.groups.indices.values())
            bag = bag.map(aggregate, obj, self.axis, v)
            values = bag.compute(scheduler="threads")
        else:
            values = [
                getattr(
                    obj.iloc.__getitem__(
                        tuple([slice(None)] * self.axis + [i])), v
                )(self.axis, auto_sparse=False, keepdims=True)
                for i in self.groups.indices.values()
            ]
        obj = concat(values, axis=self.axis, ignore_index=True)
        group_index = self.groups.first().index
        if self.axis == 0:
            if isinstance(group_index, pd.MultiIndex):
                index = (
                    pd.DataFrame(
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
                obj.key_labels = index.columns.tolist()
                obj.kdims = index.values
        if self.axis == 1:
            obj.vdims = pd.DataFrame(group_index).values[:, 0]
        if self.axis == 2:
            odims = self.obj._to_datetime(
                pd.Series(self.groups.indices.keys()).to_frame(), [0]
            )
            obj.origin_grain = self.obj._get_grain(odims)
            split = obj.origin_grain.split("-")
            obj.origin_grain = {"2Q": "S"}.get(split[0], split[0])
            obj.odims = odims.values
        obj._set_slicers()
        if auto_sparse:
            obj = obj._auto_sparse()
        return obj

    set_method(
        cls=cls,
        func=agg_func,
        k=k
    )


def add_df_passthru(cls, k):
    """Pass Through of pandas functionality"""

    def df_passthru(self, *args, **kwargs):
        return getattr(pd.DataFrame, k)(self.to_frame(), *args, **kwargs)

    set_method(cls, df_passthru, k)


def set_method(
        cls: Type[TrianglePandas | TriangleGroupBy],
        func: Callable,
        k: str
) -> None:
    """
    Assigns methods to a class.

    Parameters
    ----------

    cls: Type[TrianglePandas | TriangleGroupBy]
        Class to be modified.

    func: Callable
        Method to be added to the class supplied to parameter cls.

    k: str
        Name of the method to be added to the class supplied to parameter cls.

    Returns
    -------

    None
    """
    func.__doc__ = "Refer to pandas for ``{}`` functionality.".format(k)
    func.__name__ = k
    setattr(cls, func.__name__, func)


df_passthru = (
    [
        "to_clipboard",
        "to_csv",
        "to_excel",
        "to_json",
        "to_html",
    ]
    + [
        "to_dict",
        "unstack",
        "pivot",
        "drop_duplicates",
        "describe",
        "melt",
    ]
    + [
        "pct_chg",
    ]
)
for item in df_passthru:
    add_df_passthru(TrianglePandas, item)

agg_funcs = ["sum", "mean", "median", "max", "min", "prod", "var"]
agg_funcs = agg_funcs + ["std", "cumsum", "quantile"]
for k in agg_funcs:
    add_groupby_agg_func(TriangleGroupBy, k, k)
agg_funcs = {item: "nan" + item for item in agg_funcs}
more_aggs = ["diff"]
agg_funcs = {**agg_funcs, **{item: item for item in more_aggs}}
for k, v in agg_funcs.items():
    add_triangle_agg_func(TrianglePandas, k, v)
