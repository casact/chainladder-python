# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.utility_functions import num_to_nan


class TriangleGroupBy:
    def __init__(self, obj, by, axis=0):
        self.obj = obj.copy()
        axis = self.obj._get_axis(axis)
        self.by = by
        if axis > 1:
            raise ValueError(
                "Use grain method to group by origin and development axes."
            )
        self.axis = self.obj._get_axis(axis)
        if self.axis == 0:
            self.groups = obj.index.groupby(by)
        if self.axis == 1:
            self.groups = pd.DataFrame(obj.columns).groupby(by)


class TrianglePandas:
    def to_frame(self, origin_as_datetime=False, keepdims=False, implicit_axis=False,
                 *args, **kwargs):
        """ Converts a triangle to a pandas.DataFrame.

        Parameters
        ----------
        origin_as_datetime : bool
            Whether the origin vector should be converted from PeriodIndex
            into a datetime dtype. Default is False.
        keepdims : bool
            If True, the triangle will be converted to a DataFrame with all
            dimensions intact.  The argument will force a consistent DataFrame
            format regardless of whether any dimensions are of length 1.
        implicit_axis : bool
            When keepdims is True, this denotes whether to include the implicit
            valuation axis in addition to the origin and development.
        Returns
        -------
            pandas.DataFrame representation of the Triangle.
        """
        axes = [num for num, item in enumerate(self.shape) if item > 1]
        if keepdims:
            is_val_tri = self.is_val_tri
            obj = self.val_to_dev().set_backend("sparse")
            out = pd.DataFrame(obj.index.iloc[obj.values.coords[0]])
            out["columns"] = obj.columns[obj.values.coords[1]]
            out["origin"] = obj.odims[obj.values.coords[2]]
            out["development"] = obj.ddims[obj.values.coords[3]]
            out["values"] = obj.values.data
            out = pd.pivot_table(
                out, index=obj.key_labels + ["origin", "development"], columns="columns"
            )
            out = out.reset_index().set_index(obj.key_labels)
            out.columns = ["origin", "development"] + list(
                out.columns.get_level_values(1)[2:]
            )

            valuation = pd.DataFrame(
                obj.valuation.values.reshape(obj.shape[-2:], order='F'),
                index=obj.odims, columns=obj.ddims
            ).unstack().rename('valuation').reset_index().rename(
                columns={'level_0': 'development', 'level_1': 'origin'})

            val_dict = dict(zip(list(zip(
                valuation['origin'], valuation['development'])),
                valuation['valuation']))
            out['valuation'] = out.apply(
                lambda x: val_dict[(x['origin'], x['development'])], axis=1)
            col_order = list(out.columns)
            if implicit_axis:
                col_order = ['origin', 'development', 'valuation'] + col_order[2:-1]
            else:
                if is_val_tri:
                    col_order = ['origin', 'valuation'] + col_order[2:-1]
                else:
                    col_order = ['origin', 'development'] + col_order[2:-1]
            return out[col_order]
        if self.shape[:2] == (1, 1):
            return self._repr_format(origin_as_datetime)
        elif len(axes) in [1, 2]:
            tri = np.squeeze(self.set_backend("numpy").values)
            axes_lookup = {
                0: self.kdims,
                1: self.vdims,
                2: self.origin,
                3: self.development,
            }
            if axes[0] == 0:
                idx = self.index.set_index(self.key_labels).index
            else:
                idx = axes_lookup[axes[0]]
            if len(axes) == 2:
                return pd.DataFrame(
                    tri, index=idx, columns=axes_lookup[axes[1]]
                ).fillna(0)
            if len(axes) == 1:
                return pd.Series(tri, index=idx).fillna(0)
        else:
            return self.to_frame(
                origin_as_datetime=origin_as_datetime, keepdims=True,
                implicit_axis=implicit_axis)

    def plot(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        return self.to_frame().plot(*args, **kwargs)

    def hvplot(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        df = self.to_frame()
        if type(df.index) == pd.PeriodIndex and len(df.columns) > 1:
            df.index = df.index.to_timestamp(how="s")
        return df.hvplot(*args, **kwargs)

    def _get_axis(self, axis):
        ax = {
            **{0: 0, 1: 1, 2: 2, 3: 3},
            **{-1: 3, -2: 2, -3: 1, -4: 0},
            **{"index": 0, "columns": 1, "origin": 2, "development": 3},
        }
        return ax.get(axis, 0)

    def dropna(self):
        """  Method that removes orgin/development vectors from edge of a
        triangle that are all missing values. This may come in handy for a
        new line of business that doesn't have origins/developments of an
        existing line in the same triangle.
        """
        obj = self.sum(axis=0).sum(axis=1)
        xp = obj.get_array_module()
        odim = list((xp.nansum(obj.values[0, 0, :], -1) != 0).astype("int"))
        min_odim = obj.origin[odim.index(1)]
        max_odim = obj.origin[::-1][odim[::-1].index(1)]
        if obj.shape[-1] != 1:
            ddim = list((xp.nansum(obj.values[0, 0, :], -2) != 0).astype("int"))
            ddim = obj.development[pd.Series(ddim).astype(bool)]
            obj = self[
                (self.development >= ddim.min()) & (self.development <= ddim.max())
            ]
            return obj[(self.origin >= min_odim) & (self.origin <= max_odim)]
        obj = self[(self.origin >= min_odim) & (self.origin <= max_odim)]
        return obj

    def drop(self, labels=None, axis=1):
        """ Drop specified labels from rows or columns.

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
        if axis == 1:
            return self[[item for item in self.columns if item not in labels]]
        else:
            raise NotImplementedError("drop only inpemented for column axis")

    @property
    def T(self):
        return self.to_frame().T

    def groupby(self, by, axis=0, *args, **kwargs):
        """ Group Triangle by index values.  If the triangle is convertable to a
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
        """ Append rows of other to the end of caller, returning a new object.

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

    def rename(self, axis, value):
        """ Alter axes labels.

        Parameters
        ----------
        axis: str or int
            A value of 0 <= axis <= 4 corresponding to axes 'index',
            'columns', 'origin', 'development' respectively.  Both the
            int and str representation can be used.
        value: list or str
            List of new labels to be assigned to the axis. List must be of
            same length of the specified axis.

        Returns
        -------
            Triangle with relabeled axis.
        """
        value = [value] if type(value) is str else value
        if axis == "index" or axis == 0:
            self.index = value
        if axis == "columns" or axis == 1:
            self.columns = value
        if axis == "origin" or axis == 2:
            self.origin = value
        if axis == "development" or axis == 3:
            self.development = value
        return self

    def astype(self, dtype, inplace=True):
        """ Copy of the array, cast to a specified type.

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
        obj = self.copy() if inplace is True else self
        obj.values = obj.values.astype(dtype)
        return obj

    def head(self, n=5):
        return self.iloc[:n]

    def tail(self, n=5):
        return self.iloc[-n:]

    def sort_index(self, *args, **kwargs):
        return self.iloc[self.index.sort_values(self.key_labels, *args, **kwargs).index]

    def exp(self):
        xp = self.get_array_module()
        obj = self.copy()
        obj.values = xp.exp(obj.values)
        return obj

    def log(self):
        xp = self.get_array_module()
        obj = self.copy()
        obj.values = xp.log(obj.values)
        return obj

    def minimum(self, other):
        obj = self.copy()
        xp = self.get_array_module()
        obj.values = xp.minimum(self.values, other)
        return obj

    def maximum(self, other):
        obj = self.copy()
        xp = self.get_array_module()
        obj.values = xp.maximum(self.values, other)
        return obj

    def sqrt(self):
        obj = self.copy()
        xp = self.get_array_module()
        obj.values = xp.sqrt(self.values)
        return obj


def add_triangle_agg_func(cls, k, v):
    """ Aggregate Overrides in Triangle """

    def agg_func(self, axis=None, *args, **kwargs):
        keepdims = kwargs.get("keepdims", None)
        obj = self.copy()
        auto_sparse = kwargs.pop("auto_sparse", True)
        if axis is None:
            if max(obj.shape) == 1:
                axis = 0
            else:
                axis = min([num for num, _ in enumerate(obj.shape) if _ != 1])
        else:
            axis = self._get_axis(axis)
        xp = obj.get_array_module()
        func = getattr(xp, v)
        kwargs.update({"keepdims": True})
        obj.values = func(obj.values, axis=axis, *args, **kwargs)
        if axis == 0 and obj.values.shape[axis] == 1 and len(obj.kdims) > 1:
            obj.kdims = np.array([["(All)"] * len(obj.key_labels)])
        if axis == 1 and obj.values.shape[axis] == 1 and len(obj.vdims) > 1:
            obj.vdims = np.array([0])
        if axis == 2 and obj.values.shape[axis] == 1 and len(obj.odims) > 1:
            obj.odims = obj.odims[0:1]
        if axis == 3 and obj.values.shape[axis] == 1 and len(obj.ddims) > 1:
            obj.ddims = pd.DatetimeIndex(
                [self.valuation_date], dtype="datetime64[ns]", freq=None
            )
        if auto_sparse:
            obj._set_slicers()
        obj.values = num_to_nan(obj.values)
        if not keepdims and obj.shape == (1, 1, 1, 1):
            return obj.values[0, 0, 0, 0]
        else:
            return obj

    set_method(cls, agg_func, k)


def add_groupby_agg_func(cls, k, v):
    """ Aggregate Overrides in GroupBy """

    def agg_func(self, *args, **kwargs):
        from chainladder.utils import concat
        from chainladder.methods import Chainladder

        xp = self.obj.get_array_module()
        obj = self.obj.copy()
        values = [
            getattr(obj.iloc.__getitem__(tuple([slice(None)] * self.axis + [i])), v)(
                self.axis, auto_sparse=False, keepdims=True
            )
            for i in self.groups.indices.values()
        ]
        obj = concat(values, axis=self.axis, ignore_index=True)
        if self.axis == 0:
            if isinstance(self.groups.dtypes.index, pd.MultiIndex):
                index = (
                    pd.DataFrame(
                        np.zeros(len(self.groups.dtypes.index)),
                        index=self.groups.dtypes.index,
                        columns=["_"],
                    )
                    .reset_index()
                    .iloc[:, :-1]
                )
                obj.index = index
            else:
                index = pd.DataFrame(self.groups.dtypes.index)
                obj.key_labels = index.columns.tolist()
                obj.kdims = index.values
        else:
            index = pd.DataFrame(self.groups.dtypes.index).values[:, 0]
        if self.axis == 1:
            obj.vdims = index
        obj._set_slicers()
        return obj

    set_method(cls, agg_func, k)


def add_df_passthru(cls, k):
    """Pass Through of pandas functionality """

    def df_passthru(self, *args, **kwargs):
        return getattr(pd.DataFrame, k)(self.to_frame(), *args, **kwargs)

    set_method(cls, df_passthru, k)


def set_method(cls, func, k):
    """ Assigns methods to a class """
    func.__doc__ = "Refer to pandas for ``{}`` functionality.".format(k)
    func.__name__ = k
    setattr(cls, func.__name__, func)


df_passthru = (
    ["to_clipboard", "to_csv", "to_excel", "to_json", "to_html",]
    + ["to_dict", "unstack", "pivot", "drop_duplicates", "describe", "melt",]
    + ["pct_chg", "round",]
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
