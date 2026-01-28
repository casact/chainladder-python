# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd

from chainladder.utils.utility_functions import (
    num_to_nan,
    concat
)

from chainladder.core.pandas import TriangleGroupBy
from chainladder.utils.sparse import sp

try:
    import dask.bag as db
except ImportError:
    db = None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

class TriangleDunders:
    """ Class that implements the dunder (double underscore) methods for the
        Triangle class
    """

    def _validate_arithmetic(self, other: Any) -> tuple:
        """ Common functionality BEFORE arithmetic operations """
        if isinstance(other, TriangleDunders):
            obj, other = self._compatibility_check(self, other)
            if obj.is_pattern != other.is_pattern:
                obj.is_pattern = other.is_pattern = False
            obj.valuation_date = max(obj.valuation_date, other.valuation_date)
            obj, other = self._prep_columns(obj, other)
            obj, other = self._prep_origin_development(obj, other)
            obj, other = self._prep_index(obj, other)
            if isinstance(other, TriangleDunders):
                other = other.values
        else:
            if isinstance(other, np.ndarray) and self.array_backend != 'numpy':
                obj = self.copy()
                other = obj.get_array_module().array(other)
            elif isinstance(other, sp) and self.array_backend != 'sparse':
                obj = self.set_backend('sparse')
            else:
                obj = self.copy()
        return obj, other

    def _arithmetic_cleanup(self, obj):
        """ Common functionality AFTER arithmetic operations """
        obj.values = obj.values * obj.get_array_module().nan_to_num(obj.nan_triangle)
        obj.values = num_to_nan(obj.values)
        return obj

    def _compatibility_check(self, x, y):
        from chainladder.utils.utility_functions import set_common_backend

        x, y = set_common_backend([x, y])
        if (
            x.origin_grain != y.origin_grain
            or (x.development_grain != y.development_grain and 
                min(x.shape[-1], y.shape[-1]) > 1)
        ):
            raise ValueError(
                "Triangle arithmetic requires both triangles to be the same grain."
            )
        return x, y

    def _prep_index(self, x, y):
        if x.kdims.shape[0] == 1 and y.kdims.shape[0] > 1:
            x.kdims = y.kdims
            x.key_labels = y.key_labels
            return x, y
        if x.kdims.shape[0] > 1 and y.kdims.shape[0] == 1:
            y.kdims = x.kdims
            y.key_labels = x.key_labels
            return x, y
        if x.kdims.shape[0] == y.kdims.shape[0] == 1 and x.key_labels != y.key_labels:
            kdims = x.kdims if len(x.key_labels) > len(y.key_labels) else y.kdims
            key_labels = x.key_labels if len(x.key_labels) > len(y.key_labels) else y.key_labels
            x.kdims = y.kdims = kdims
            x.key_labels = y.key_labels = key_labels
            return x, y

        # Use sets for faster operations
        x_labels = set(x.key_labels)
        y_labels = set(y.key_labels)
        common = x_labels.intersection(y_labels)

        if common == x_labels or common == y_labels:
            if x_labels != y_labels or x.kdims.shape[0] != y.kdims.shape[0]:
                x = x.groupby(list(common))
                y = y.groupby(list(common))
            elif x.kdims.shape[0] > 1 and not np.array_equal(x.kdims, y.kdims) and not x.index.equals(y.index):
                x = x.sort_index()
                try:
                    y = y.loc[x.index]
                except:
                    x = x.groupby(list(common))
                    y = y.groupby(list(common))
            return x, y

        if common != x_labels and common != y_labels:
            raise ValueError('Index broadcasting is ambiguous between ' + str(x_labels) + ' and ' + str(y_labels))

        return x, y

    def _prep_columns(self, x, y):
        x_backend, y_backend = x.array_backend, y.array_backend
        
        if len(x.columns) == 1 and len(y.columns) > 1:
            x.vdims = y.vdims
        elif len(y.columns) == 1 and len(x.columns) > 1:
            y.vdims = x.vdims
        elif len(y.columns) == len(x.columns) == 1 and x.columns != y.columns:
            y.vdims = x.vdims
        elif x.shape[1] == y.shape[1] and np.array_equal(x.columns, y.columns):
            return x, y
        else:            
            # Find columns to add to each triangle
            cols_to_add_to_x = [col for col in y.columns if col not in x.columns] 
            cols_to_add_to_y = [col for col in x.columns if col not in y.columns]

            # Start with case with no new columns, y simply has a different order
            new_x_cols = list(x.columns)

            # Then, if there are new columns, add them.
            if cols_to_add_to_x:
                new_x_cols = list(x.columns) + list(cols_to_add_to_x)
                x = x.reindex(columns=new_x_cols, fill_value=0)

            if cols_to_add_to_y:
                new_y_cols = list(y.columns) + list(cols_to_add_to_y)
                y = y.reindex(columns=new_y_cols, fill_value=0)

            # Ensure both triangles have the same column order
            x = x[new_x_cols]
            y = y[new_x_cols]
        
        # Reset backends only if they've changed
        if x.array_backend != x_backend:
            x = x.set_backend(x_backend, inplace=True)
        if y.array_backend != y_backend:
            y = y.set_backend(y_backend, inplace=True)
        
        return x, y

    def _prep_origin_development(self, obj, other):
        xp = obj.get_array_module()
        is_broadcastable = all(
            (m == n) or (m == 1) or (n == 1)
            for m, n in zip(obj.shape[-2:][::-1], other.shape[-2:][::-1]))
        if (len(obj.odims), len(obj.ddims)) == (len(other.odims), len(other.ddims)):
            if np.all(obj.ddims != other.ddims) and len(obj.ddims) > 1:
                is_broadcastable = False
            if np.all(obj.odims != other.odims) and len(obj.odims) > 1:
                is_broadcastable = False
        if len(other.odims) == 1 and len(obj.odims) > 1:
            other.odims = obj.odims
            other.origin_grain = obj.origin_grain
        elif len(obj.odims) == 1 and len(other.odims) > 1:
            obj.odims = other.odims
            obj.origin_grain = other.origin_grain
        if len(other.ddims) == 1 and len(obj.ddims) > 1:
            other.ddims = obj.ddims
            other.development_grain = obj.development_grain
        elif len(obj.ddims) == 1 and len(other.ddims) > 1:
            obj.ddims = other.ddims
            obj.development_grain = other.development_grain
        if not is_broadcastable:
            # If broadcasting doesn't work, union axes similar to pandas
            ddims = pd.concat(
                (
                    pd.Series(obj.ddims, index=obj.ddims),
                    pd.Series(other.ddims, index=other.ddims),
                ),
                axis=1,
            ).sort_index()
            odims = pd.concat(
                (
                    pd.Series(obj.odims, index=obj.odims),
                    pd.Series(other.odims, index=other.odims),
                ),
                axis=1,
            ).sort_index()
            o_arr0, o_arr1 = odims[0].isna().values, odims[1].isna().values
            d_arr0, d_arr1 = ddims[0].isna().values, ddims[1].isna().values
            # rol = right hand side, origin, lower
            rol = int(np.where(~o_arr1 == 1)[0].min())
            roh = int(np.where(~o_arr1 == 1)[0].max() + 1)
            rdl = int(np.where(~d_arr1 == 1)[0].min())
            rdh = int(np.where(~d_arr1 == 1)[0].max() + 1)
            lol = int(np.where(~o_arr0 == 1)[0].min())
            loh = int(np.where(~o_arr0 == 1)[0].max() + 1)
            ldl = int(np.where(~d_arr0 == 1)[0].min())
            ldh = int(np.where(~d_arr0 == 1)[0].max() + 1)
            if obj.array_backend != "sparse":
                other_arr = xp.zeros((other.shape[0], other.shape[1], len(odims), len(ddims)))
                other_arr[:] = xp.nan
                other_arr[:, :, rol:roh, rdl:rdh] = other.values
                obj_arr = xp.zeros((self.shape[0], self.shape[1], len(odims), len(ddims)))
                obj_arr[:] = xp.nan
                obj_arr[:, :, lol:loh, ldl:ldh] = obj.values
            else:
                obj_arr, other_arr = obj.values.copy(), other.values.copy()
                other_arr.coords[2] = other_arr.coords[2] + rol
                other_arr.coords[3] = other_arr.coords[3] + rdl
                obj_arr.coords[2] = obj_arr.coords[2] + lol
                obj_arr.coords[3] = obj_arr.coords[3] + ldl
                other_arr.shape = (other.shape[0], other.shape[1], len(odims), len(ddims))
                obj_arr.shape = (self.shape[0], self.shape[1], len(odims), len(ddims))
            obj.odims = np.array(odims.index)
            if type(obj.ddims) == pd.DatetimeIndex:
                obj.ddims = pd.DatetimeIndex(ddims.index)
            else:
                obj.ddims = np.array(ddims.index)
            obj.values = obj_arr
            other.values = other_arr
        return obj, other

    @staticmethod
    def _slice_or_nan(obj, other, k):
        """ Return a broadcastable slice or NaNs for missing slices """
        if k in obj.groups.indices.keys():
            return obj.obj.iloc[obj.groups.indices[k]]
        else:
            other = other.obj.iloc[other.groups.indices[k]]
            new_obj = obj.obj.iloc[:1] * 0
            labels = list(set(other.key_labels).intersection(set(new_obj.key_labels)))
            new_idx = other.index.set_index(labels).join(
                new_obj.index.set_index(labels)).reset_index()
            new_idx = new_idx[new_obj.key_labels].iloc[-1:]
            new_obj.kdims = new_idx.values
            new_obj.key_labels = list(new_idx.columns)
            return new_obj

    @staticmethod
    def _get_key_union(obj, other):
        return set(list(obj.groups.indices.keys()) +
                   list(other.groups.indices.keys()))

    def _arithmetic_mapper(self, obj, other, f):
        """ Use Dask if available, otherwise basic list comprehension """
        if db and obj.obj.array_backend == 'sparse':
            bag = db.from_sequence(self._get_key_union(obj, other))
            bag = bag.map(f, self, obj, other)
            c = bag.compute(scheduler='threads')
        else:
            c = [f(k, self, obj, other) for k in self._get_key_union(obj, other)]
        return concat(c, 0).sort_index()

    def __add__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            def f(k, self, obj, other):
                return (self._slice_or_nan(obj, other, k) +
                        self._slice_or_nan(other, obj, k))
            obj = self._arithmetic_mapper(obj, other, f)
        else:
            xp = obj.get_array_module()
            obj.values = xp.nan_to_num(obj.values) + xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            def f(k, self, obj, other):
                return (self._slice_or_nan(obj, other, k) -
                        self._slice_or_nan(other, obj, k))
            obj = self._arithmetic_mapper(obj, other, f)
        else:
            xp = obj.get_array_module()
            obj.values = xp.nan_to_num(obj.values) - xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj)

    def __rsub__(self, other):
        obj, other = self._validate_arithmetic(other)
        xp = obj.get_array_module()
        obj.values = xp.nan_to_num(other) - xp.nan_to_num(obj.values)
        return self._arithmetic_cleanup(obj)

    def __len__(self):
        return self.shape[0]

    def __neg__(self):
        obj = self.copy()
        obj.values = -obj.values
        return obj

    def __pos__(self):
        return self

    def __abs__(self):
        obj = self.copy()
        obj.values = abs(obj.values)
        return obj

    def __mul__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            def f(k, self, obj, other):
                return (self._slice_or_nan(obj, other, k) *
                        self._slice_or_nan(other, obj, k))
            obj = self._arithmetic_mapper(obj, other, f)
        else:
            xp = obj.get_array_module()
            obj.values = obj.values * other
        return obj

    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    def __pow__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            def f(k, self, obj, other):
                return (self._slice_or_nan(obj, other, k) **
                        self._slice_or_nan(other, obj, k))
            obj = self._arithmetic_mapper(obj, other, f)
        else:
            xp = obj.get_array_module()
            obj.values = xp.nan_to_num(obj.values) ** other
        return obj

    def __round__(self, other):
        obj = self.copy()
        xp = obj.get_array_module()
        obj.values = xp.round(obj.values, other)
        return obj

    def __truediv__(self, other: Any):
        """
        Implements the true division operator, /, on the triangle.

        Parameters
        ----------

        other: Any
        The thing that divides the triangle.
        """
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            def f(k, self, obj, other):
                return (self._slice_or_nan(obj, other, k) /
                        self._slice_or_nan(other, obj, k))
            obj = self._arithmetic_mapper(obj, other, f)
        else:
            obj.values = obj.values / other
        return obj

    def __rtruediv__(self, other):
        obj = self.copy()
        obj.values = other / self.values
        obj.values = num_to_nan(obj.values)
        return obj

    def __eq__(self, other):
        if not isinstance(other, TriangleDunders):
            return False
        from chainladder import options

        backend = options.ARRAY_PRIORITY[
            min(
                [
                    options.ARRAY_PRIORITY.index(x)
                    for x in [self.array_backend, other.array_backend]
                ]
            )
        ]
        x, y = self.set_backend(backend), other.set_backend(backend)
        xp = x.get_array_module()
        return xp.all(xp.nan_to_num(x.values) == xp.nan_to_num(y.values))

    def __contains__(self, value):
        return self.__dict__.get(value, None) is not None

    def __lt__(self, value):
        obj = self.copy()
        xp = self.get_array_module()
        obj.values = xp.nan_to_num(obj.values) < xp.nan_to_num(value)
        return obj

    def __le__(self, value):
        obj = self.copy()
        xp = self.get_array_module()
        obj.values = xp.nan_to_num(obj.values) < xp.nan_to_num(value)
        return obj
