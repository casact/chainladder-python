# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
import warnings
from chainladder.utils.utility_functions import num_to_nan, concat
from chainladder.core.pandas import TriangleGroupBy
from chainladder.utils.sparse import sp

class TriangleDunders:
    """ Class that implements the dunder (double underscore) methods for the
        Triangle class
    """

    def _validate_arithmetic(self, other):
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
            or x.development_grain != y.development_grain
        ):
            raise ValueError(
                "Triangle arithmetic requires both triangles to be the same grain."
            )
        return x, y

    def _prep_index(self, x, y):
        """ Preps index and column axes for arithmetic """
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
            y.kdims = x.kdims = kdims
            return x, y
        if x.key_labels != y.key_labels:
            common = list(set(x.key_labels).intersection(set(y.key_labels)))
            x = x.groupby(common)
            y = y.groupby(common)
            return x, y
        if (
            x.key_labels == y.key_labels
            and x.kdims.shape[0] == y.kdims.shape[0]
            and y.kdims.shape[0] > 1
            and not x.kdims is y.kdims
            and not x.index.equals(y.index)
        ):
            x = x.sort_index()
            y = y.loc[x.index]
        return x, y

    def _prep_columns(self, x, y):
        x_backend, y_backend = x.array_backend, y.array_backend
        if len(x.columns) == 1 and len(y.columns) > 1:
            x.vdims = y.vdims
        elif len(y.columns) == 1 and len(x.columns) > 1:
            y.vdims = x.vdims
        elif len(y.columns) == 1 and len(x.columns) == 1 and x.columns != y.columns:
            y.vdims = x.vdims
        elif x.shape[1] == y.shape[1] and np.all(x.columns == y.columns):
            pass
        else:
            col_union = list(x.columns) + [
                item for item in y.columns if item not in x.columns
            ]
            for item in [item for item in col_union if item not in x.columns]:
                x[item] = 0
            x = x[col_union]
            for item in [item for item in col_union if item not in y.columns]:
                y[item] = 0
            y = y[col_union]
        x, y = (
            x.set_backend(x_backend, inplace=True),
            y.set_backend(y_backend, inplace=True),
        )
        return x, y

    def _prep_origin_development(self, obj, other):
        xp = obj.get_array_module()
        is_broadcastable = all(
            (m == n) or (m == 1) or (n == 1)
            for m, n in zip(obj.shape[-2:][::-1], other.shape[-2:][::-1]))
        if len(other.odims) == 1 and len(obj.odims) > 1:
            other.odims = obj.odims
        elif len(obj.odims) == 1 and len(other.odims) > 1:
            obj.odims = other.odims
        if len(other.ddims) == 1 and len(obj.ddims) > 1:
            other.ddims = obj.ddims
        elif len(obj.ddims) == 1 and len(other.ddims) > 1:
            obj.ddims = other.ddims
        if not is_broadcastable:
            # If broadcasting doesn't work, union axes similar to pandas
            ddims = pd.concat(
                (
                    pd.Series(obj.ddims, index=obj.ddims),
                    pd.Series(other.ddims, index=other.ddims),
                ),
                axis=1,
            )
            odims = pd.concat(
                (
                    pd.Series(obj.odims, index=obj.odims),
                    pd.Series(other.odims, index=other.odims),
                ),
                axis=1,
            )
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

    def __add__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            if len(obj.obj) < len(other.obj):
                obj, other = other, obj
            obj = concat(
                [obj.obj.iloc[v] + other.obj.iloc[other.groups.indices[k]].values
                for k, v in obj.groups.indices.items()], 0
            ).sort_index()
        else:
            xp = obj.get_array_module()
            obj.values = xp.nan_to_num(obj.values) + xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            if len(obj.obj) < len(other.obj):
                obj = concat(
                    [-other.obj.iloc[other.groups.indices[k]] + obj.obj.iloc[v].values
                    for k, v in obj.groups.indices.items()], 0
                ).sort_index()
            else:
                obj = concat(
                    [obj.obj.iloc[v] - other.obj.iloc[other.groups.indices[k]].values
                    for k, v in obj.groups.indices.items()], 0
                ).sort_index()
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
            if len(obj.obj) < len(other.obj):
                obj, other = other, obj
            obj = concat(
                [obj.obj.iloc[v]*other.obj.iloc[other.groups.indices[k]].values
                for k, v in obj.groups.indices.items()], 0
            ).sort_index()
        else:
            xp = obj.get_array_module()
            obj.values = obj.values * other
        return obj

    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    def __pow__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            if len(obj.obj) < len(other.obj):
                obj = concat(
                    [obj.obj.iloc[v].values ** other.obj.iloc[other.groups.indices[k]]
                    for k, v in obj.groups.indices.items()], 0
                ).sort_index()
            else:
                obj = concat(
                    [obj.obj.iloc[v] ** other.obj.iloc[other.groups.indices[k]].values
                    for k, v in obj.groups.indices.items()], 0
                ).sort_index()
        else:
            xp = obj.get_array_module()
            obj.values = xp.nan_to_num(obj.values) ** other
        return obj

    def __round__(self, other):
        obj = self.copy()
        xp = obj.get_array_module()
        obj.values = xp.nan_to_num(obj.values).round(other)
        return obj

    def __truediv__(self, other):
        obj, other = self._validate_arithmetic(other)
        if isinstance(obj, TriangleGroupBy):
            if len(obj.obj) < len(other.obj):
                obj = concat(
                    [(1 / other.obj.iloc[other.groups.indices[k]])*obj.obj.iloc[v].values
                    for k, v in obj.groups.indices.items()], 0
                ).sort_index()
            else:
                obj = concat(
                    [obj.obj.iloc[v] / other.obj.iloc[other.groups.indices[k]].values
                    for k, v in obj.groups.indices.items()], 0
                ).sort_index()
        else:
            xp = obj.get_array_module()
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
        from chainladder import ARRAY_PRIORITY

        backend = ARRAY_PRIORITY[
            min(
                [
                    ARRAY_PRIORITY.index(x)
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

    def __contains__(self, value):
        if self.__dict__.get(value, None) is None:
            return False
        return True
