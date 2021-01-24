# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
import warnings
from chainladder.utils.utility_functions import num_to_nan


class TriangleDunders:
    """ Class that implements the dunder (double underscore) methods for the
        Triangle class
    """

    def _validate_arithmetic(self, other):
        """ Common functionality BEFORE arithmetic operations """
        if isinstance(other, TriangleDunders):
            obj, other = self._compatibility_check(self, other)
            obj.valuation_date = max(obj.valuation_date, other.valuation_date)
            obj, other = self._prep_index(obj, other)
            obj, other = self._prep_columns(obj, other)
            xp = obj.get_array_module()
            a, b = self.shape[-2:], other.shape[-2:]
            is_broadcastable = (
                a[0] == 1 or b[0] == 1 or np.all(other.odims == obj.odims)
            ) and (a[1] == 1 or b[1] == 1 or np.all(other.ddims == obj.ddims))
            if is_broadcastable:
                if len(other.odims) == 1 and len(obj.odims) > 1:
                    other.odims = obj.odims
                elif len(obj.odims) == 1 and len(other.odims) > 1:
                    obj.odims = other.odims
                if len(other.ddims) == 1 and len(obj.ddims) > 1:
                    other.ddims = obj.ddims
                elif len(obj.ddims) == 1 and len(other.ddims) > 1:
                    obj.ddims = other.ddims
                other = other.values
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
                new_shape = (self.shape[0], self.shape[1], len(odims), len(ddims))
                if obj.array_backend != "sparse":
                    other_arr = xp.zeros(new_shape)
                    other_arr[:] = xp.nan
                    other_arr[:, :, rol:roh, rdl:rdh] = other.values
                    obj_arr = xp.zeros(new_shape)
                    obj_arr[:] = xp.nan
                    obj_arr[:, :, lol:loh, ldl:ldh] = obj.values
                else:
                    obj_arr, other_arr = obj.values.copy(), other.values.copy()
                    other_arr.coords[2] = other_arr.coords[2] + rol
                    other_arr.coords[3] = other_arr.coords[3] + rdl
                    obj_arr.coords[2] = obj_arr.coords[2] + lol
                    obj_arr.coords[3] = obj_arr.coords[3] + ldl
                    other_arr.shape = obj_arr.shape = new_shape
                obj.odims = np.array(odims.index)
                if type(obj.ddims) == pd.DatetimeIndex:
                    obj.ddims = pd.DatetimeIndex(ddims.index)
                else:
                    obj.ddims = np.array(ddims.index)
                obj.values = obj_arr
                other = other_arr
        else:
            obj = self.copy()
        return obj, other

    def _arithmetic_cleanup(self, obj, other):
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
                "Triangle arithmetic requires both triangles to", "be the same grain."
            )
        return x, y

    def _prep_index(self, x, y):
        """ Preps index and column axes for arithmetic """
        # Set index axes for x, y
        def apply_axis(common, x, y):
            idx = (
                y.index[common]
                .merge(x.index[common].reset_index(), how="left", on=common)["index"]
                .values
            )
            x.values = x.values[idx]
            x.kdims = y.kdims
            x.key_labels = y.key_labels
            x.index = y.index
            return x

        if x.key_labels != y.key_labels:
            common = list(set(x.key_labels).intersection(set(y.key_labels)))
            if len(common) == len(x.key_labels):
                x = apply_axis(common, x, y) if len(x.index) > 1 else x
            elif len(common) == len(y.key_labels):
                y = apply_axis(common, y, x) if len(y.index) > 1 else y
            else:
                raise ValueError("Triangle arithmetic along index is ambiguous.")
        if (
            x.key_labels == y.key_labels
            and len(x) == len(y)
            and len(y) > 1
            and not np.all(x.index == y.index)
        ):
            y = y.loc[x.index]
        if len(x) == 1 and len(y) > 1:
            x.kdims = y.kdims
        return x, y

    def _prep_columns(self, x, y):
        x_backend, y_backend = x.array_backend, y.array_backend
        if len(x.columns) == 1 and len(y.columns) > 1:
            x.columns = y.columns
        elif len(y.columns) == 1 and len(x.columns) > 1:
            y.columns = x.columns
        elif len(y.columns) == 1 and len(x.columns) == 1 and x.columns != y.columns:
            y.columns = x.columns = [0]
        elif np.all(x.columns == y.columns):
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

    def __add__(self, other):
        obj, other = self._validate_arithmetic(other)
        xp = obj.get_array_module()
        obj.values = xp.nan_to_num(obj.values) + xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj, other)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        obj, other = self._validate_arithmetic(other)
        xp = obj.get_array_module()
        obj.values = xp.nan_to_num(obj.values) - xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj, other)

    def __rsub__(self, other):
        obj, other = self._validate_arithmetic(other)
        xp = obj.get_array_module()
        obj.values = xp.nan_to_num(other) - xp.nan_to_num(obj.values)
        return self._arithmetic_cleanup(obj, other)

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
        xp = obj.get_array_module()
        obj.values = obj.values * other
        return self._arithmetic_cleanup(obj, other)

    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    def __pow__(self, other):
        obj, other = self._validate_arithmetic(other)
        xp = obj.get_array_module()
        obj.values = xp.nan_to_num(obj.values) ** other
        return self._arithmetic_cleanup(obj, other)

    def __round__(self, other):
        obj = self.copy()
        xp = obj.get_array_module()
        obj.values = xp.nan_to_num(obj.values).round(other)
        return self._arithmetic_cleanup(obj, other)

    def __truediv__(self, other):
        obj, other = self._validate_arithmetic(other)
        xp = obj.get_array_module()
        obj.values = obj.values / other
        return self._arithmetic_cleanup(obj, other)

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
