# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
import copy
import warnings


class TriangleDunders:
    ''' Class that implements the dunder (double underscore) methods for the
        Triangle class
    '''
    def _validate_arithmetic(self, other):
        ''' Common functionality BEFORE arithmetic operations '''
        obj = copy.deepcopy(self)
        xp = cp.get_array_module(obj.values)
        other = other if type(other) in [int, float] else copy.deepcopy(other)
        if isinstance(other, TriangleDunders):
            self._compatibility_check(obj, other)
            obj.valuation_date = max(obj.valuation_date, other.valuation_date)
            obj, other = self._prep_index_columns(obj, other)
            a, b = self.shape[-2:], other.shape[-2:]
            is_broadcastable = (
                (a[0] == 1 or b[0] == 1 or np.all(other.odims == obj.odims)) and
                (a[1] == 1 or b[1] == 1 or np.all(other.ddims == obj.ddims)))
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
                    (pd.Series(obj.ddims, index=obj.ddims),
                     pd.Series(other.ddims, index=other.ddims)), axis=1)
                odims = pd.concat(
                    (pd.Series(obj.odims, index=obj.odims),
                     pd.Series(other.odims, index=other.odims)), axis=1)
                o_arr0, o_arr1 = odims[0].isna().values, odims[1].isna().values
                d_arr0, d_arr1 = ddims[0].isna().values, ddims[1].isna().values
                # rol = right hand side, origin, lower
                rol = int(np.where(~o_arr1 == 1)[0].min())
                roh = int(np.where(~o_arr1 == 1)[0].max()+1)
                rdl = int(np.where(~d_arr1 == 1)[0].min())
                rdh = int(np.where(~d_arr1 == 1)[0].max()+1)
                lol = int(np.where(~o_arr0 == 1)[0].min())
                loh = int(np.where(~o_arr0 == 1)[0].max()+1)
                ldl = int(np.where(~d_arr0 == 1)[0].min())
                ldh = int(np.where(~d_arr0 == 1)[0].max()+1)
                new_shape = (self.shape[0], self.shape[1], len(odims), len(ddims))
                if xp != sp:
                    other_arr = xp.zeros(new_shape)
                    other_arr[:] = xp.nan
                    other_arr[:, :, rol:roh, rdl:rdh] = other.values
                    obj_arr = xp.zeros(new_shape)
                    obj_arr[:] = xp.nan
                    obj_arr[:, :, lol:loh, ldl:ldh] = self.values
                else:
                    obj_arr, other_arr = obj.values, other.values
                    other_arr.coords[2] = other_arr.coords[2] + rol
                    other_arr.coords[3] = other_arr.coords[3] + rdl
                    obj_arr.coords[2] = obj_arr.coords[2] + lol
                    obj_arr.coords[3] = obj_arr.coords[3] + ldl
                    other_arr.shape = obj_arr.shape = new_shape
                obj.odims = np.array(odims.index)
                obj.ddims = np.array(ddims.index)
                obj.values = obj_arr
                other = other_arr
        return obj, other

    def _arithmetic_cleanup(self, obj, other):
        ''' Common functionality AFTER arithmetic operations '''
        from chainladder.utils.utility_functions import num_to_nan
        xp = cp.get_array_module(obj.values)
        if xp != sp:
            obj.values = obj.values * obj._expand_dims(obj.nan_triangle)
        obj.num_to_nan()
        return obj

    def _compatibility_check(self, x, y):
        if x.key_labels != y.key_labels:
            raise ValueError("Triangle arithmetic requires both triangles to have the same key_labels.")
        if x.origin_grain != y.origin_grain or x.development_grain != y.development_grain:
            raise ValueError("Triangle arithmetic requires both triangles to be the same grain.")
        #if x.is_val_tri != y.is_val_tri:
        #    raise ValueError("Triangle arithmetic cannot be performed between a development triangle and a valuation Triangle.")
        #if x.is_cumulative != y.is_cumulative:
        #    warnings.warn('Arithmetic is being performed between an incremental triangle and a cumulative triangle.')

    def _prep_index_columns(self, x, y):
        """ Preps index and column axes for arithmetic """
        # Union columns
        if len(x.columns) == 1 and len(y.columns) > 1:
            x.columns = y.columns
        elif len(y.columns) == 1 and len(x.columns) > 1:
            y.columns = x.columns
        elif len(y.columns) == 1 and len(x.columns) == 1:
            y.columns = x.columns = [0]
        elif np.all(x.columns == y.columns):
            pass
        else:
            col_union = list(x.columns) + \
                [item for item in y.columns if item not in x.columns]
            for item in [item for item in col_union if item not in x.columns]:
                x[item] = 0
            x = x[col_union]
            for item in [item for item in col_union if item not in y.columns]:
                y[item] = 0
            y = y[col_union]
        # Union index
        if len(x.index) == 1 and len(y.index) > 1:
            x.kdims = y.kdims
        elif len(y.index) == 1 and len(x.index) > 1:
            y.kdims = x.kdims
        elif len(y.index) == 1 and len(x.index) == 1 and np.all(x.index != y.index):
            y.kdims = x.kdims = np.array([0])
        elif np.all(x.index == y.index):
            pass
        else:
            x_index = x.index.set_index(x.key_labels)
            y_index = y.index.set_index(y.key_labels)
            ind_union = (x_index + y_index).index
            not_in_x = pd.DataFrame(set(ind_union)-set(x_index.index), columns=x.key_labels)
            not_in_y = pd.DataFrame(set(ind_union)-set(y_index.index), columns=y.key_labels)

            y.values = np.append(
                y.values, np.repeat((y.iloc[-1:]*0).values, len(not_in_y), 0), 0)
            y.kdims = y.index.append(not_in_y).values
            y._set_slicers()
            y = y.loc[ind_union]
            x.values = np.append(
                x.values, np.repeat((x.iloc[-1:]*0).values, len(not_in_x), 0), 0)
            x.kdims = x.index.append(not_in_x).values
            x._set_slicers()
            x = x.loc[ind_union]
        return x, y

    def __add__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values) + xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj, other)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values) - \
            xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj, other)

    def __rsub__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(other) - \
            xp.nan_to_num(obj.values)
        return self._arithmetic_cleanup(obj, other)

    def __len__(self):
        return self.shape[0]

    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.values = -obj.values
        return obj

    def __pos__(self):
        return self

    def __abs__(self):
        obj = copy.deepcopy(self)
        obj.values = abs(obj.values)
        return obj

    def __mul__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values)*other
        return self._arithmetic_cleanup(obj, other)

    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    def __pow__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values)**other
        return self._arithmetic_cleanup(obj, other)

    def __round__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values).round(other)
        return self._arithmetic_cleanup(obj, other)


    def __truediv__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values) / other
        return self._arithmetic_cleanup(obj, other)

    def __rtruediv__(self, other):
        obj = copy.deepcopy(self)
        obj.values = other / self.values
        obj.num_to_nan()
        return obj

    def __eq__(self, other):
        xp = cp.get_array_module(self.values)
        return xp.all(xp.nan_to_num(self.values) == xp.nan_to_num(other.values))

    def __contains__(self, value):
        return self.__dict__.get(value, None) is not None
