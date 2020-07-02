# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
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
                    obj.valuation = other.valuation
                if len(other.ddims) == 1 and len(obj.ddims) > 1:
                    other.ddims = obj.ddims
                elif len(obj.ddims) == 1 and len(other.ddims) > 1:
                    obj.ddims = other.ddims
                    obj.valuation = other.valuation
                other = other.values
            if not is_broadcastable:
                # If broadcasting doesn't work, union axes similar to pandas
                ddims = pd.concat(
                    (pd.Series(obj.ddims, index=obj.ddims),
                     pd.Series(other.ddims, index=other.ddims)), axis=1)
                odims = pd.concat(
                    (pd.Series(obj.odims, index=obj.odims),
                     pd.Series(other.odims, index=other.odims)), axis=1)
                other_arr = xp.zeros(
                    (other.shape[0], other.shape[1], len(odims), len(ddims)))
                other_arr[:] = xp.nan
                o_arr1 = odims[1].isna().values
                o_arr0 = odims[0].isna().values
                d_arr1 = ddims[1].isna().values
                d_arr0 = ddims[0].isna().values
                if xp == cp:
                    o_arr1 = cp.array(o_arr1)
                    o_arr0 = cp.array(o_arr0)
                    d_arr1 = cp.array(d_arr1)
                    d_arr0 = cp.array(d_arr0)
                ol = int(xp.where(~o_arr1 == 1)[0].min())
                oh = int(xp.where(~o_arr1 == 1)[0].max()+1)
                if np.any(self.ddims != other.ddims):
                    dl = int(xp.where(~d_arr1 == 1)[0].min())
                    dh = int(xp.where(~d_arr1 == 1)[0].max()+1)
                    other_arr[:, :, ol:oh, dl:dh] = other.values
                else:
                    other_arr[:, :, ol:oh, :] = other.values
                obj_arr = xp.zeros(
                    (self.shape[0], self.shape[1], len(odims), len(ddims)))
                obj_arr[:] = xp.nan
                ol = int(xp.where(~o_arr0 == 1)[0].min())
                oh = int(xp.where(~o_arr0 == 1)[0].max()+1)
                if np.any(self.ddims != other.ddims):
                    dl = int(xp.where(~d_arr0 == 1)[0].min())
                    dh = int(xp.where(~d_arr0 == 1)[0].max()+1)
                    obj_arr[:, :, ol:oh, dl:dh] = self.values
                else:
                    obj_arr[:, :, ol:oh, :] = self.values
                odims = np.array(odims.index)
                ddims = np.array(ddims.index)
                obj.ddims = ddims
                obj.odims = odims
                obj.values = obj_arr
                obj.valuation = obj._valuation_triangle()
                other = other_arr
        return obj, other

    def _arithmetic_cleanup(self, obj, other):
        ''' Common functionality AFTER arithmetic operations '''
        obj.values = obj.values * obj._expand_dims(obj._nan_triangle())
        obj.values[obj.values == 0] = np.nan
        return obj

    def _compatibility_check(self, x, y):
        if x.key_labels != y.key_labels:
            raise ValueError("Triangle arithmetic requires both triangles to have the same key_labels.")
        if x.origin_grain != y.origin_grain or x.development_grain != y.development_grain:
            raise ValueError("Triangle arithmetic requires both triangles to be the same grain.")
        #if x.is_val_tri != y.is_val_tri:
        #    raise ValueError("Triangle arithmetic cannot be performed between a development triangle and a valuation Triangle.")
        if x.is_cumulative != y.is_cumulative:
            warnings.warn('Arithmetic is being performed between an incremental triangle and a cumulative triangle.')

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
        elif len(y.index) == 1 and len(x.index) == 1:
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
        obj.values = xp.nan_to_num(obj.values)/other
        return self._arithmetic_cleanup(obj, other)

    def __rtruediv__(self, other):
        obj = copy.deepcopy(self)
        obj.values = other / self.values
        obj.values[obj.values == 0] = np.nan
        return obj

    def __eq__(self, other):
        xp = cp.get_array_module(self.values)
        if xp.all(xp.nan_to_num(self.values) ==
           xp.nan_to_num(other.values)):
            return True
        else:
            return False

    def __contains__(self, value):
        if self.__dict__.get(value, None) is None:
            return False
        return True
