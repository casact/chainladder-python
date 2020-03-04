# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
import copy

class TriangleDunders:
    ''' Class that implements the dunder (double underscore) methods for the
        Triangle class
    '''
    def _validate_arithmetic(self, other):
        ''' Common functionality BEFORE arithmetic operations '''
        obj = copy.deepcopy(self)
        xp = cp.get_array_module(obj.values)
        other = other if type(other) in [int, float] else copy.deepcopy(other)
        ddims = None
        odims = None
        if type(other) not in [int, float, np.float64, np.int64, xp.ndarray]:
            if len(self.vdims) != len(other.vdims):
                raise ValueError('Triangles must have the same number of ' +
                                 'columns')
            if len(self.kdims) != len(other.kdims):
                raise ValueError('Triangles must have the same number of ' +
                                 'index')
            if len(self.vdims) == 1:
                other.vdims = np.array([None])
            # If broadcasting doesn't work, then try union of origin/developments
            # before failure
            a, b = self.shape[-2:], other.shape[-2:]
            if not (a[0] == 1 or b[0] == 1 or a[0] == b[0]) or \
               not (a[1] == 1 or b[1] == 1 or a[1] == b[1]):
                ddims = pd.concat((pd.Series(self.ddims, index=self.ddims),
                                pd.Series(other.ddims, index=other.ddims)), axis=1)
                odims = pd.concat((pd.Series(self.odims, index=self.odims),
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
                obj.ddims =  ddims
                obj.odims =  odims
                obj.values = obj_arr
                other.values = other_arr
                obj._set_slicers()
                obj.valuation = obj._valuation_triangle()
                if hasattr(obj, '_nan_triangle_'):
                    # Force update on _nan_triangle at next access.
                    del obj._nan_triangle_
            other = other.values
        return obj, other

    def _arithmetic_cleanup(self, obj):
        ''' Common functionality AFTER arithmetic operations '''
        obj.values = obj.values * self._expand_dims(obj._nan_triangle())
        obj.values[obj.values == 0] = np.nan
        return obj

    def __add__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values) + xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values) - \
            xp.nan_to_num(other)
        return self._arithmetic_cleanup(obj)

    def __rsub__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(other) - \
            xp.nan_to_num(obj.values)
        return self._arithmetic_cleanup(obj)

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
        return self._arithmetic_cleanup(obj)

    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    def __pow__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values)**other
        return self._arithmetic_cleanup(obj)

    def __round__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values).round(other)
        return self._arithmetic_cleanup(obj)


    def __truediv__(self, other):
        xp = cp.get_array_module(self.values)
        obj, other = self._validate_arithmetic(other)
        obj.values = xp.nan_to_num(obj.values)/other
        return self._arithmetic_cleanup(obj)

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
