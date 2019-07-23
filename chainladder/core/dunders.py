import pandas as pd
import numpy as np
import copy


class TriangleDunders:
    ''' Class that implements the dunder (double underscore) methods for the
        Triangle class
    '''
    def _validate_arithmetic(self, other):
        ''' Common functionality BEFORE arithmetic operations '''
        obj = copy.deepcopy(self)
        other = other if type(other) in [int, float] else copy.deepcopy(other)
        ddims = None
        odims = None
        if type(other) not in [int, float, np.float64, np.int64]:
            if len(self.vdims) != len(other.vdims):
                raise ValueError('Triangles must have the same number of ' +
                                 'columns')
            if len(self.kdims) != len(other.kdims):
                raise ValueError('Triangles must have the same number of ' +
                                 'index')
            if len(self.vdims) == 1:
                other.vdims = np.array([None])
            # If broadcasting doesn't work, then try intersecting before
            # failure
            a, b = self.shape[-2:], other.shape[-2:]
            if not (a[0] == 1 or b[0] == 1 or a[0] == b[0]) and \
               not (a[1] == 1 or b[1] == 1 or a[1] == b[1]):
                ddims = set(self.ddims).intersection(set(other.ddims))
                odims = set(self.odims).intersection(set(other.odims))
                # Need to set string vs int type-casting
                odims = pd.PeriodIndex(np.array(list(odims)),
                                       freq=self.origin_grain)
                obj = obj[obj.origin.isin(odims)][obj.development.isin(ddims)]
                other = other[other.origin.isin(odims)][other.development.isin(ddims)]
                obj.odims = np.sort(np.array(list(odims)))
                obj.ddims = np.sort(np.array(list(ddims)))
            other = other.values
        return obj, other

    def _arithmetic_cleanup(self, obj):
        ''' Common functionality AFTER arithmetic operations '''
        obj.values = obj.values * self.expand_dims(obj.nan_triangle())
        obj.values[obj.values == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    def __add__(self, other):
        obj, other = self._validate_arithmetic(other)
        obj.values = np.nan_to_num(obj.values) + np.nan_to_num(other)
        return self._arithmetic_cleanup(obj)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        obj, other = self._validate_arithmetic(other)
        obj.values = np.nan_to_num(obj.values) - \
            np.nan_to_num(other)
        return self._arithmetic_cleanup(obj)

    def __rsub__(self, other):
        obj, other = self._validate_arithmetic(other)
        obj.values = np.nan_to_num(other) - \
            np.nan_to_num(obj.values)
        return self._arithmetic_cleanup(obj)

    def __len__(self):
        return self.shape[0]

    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.values = -obj.values
        return obj

    def __pos__(self):
        return self

    def __mul__(self, other):
        obj, other = self._validate_arithmetic(other)
        obj.values = np.nan_to_num(obj.values)*other
        return self._arithmetic_cleanup(obj)

    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    def __truediv__(self, other):
        obj, other = self._validate_arithmetic(other)
        obj.values = np.nan_to_num(obj.values)/other
        return self._arithmetic_cleanup(obj)

    def __rtruediv__(self, other):
        obj = copy.deepcopy(self)
        obj.values = other / self.values
        obj.values[obj.values == 0] = np.nan
        return obj

    def __eq__(self, other):
        if np.all(np.nan_to_num(self.values) ==
           np.nan_to_num(other.values)):
            return True
        else:
            return False
