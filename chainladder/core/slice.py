# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pandas as pd
import numpy as np
from sparse import _slicing
from chainladder.utils.utility_functions import num_to_nan

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle

class _LocBase:
    """
    Base class for pandas style loc/iloc indexing.
    """

    def __init__(self, obj):
        self.obj = obj

    def get_idx(self, idx):
        """ Returns a slice of the original Triangle """
        obj = self.obj.copy()
        i_idx = _LocBase._contig_slice(idx[0])
        c_idx = _LocBase._contig_slice(idx[1])
        o_idx = _LocBase._contig_slice(idx[2])
        d_idx = _LocBase._contig_slice(idx[3])
        if type(o_idx) != slice or type(d_idx) != slice:
            raise ValueError("Fancy indexing on origin/development is not supported.")
        if type(i_idx) is slice or type(c_idx) is slice:
            obj.values = obj.values[i_idx, c_idx, o_idx, d_idx]
        else:
            obj.values = obj.values[i_idx, :, o_idx, d_idx][:, c_idx, ...]
        obj.kdims = obj.kdims[i_idx]
        obj.vdims = obj.vdims[c_idx]
        obj.odims, obj.ddims = obj.odims[o_idx], obj.ddims[d_idx]
        obj.iloc, obj.loc = Ilocation(obj), Location(obj)
        obj.valuation_date = np.minimum(obj.valuation.max(), obj.valuation_date)
        return obj

    @staticmethod
    def _contig_slice(arr):
        """ Try to make a contiguous slicer from an array of indices """
        if type(arr) is slice:
            return arr
        if type(arr) in [int, np.int64, np.int32]:
            arr = [arr]
        if len(arr) == 1:
            return slice(arr[0], arr[0] + 1)
        diff = np.diff(arr)
        if len(diff) == 0:
            raise ValueError("Slice returns empty Triangle")
        if max(diff) == min(diff):
            step = max(diff)
        else:
            return arr
        step = None if step == 1 else step
        min_arr = None if min(arr) == 0 else min(arr)
        max_arr = max(arr) + 1
        if step and step < 0:
            min_arr, max_arr = max_arr - 1, min_arr - 1 if min_arr else min_arr
        return slice(min_arr, max_arr, step)

    def __setitem__(self, key, values):
        if self.obj.array_backend == "sparse":
            raise ValueError('Setting values with sparse backend requires .at or .iat')
        if isinstance(values, TriangleSlicer):
            values = values.values
        key = tuple(
            [slice(item, item + 1) if type(item) is int else item for item in key]
        )
        self.obj.values.__setitem__(self._normalize_index(key), values)

    def _normalize_index(self, key):
        key = _slicing.normalize_index(key, self.obj.shape)
        l = []
        for n, i in enumerate(key):
            if type(i) is slice:
                start = i.start if i.start > 0 else None
                stop = i.stop if i.stop > -1 else None
                stop = None if stop == self.obj.shape[n] else stop
                step = None if start is None and stop is None else i.step
                l.append(slice(start, stop, step))
            else:
                l.append(i)
        key = tuple(l)
        return key

    def _sparse_setitem(self, key, values):
        check = (
            (self.obj.values.coords[0] == key[0]) *
            (self.obj.values.coords[1] == key[1]) *
            (self.obj.values.coords[2] == key[2]) *
            (self.obj.values.coords[3] == key[3]))
        if check.max():
            data_index = np.where(check == True)[0][0]
            self.obj.values.data[data_index] = values
        else:
            self.obj.values.coords = np.concatenate(
                (self.obj.values.coords, np.array(key)[:, None]), 1)
            self.obj.values.data = np.concatenate(
                (self.obj.values.data, np.array([values])), 0)
            self.obj.values = self.obj.get_array_module()(
                self.obj.values.coords, self.obj.values.data, prune=True,
                has_duplicates=False, shape=self.obj.shape,
                fill_value=self.obj.values.fill_value)


class Location(_LocBase):
    """ class to generate .loc[] functionality """

    def __getitem__(self, key):
        obj = self.get_idx(self.key_to_slice(key))
        if len(obj) > 1 and obj.index.iloc[:, 0].nunique() == 1:
            obj.set_index(obj.index.iloc[:, 1:], inplace=True)
        return obj

    def format_key(self, key):
        """ Converts keys to loc slices """
        if (type(key) is tuple and len(key) > 1
            and len(self.obj.key_labels) > 1 and type(key[1]) is str
            and key[1] in self.obj.index[self.obj.key_labels[1]].unique()):
            key = (key,)
        else:
            key = (key,) if type(key) is not tuple else key
        key_mask = tuple([i if i is Ellipsis else 0 for i in key])
        if len(key_mask) < len(self.obj.shape) and Ellipsis not in key_mask:
            key_mask = tuple(list(key_mask) + [Ellipsis])
        key_mask = list(self._normalize_index(key_mask))
        key = [item for item in key if item is not Ellipsis]
        for i in range(len(self.obj.shape)):
            if key_mask[i] == 0:
                key_mask[i] = key[0]
                key.pop(0)
        return key_mask

    def index_key(self, key):
        if type(key) == pd.Series and len(key) != len(self.obj):
                key = key.to_frame().T
        if type(key) == pd.Series:
            idx = np.where(key)[0]
        elif type(key) == pd.DataFrame:
            idx = (self.obj.index.reset_index().set_index(self.obj.key_labels)
                       .loc[key.set_index(list(key.columns)).index]).values.flatten()
        elif type(key) in [slice, list, tuple]:
            idx = (self.obj.index.reset_index()
                       .set_index(self.obj.key_labels).loc[key]).values.flatten()
        else:
            idx = np.where(self.obj.kdims[:, 0]==key)[0]
        return idx

    def other_key(self, key, idx):
        if type(key) is np.ndarray:
            return np.where(key)[0]
        if key == slice(None, None, None):
            return key
        s = getattr(self.obj, idx)
        if type(key) in [slice, list]:
            return pd.Series(range(len(s)), index=s).loc[key].values
        if not hasattr(key, '__iter__') or type(key) is str:
            return np.array([pd.Series(range(len(s)), index=s).loc[key]])
        else:
            raise AttributeError("Unable to slice.")


    def key_to_slice(self, key):
        """ Converts keys to integer slices """
        key = self.format_key(key)
        out = [self.index_key(key[0]),
                self.other_key(key[1], 'columns'),
                self.other_key(key[2], 'origin'),
                self.other_key(key[3], 'development')]
        return out

    def __setitem__(self, key, values):
        super().__setitem__(self.key_to_slice(key), values)

class Ilocation(_LocBase):
    """
    Class to generate .iloc[] functionality.
    """

    def __getitem__(self, key):
        return self.get_idx(self._normalize_index(key))

    def __setitem__(self, key, values):
        super().__setitem__(self._normalize_index(key), values)


class TriangleSlicer:
    def __getitem__(self, key):
        """ Boolean Slicer functionality """
        if type(key) is pd.Series and key.name == "development":
            return self._slice(key, "ddims")
        if type(key) is np.ndarray:
            if len(key) == np.prod(self.shape[-2:]) and self.shape[-1] > 1:
                return self._slice_valuation(key)
            return self._slice(key, "odims")
        if type(key) is pd.Series:
            return self.iloc[self.index[key].index]
        elif key in self.key_labels:
            return self.index[key]
        else:
            if type(key) is str and self.virtual_columns.columns.get(key, None):
                out = self.virtual_columns[key].copy()
                out.virtual_columns.columns = {}
                return out
            key = [key] if not hasattr(key, '__iter__') or type(key) is str else key
            idx = [list(self.vdims).index(item) for item in key]
            return self.iloc[:, idx]

    def __setitem__(self, key, value):
        """ Function for pandas style column setting """
        xp = self.get_array_module()
        if callable(value):
            self.virtual_columns[key] = value
            if self.array_backend == "sparse":
                if key not in self.vdims:
                    k, v, o, d = self.values.shape
                    self.values.shape = k, v + 1, o, d
                    self.vdims = np.append(self.vdims, key)
                return
            value = (self.iloc[:, 0].copy() * xp.nan).set_backend(self.array_backend)
        else:
            self.virtual_columns.pop(key)
        if isinstance(value, TriangleSlicer) and value.array_backend != self.array_backend:
            value = value.set_backend(self.array_backend)
        if key in self.vdims:
            i = np.where(self.vdims == key)[0][0]
            if self.array_backend == "sparse":
                before = self.drop(key).values
                bc = before.coords[1, :]
                before.coords[1] = np.where(bc >= i, bc + 1, bc,)
                value.values.coords[1] = i
                coords = np.concatenate((before.coords, value.values.coords), axis=1)
                data = np.concatenate((before.data, value.values.data))
                self.values = xp(
                    coords, data, shape=self.shape, prune=True, fill_value=xp.nan
                )
            else:
                if isinstance(value, TriangleSlicer):
                    value = value.values
                self.values[:, i : i + 1] = value
        else:
            self.vdims = self.vdims if key in self.vdims else np.append(self.vdims, key)
            try:
                self.values = xp.concatenate((self.values, value.values), axis=1)
            except:
                # For misaligned triangle support
                conc = (self.values, (self.iloc[:, 0] * 0 + value).values)
                self.values = xp.concatenate(conc, axis=1)

    def _slice_valuation(self, key):
        """ private method for handling of valuation slicing """
        obj = self.copy()
        obj.valuation_date = min(obj.valuation[key].max(), obj.valuation_date)
        key = key.reshape(self.shape[-2:], order="f")
        obj.values = num_to_nan(obj.values * obj.get_array_module().array(key))
        return _LocBase(obj).get_idx(
            (slice(None), slice(None),
             np.arange(obj.shape[2])[np.sum(~key, 1) != obj.shape[3]],
             np.arange(obj.shape[3])[np.sum(~key, 0) != obj.shape[2]]))

    def _slice(self, key, axis):
        """ private method for handling of origin/development slicing """
        obj = self.copy()
        setattr(obj, axis, getattr(obj, axis)[key])
        slicer = ..., _LocBase._contig_slice(np.arange(len(key))[key])
        if axis == "odims":
            slicer = tuple(list(slicer) + [slice(None)])
        obj.values = obj.values[slicer]
        return obj

    def _set_slicers(self) -> None:
        """ Call any time the shape of index or column changes """
        self.iloc, self.loc = Ilocation(self), Location(self)
        self.iat, self.at = Iat(self), At(self)
        self.virtual_columns = VirtualColumns(self, self.virtual_columns.columns)


class At(Location):
    def _check_index(self, key):
        idx = self.key_to_slice(key)
        for item in idx:
            if type(item) is slice:
                if item.stop - item.start == 1:
                    next
                else:
                    raise ValueError('Invalid Index in At slicer')
            else:
                if len(item) == 1:
                    next
                else:
                    raise ValueError('Invalid Index in At slicer')
        return idx

    def __getitem__(self, key):
        obj = self.get_idx(self._check_index(key))
        if len(obj) > 1 and obj.index.iloc[:, 0].nunique() == 1:
            idx = obj.index.iloc[:, 1:]
            obj.index = idx
        return obj.values[0, 0, 0, 0]

    def __setitem__(self, key, values):
        key = self._check_index(key)
        if self.obj.array_backend == 'sparse':
            key = (key[0][0], key[1][0], key[2][0], key[3][0])
            self._sparse_setitem(key, values)
        else:
            if isinstance(values, TriangleSlicer):
                values = values.values
            key = tuple(
                [slice(item, item + 1) if type(item) is int else item for item in key]
            )
            self.obj.values.__setitem__(self._normalize_index(key), values)


class Iat(Ilocation):
    def _check_index(self, key):
        idx = self._normalize_index(key)
        types = {type(i) for i in idx}
        if len(types) > 1 or list(types)[0] != int:
            raise ValueError('iAt based indexing can only have integer indexers')
        return idx

    def __getitem__(self, key):
        return self.get_idx(self._check_index(key)).values[0, 0, 0, 0]

    def __setitem__(self, key, values):
        key = self._normalize_index(key)
        if self.obj.array_backend == 'sparse':
            self._sparse_setitem(key, values)
        else:
            super().__setitem__(key, values)


class VirtualColumns:
    def __init__(self, triangle: TriangleSlicer, columns=None):
        self.triangle = triangle
        self.columns = {} if not columns else columns

    def __getitem__(self, value):
        return self.columns[value](self.triangle).rename("columns", [value])

    def __setitem__(self, name, value):
        self.columns[name] = value

    def __repr__(self):
        return str(pd.Index(self.columns.keys()))

    def pop(self, key):
        if key in list(self.columns.keys()):
            self.columns.pop(key)
