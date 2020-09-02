# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np


class _LocBase:
    """ Base class for pandas style loc/iloc indexing """

    def __init__(self, obj):
        self.obj = obj

    def get_idx(self, idx, filter_idx=None):
        """ Returns a slice of the original Triangle """
        obj = self.obj.copy()
        if filter_idx is not None:
            obj.index = filter_idx
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
        return obj

    @staticmethod
    def _contig_slice(arr):
        if type(arr) is slice:
            return arr
        if type(arr) in [int, np.int64, np.int32]:
            arr = [arr]
        if len(arr) == 1:
            return slice(arr[0], arr[0] + 1)
        diff = np.diff(arr)
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


class Location(_LocBase):
    """ class to generate .loc[] functionality """

    def __getitem__(self, key):
        if type(key) == pd.Series:
            return self.obj[key]
        if type(key) == tuple and type(key[0]) == pd.Series:
            return self.obj[key[0]][key[1]]
        if type(key) == pd.DataFrame:
            if len(self.obj.key_labels) == 1:
                key = pd.Index(key.iloc[:, 0])
            else:
                key = pd.Index(key)
        key = (key,) if type(key) is not tuple else key
        key = list(key) + [slice(None)] * (4 - len(key))
        idx = self.obj.index.reset_index().set_index(self.obj.key_labels)
        sliced = idx.loc[key[0]]
        if type(sliced) is pd.Series:
            sliced = sliced.to_frame().T
            sliced.index.rename(idx.index.name, inplace=True)
        sliced = sliced.iloc[:, 0]
        key[0] = sliced.to_list()
        filter_idx = idx.reset_index()[
            list(sliced.reset_index().drop("index", 1).columns)
        ]
        s = pd.Series(self.obj.columns).to_frame().reset_index()
        key[1] = s.set_index(0).loc[key[1]].values.flatten()
        s = pd.Series(self.obj.origin).to_frame().reset_index()
        key[2] = s.set_index("origin").loc[key[2]].values.flatten()
        s = pd.Series(self.obj.development).to_frame().reset_index()
        key[3] = s.set_index("development").loc[key[3]].values.flatten()
        return self.get_idx(key, filter_idx)


class Ilocation(_LocBase):
    """ class to generate .iloc[] functionality """

    def __getitem__(self, key):
        key = (key,) if type(key) is not tuple else key
        key = list(key) + [slice(None)] * (4 - len(key))
        return self.get_idx(key)


class TriangleSlicer:
    """ Slicer functionality """

    def __getitem__(self, key):
        """ Function for pandas style column indexing"""
        if type(key) is pd.Series and key.name == "development":
            return self._slice(key, "ddims")
        if type(key) is np.ndarray:
            # Presumes that if I have a 1D array, I will want to slice origin.
            if len(key) == np.prod(self.shape[-2:]) and self.shape[-1] > 1:
                return self._slice_valuation(key)
            return self._slice(key, "odims")
        if type(key) is pd.Series:
            return self.iloc[list(self.index[key].index), :]
        elif key in self.key_labels:
            return self.index[key]
        else:
            key = [key] if type(key) is str else key
            idx = [list(self.vdims).index(item) for item in key]
            return self.iloc[:, idx]

    def __setitem__(self, key, value):
        """ Function for pandas style column indexing setting """
        xp = self.get_array_module()
        if key in self.vdims:
            i = np.where(self.vdims == key)[0][0]
            if self.array_backend == "sparse":
                before = self.drop(key).values
                before.coords[1] = np.where(
                    before.coords[1, :] >= i,
                    before.coords[1, :] + 1,
                    before.coords[1, :],
                )
                value.values.coords[1] = i
                coords = np.concatenate((before.coords, value.values.coords), axis=1)
                data = np.concatenate((before.data, value.values.data))
                self.values = xp(coords, data, shape=self.shape, prune=True)
            else:
                self.values[:, i : i + 1] = value.values
        else:
            self.vdims = self.vdims if key in self.vdims else np.append(self.vdims, key)
            try:
                self.values = xp.concatenate((self.values, value.values), axis=1)
            except:
                # For misaligned triangle support
                self.values = xp.concatenate(
                    (self.values, (self.iloc[:, 0] * 0 + value).values), axis=1
                )

    def _slice_valuation(self, key):
        """ private method for handling of valuation slicing """
        obj = self.copy()
        obj.valuation_date = min(obj.valuation[key].max(), obj.valuation_date)
        key = key.reshape(self.shape[-2:], order="f")
        nan_tri = np.ones(self.shape[-2:])
        nan_tri = key * nan_tri
        nan_tri[nan_tri == 0] = np.nan
        o, d = nan_tri.shape
        o_idx = np.arange(o)[(np.sum(np.isnan(nan_tri), 1) != d)]
        d_idx = np.arange(d)[(np.sum(np.isnan(nan_tri), 0) != o)]
        o_idx = _LocBase._contig_slice(o_idx)
        d_idx = _LocBase._contig_slice(d_idx)
        obj.odims = obj.odims[np.sum(np.isnan(nan_tri), 1) != d]
        obj.ddims = obj.ddims[np.sum(np.isnan(nan_tri), 0) != o]
        obj.values = obj.values * obj.get_array_module().array(nan_tri)
        if type(o_idx) is slice or type(d_idx) is slice:
            obj.values = obj.values[..., o_idx, d_idx]
        else:
            obj.values = obj.values[..., o_idx, :][..., d_idx]
        return obj

    def _slice(self, key, axis):
        """ private method for handling of origin/development slicing """
        obj = self.copy()
        setattr(obj, axis, getattr(obj, axis)[key])
        slicer = ..., _LocBase._contig_slice(np.arange(len(key))[key])
        if axis == "odims":
            slicer = tuple(list(slicer) + [slice(None)])
        obj.values = obj.values[slicer]
        return obj

    def _set_slicers(self):
        """ Call any time the shape of index or column changes """
        self.iloc, self.loc = Ilocation(self), Location(self)
        self = self._auto_sparse()
