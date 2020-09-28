# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from sparse._slicing import normalize_index


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
        obj.valuation_date = np.minimum(obj.valuation.max(), obj.valuation_date)
        return obj

    @staticmethod
    def _contig_slice(arr):
        if type(arr) is pd.Series:
            arr = arr[arr].index.tolist()
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
        if isinstance(values, TriangleSlicer):
            if values.array_backend == "sparse":
                values = values.set_backend("numpy").values
            else:
                values = values.values
        key = tuple(
            [slice(item, item + 1) if type(item) is int else item for item in key]
        )
        if self.obj.array_backend == "sparse":
            self.obj.set_backend("numpy", inplace=True)
            print(self.obj.values)
            self.obj.values.__setitem__(normalize_index(key, self.obj.shape), values)
            self.obj.set_backend("sparse", inplace=True)
        else:
            self.obj.values.__setitem__(normalize_index(key, self.obj.shape), values)


class Location(_LocBase):
    """ class to generate .loc[] functionality """

    def __getitem__(self, key):
        return self.get_idx(*self.format_key(key))

    def format_key(self, key):
        key = (key,) if type(key) is not tuple else key
        key_mask = tuple([i if i is Ellipsis else 0 for i in key])
        if len(key_mask) < len(self.obj.shape) and Ellipsis not in key_mask:
            key_mask = tuple(list(key_mask) + [Ellipsis])
        key_mask = list(normalize_index(key_mask, self.obj.shape))
        key = [item for item in key if item is not Ellipsis]
        for i in range(len(self.obj.shape)):
            if key_mask[i] == 0:
                key_mask[i] = key[0]
                key.pop(0)
        key = key_mask
        # Support for multi-index
        default = slice(None, None, None)
        norm = lambda k: type(k) is slice and (k.start == 0 or k == default)
        filter_idx = None
        if not norm(key[0]) and not type(key[0]) is pd.Series:
            idx = self.obj.index.reset_index().set_index(self.obj.key_labels)
            sliced = idx.loc[key[0]]
            if type(sliced) is pd.Series:
                sliced = sliced.to_frame().T
                sliced.index.rename(idx.index.name, inplace=True)
            sliced = sliced.iloc[:, 0]
            key[0] = sliced.to_list()
            filter_idx = list(sliced.reset_index().drop("index", 1).columns)
            filter_idx = idx.reset_index()[filter_idx]

        def normalize(key, idx):
            mapper = {1: "columns", 2: "origin", 3: "development"}
            out = key[idx]
            if not norm(key[idx]) and not isinstance(key, pd.Series):
                s = pd.Series(getattr(self.obj, mapper[idx])).to_frame().reset_index()
                out = s.set_index(mapper[idx]).loc[key[idx]].values.flatten()
            return out

        key = [key[0]] + [normalize(key, 1), normalize(key, 2), normalize(key, 3)]
        return key, filter_idx

    def __setitem__(self, key, values):
        key = self.format_key(key)[0]
        super().__setitem__(key, values)


class Ilocation(_LocBase):
    """ class to generate .iloc[] functionality """

    def __getitem__(self, key):
        return self.get_idx(normalize_index(key, self.obj.shape))

    def __setitem__(self, key, values):
        key = normalize_index(key, self.obj.shape)
        super().__setitem__(key, values)


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
            key = [key] if type(key) is str else key
            idx = [list(self.vdims).index(item) for item in key]
            return self.iloc[:, idx]

    def __setitem__(self, key, value):
        """ Function for pandas style column setting """
        xp = self.get_array_module()
        if key in self.vdims:
            i = np.where(self.vdims == key)[0][0]
            if self.array_backend == "sparse":
                before = self.drop(key).values
                bc = before.coords[1, :]
                before.coords[1] = np.where(bc >= i, bc + 1, bc,)
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
                conc = (self.values, (self.iloc[:, 0] * 0 + value).values)
                self.values = xp.concatenate(conc, axis=1)

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
