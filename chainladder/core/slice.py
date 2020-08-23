# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
import copy


class _LocBase:
    ''' Base class for pandas style loc/iloc indexing '''
    def __init__(self, obj):
        self.obj = obj

    def get_idx(self, idx, key=None):
        ''' Returns a slice of the original Triangle '''
        obj = copy.deepcopy(self.obj)
        i_idx = _LocBase._contig_slice(
            pd.unique([item[0] for item in idx.values[:, 0]]))
        c_idx = _LocBase._contig_slice(
            pd.unique([item[1] for item in idx.values[0, :]]))
        if key:
            o_idx = _LocBase._contig_slice(key[0])
            d_idx = _LocBase._contig_slice(key[1])
            if type(o_idx) != slice or type(d_idx) != slice:
                raise ValueError('Fancy indexing on origin/development is not supported.')
        else:
            o_idx = d_idx = slice(None)
        if type(i_idx) is slice or type(c_idx) is slice:
            obj.values = obj.values[i_idx, c_idx, o_idx, d_idx]
        else:
            obj.values = obj.values[i_idx, :, o_idx, d_idx][:, c_idx, ...]
        obj.kdims = np.array(idx.index)
        obj.vdims = np.array(idx.columns)
        obj.key_labels = list(idx.index.names)
        obj.odims, obj.ddims = obj.odims[o_idx], obj.ddims[d_idx]
        obj.iloc, obj.loc = Ilocation(obj), Location(obj)
        return obj

    @staticmethod
    def _contig_slice(arr):
        if type(arr) is slice:
            return arr
        if type(arr) == int:
            arr = [arr]
        if len(arr) == 1:
            return slice(arr[0], arr[0]+1)
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
    ''' class to generate .loc[] functionality '''
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
        key = (key,) if type(key) not in [list, tuple] else key
        idx = self.obj._idx_table().loc[key[:min(len(key),2)]]
        key = [slice(None), slice(None)] if len(key) <= 2 else list(key[2:])
        key = key + [slice(None)] if len(key) == 1 else key
        key[0] = pd.Series(self.obj.origin).to_frame().reset_index() \
                   .set_index('origin').loc[key[0]]['index']
        key[1] = pd.Series(self.obj.development).to_frame().reset_index() \
                   .set_index('development').loc[key[1]]['index']
        key = [[item] if type(item) is not pd.Series else item.values for item in key]
        obj = self.get_idx(self.obj._idx_table_format(idx), key)
        return obj


class Ilocation(_LocBase):
    ''' class to generate .iloc[] functionality '''
    def __getitem__(self, key):
        key = (key,) if type(key) not in [list, tuple] else key
        idx = self.obj._idx_table().iloc[key[:min(len(key),2)]]
        key = [slice(None), slice(None)] if len(key) < 2 else list(key[2:])
        key = key + [slice(None)] if len(key) == 1 else key
        obj = self.get_idx(self.obj._idx_table_format(idx), key)
        return obj


class TriangleSlicer:
    ''' Slicer functionality '''
    def _idx_table_format(self, idx):
        if type(idx) is pd.Series:
            if len(set(idx.index).intersection(set(self.vdims))) == len(idx):
                # One column selection
                idx = idx.to_frame().T
                idx.index.names = self.key_labels
            else:  # One row selection
                idx = idx.to_frame()
        elif type(idx) is tuple:  # Single cell selection
            idx = self._idx_table().iloc[idx[0]:idx[0] + 1, idx[1]:idx[1] + 1]
        return idx

    def _idx_table(self):
        ''' private method that generates a dataframe of triangle indices.
            The dataframe is meant to be sliced using pandas and the resultant
            indices are then to be extracted from the Triangle object.'''
        df = pd.DataFrame(list(self.kdims), columns=self.key_labels)
        for num, item in enumerate(self.vdims):
            df[item] = list(zip(np.arange(len(df)),
                            (np.ones(len(df))*num).astype(int)))
        df.set_index(self.key_labels, inplace=True)
        return df

    def __getitem__(self, key):
        ''' Function for pandas style column indexing'''
        if type(key) is pd.Series and key.name == 'development':
            return self._slice(key, 'ddims')
        if type(key) is np.ndarray:
            # Presumes that if I have a 1D array, I will want to slice origin.
            if len(key) == np.prod(self.shape[-2:]) and self.shape[-1] > 1:
                return self._slice_valuation(key)
            return self._slice(key, 'odims')
        if type(key) is pd.Series:
            return self.iloc[list(self.index[key].index), :]
        elif key in self.key_labels:
            return self.index[key]
        else:
            idx = self._idx_table_format(self._idx_table()[key])
            obj = _LocBase(self).get_idx(idx)
            return obj

    def __setitem__(self, key, value):
        ''' Function for pandas style column indexing setting '''
        xp = self.get_array_module()
        idx = self._idx_table()
        idx[key] = 1
        if key in self.vdims:
            i = np.where(self.vdims == key)[0][0]
            if self.array_backend == 'sparse':
                before = self.drop(key).values
                before.coords[1] = np.where(
                    before.coords[1, :]>=i, before.coords[1, :] + 1,
                    before.coords[1, :])
                value.values.coords[1] = i
                coords = np.concatenate((before.coords, value.values.coords), axis=1)
                data = np.concatenate((before.data, value.values.data))
                self.values = xp(coords, data, shape=self.shape, prune=True)
            else:
                self.values[:, i:i+1] = value.values
        else:
            self.vdims = np.array(idx.columns.unique())
            try:
                self.values = xp.concatenate((self.values, value.values), axis=1)
            except:
                # For misaligned triangle support
                self.values = xp.concatenate(
                    (self.values,
                    (self.iloc[:, 0]*0+value).values), axis=1)

    def _slice_valuation(self, key):
        ''' private method for handling of valuation slicing '''
        obj = copy.deepcopy(self)
        obj.valuation_date = min(
            obj.valuation[key].max(), obj.valuation_date)
        key = key.reshape(self.shape[-2:], order='f')
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
        #if len(obj.ddims) > 1:
        obj.values = (obj.values * obj.get_array_module().array(nan_tri))
        if type(o_idx) is slice or type(d_idx) is slice:
            obj.values = obj.values[..., o_idx, d_idx]
        else:
            obj.values = obj.values[..., o_idx, :][..., d_idx]
        return obj

    def _slice(self, key, axis):
        ''' private method for handling of origin/development slicing '''
        obj = copy.deepcopy(self)
        setattr(obj, axis, getattr(obj, axis)[key])
        slicer = ..., _LocBase._contig_slice(np.arange(len(key))[key])
        if axis == 'odims':
            slicer = tuple(list(slicer) + [slice(None)])
        obj.values = obj.values[slicer]
        return obj

    def _set_slicers(self):
        ''' Call any time the shape of index or column changes '''
        self.iloc, self.loc = Ilocation(self), Location(self)
        self = self._auto_sparse()
