# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
import copy


class _LocBase:
    ''' Base class for pandas style loc/iloc indexing '''
    def __init__(self, obj):
        self.obj = obj

    def get_idx(self, idx):
        ''' Returns a slice of the original Triangle '''
        obj = copy.deepcopy(self.obj)
        vdims = pd.Series(obj.vdims)
        obj.kdims = np.array(idx.index)
        obj.vdims = np.array(idx.columns)
        obj.key_labels = list(idx.index.names)
        obj.iloc, obj.loc = Ilocation(obj), Location(obj)
        x_0 = self._contig_slice(list(pd.Series([item[0] for item in idx.values[:, 0]]).unique()))
        x_1 = self._contig_slice(list(pd.Series([item[1] for item in idx.values[0, :]]).unique()))
        if type(x_0) is slice or type(x_1) is slice:
            obj.values = obj.values[x_0, x_1, ...]
        else:
            obj.values = obj.values[x_0, ...][:, x_1, ...]
        obj.values[obj.values == 0] = np.nan
        return obj

    def _contig_slice(self, arr):
        if len(arr) == 1:
            return slice(arr[0], arr[0]+1)
        diff = np.diff(arr)
        step  = None if arr[0] < arr[-1] else -1
        if diff.max() == diff.min() and diff.max() in [1, -1]:
            # index is sorted, so use its sort for better performance
            if not step:
                return slice(min(arr), max(arr) + 1, step)
            else:
                min_arr = None if min(arr) == 0 else min(arr)
                return slice(max(arr), min_arr, step)
        return arr


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
        idx = self.obj._idx_table().loc[key]
        obj = self.get_idx(self.obj._idx_table_format(idx))
        return obj


class Ilocation(_LocBase):
    ''' class to generate .iloc[] functionality '''
    def __getitem__(self, key):
        idx = self.obj._idx_table().iloc[key]
        obj = self.get_idx(self.obj._idx_table_format(idx))
        return obj


class TriangleSlicer:
    ''' Slicer functionality '''
    def _idx_table_format(self, idx):
        if type(idx) is pd.Series:
            # One row or one column selection is it k or v?
            if len(set(idx.index).intersection(set(self.vdims))) == len(idx):
                # One column selection
                idx = idx.to_frame().T
                idx.index.names = self.key_labels
            else:
                # One row selection
                idx = idx.to_frame()
        elif type(idx) is tuple:
            # Single cell selection
            idx = self._idx_table().iloc[idx[0]:idx[0] + 1,
                                         idx[1]:idx[1] + 1]
        return idx

    def _idx_table(self):
        ''' private method that generates a dataframe of triangle indices.
            The dataframe is meant to be sliced using pandas and the resultant
            indices are then to be extracted from the Triangle object.
        '''
        df = pd.DataFrame(list(self.kdims), columns=self.key_labels)
        for num, item in enumerate(self.vdims):
            df[item] = list(zip(np.arange(len(df)),
                            (np.ones(len(df))*num).astype(int)))
        df.set_index(self.key_labels, inplace=True)
        return df

    def __getitem__(self, key):
        ''' Function for pandas style column indexing'''

        if type(key) is pd.DataFrame and 'development' in key.columns:
            return self._slice_development(key['development'])
        if type(key) is pd.Index:
            key = key.to_list()
        if type(key) is np.ndarray:
            # Presumes that if I have a 1D array, I will want to slice origin.
            if len(key) == np.prod(self.shape[-2:]) and self.shape[-1] > 1:
                return self._slice_valuation(key)
            return self._slice_origin(key)
        if type(key) is pd.Series:
            return self.iloc[list(self.index[key].index)]
        elif key in self.key_labels:
            return self.index[key]
        else:
            idx = self._idx_table_format(self._idx_table()[key])
            obj = _LocBase(self).get_idx(idx)
            return obj

    def __setitem__(self, key, value):
        ''' Function for pandas style column indexing setting '''
        idx = self._idx_table()
        idx[key] = 1
        if key in self.vdims:
            i = np.where(self.vdims == key)[0][0]
            self.values[:, i:i+1] = value.values
        else:
            self.vdims = np.array(idx.columns.unique())
            xp = cp.get_array_module(self.values)
            try:
                self.values = xp.concatenate((self.values, value.values), axis=1)
            except:
                # For misaligned triangle support
                self.values = xp.concatenate(
                    (self.values,
                    (self.iloc[:, 0]*0+value).values), axis=1)


    def _slice_origin(self, key):
        ''' private method for handling of origin slicing '''
        obj = copy.deepcopy(self)
        obj.odims = obj.odims[key]
        obj.values = obj.values[..., key, :]
        return self._cleanup_slice(obj)

    def _slice_valuation(self, key):
        ''' private method for handling of valuation slicing '''
        obj = copy.deepcopy(self)
        obj.valuation_date = min(
            obj.valuation[key].max(), obj.valuation_date)
        key = key.reshape(self.shape[-2:], order='f')
        nan_tri = np.ones(self.shape[-2:])
        nan_tri = key*nan_tri
        nan_tri[nan_tri == 0] = np.nan
        o, d = nan_tri.shape
        o_idx = np.arange(o)[list(np.sum(np.isnan(nan_tri), 1) != d)]
        d_idx = np.arange(d)[list(np.sum(np.isnan(nan_tri), 0) != o)]
        obj.odims = obj.odims[np.sum(np.isnan(nan_tri), 1) != d]
        if len(obj.ddims) > 1:
            obj.ddims = obj.ddims[np.sum(np.isnan(nan_tri), 0) != o]
        xp = cp.get_array_module(obj.values)
        if xp == cp:
            nan_tri = cp.array(nan_tri)
        obj.values = (obj.values*nan_tri)
        if np.all(o_idx == np.array(range(o_idx[0], o_idx[-1]+1))):
            o_idx = slice(o_idx[0], o_idx[-1]+1)
        if np.all(d_idx == np.array(range(d_idx[0], d_idx[-1]+1))):
            d_idx = slice(d_idx[0], d_idx[-1]+1)
        if type(o_idx) is slice or type(d_idx) is slice:
            # If contiguous slices, this is faster
            obj.values = obj.values[..., o_idx, d_idx]
        else:
            obj.values = xp.take(xp.take(obj.values, o_idx, -2), d_idx, -1)
        return self._cleanup_slice(obj)

    def _slice_development(self, key):
        ''' private method for handling of development slicing '''
        obj = copy.deepcopy(self)
        obj.ddims = obj.ddims[key]
        if cp.get_array_module(obj.values) == cp:
            key = cp.array(key)
        obj.values = obj.values[..., key]
        return self._cleanup_slice(obj)

    def _cleanup_slice(self, obj):
        ''' private method with common post-slicing functionality'''
        obj.valuation = obj._valuation_triangle()
        return obj

    def _set_slicers(self):
        ''' Call any time the shape of index or column changes '''
        self.iloc, self.loc = Ilocation(self), Location(self)
