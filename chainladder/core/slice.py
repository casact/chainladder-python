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

    def get_idx(self, idx):
        ''' Returns a slice of the original Triangle '''
        obj = copy.deepcopy(self.obj)
        vdims = pd.Series(obj.vdims)
        obj.kdims = np.array(idx.index.unique())
        # Honor order of column labels
        obj.vdims = np.array(vdims[vdims.isin(idx.columns.unique())])
        obj.key_labels = list(idx.index.names)
        obj.iloc, obj.loc = Ilocation(obj), Location(obj)
        idx_slice = np.array(idx).flatten()
        x = tuple([np.unique(np.array(item))
                   for item in list(zip(*idx_slice))])
        obj.values = obj.values[x[0]][:, x[1]]
        obj.values[obj.values == 0] = np.nan
        return obj


class Location(_LocBase):
    ''' class to generate .loc[] functionality '''
    def __getitem__(self, key):
        idx = self.obj._idx_table().loc[key]
        return self.get_idx(self.obj._idx_table_format(idx))


class Ilocation(_LocBase):
    ''' class to generate .iloc[] functionality '''
    def __getitem__(self, key):
        idx = self.obj._idx_table().iloc[key]
        return self.get_idx(self.obj._idx_table_format(idx))


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
            The dataframe is meant ot be sliced using pandas and the resultant
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
        elif type(key) is np.ndarray:
            # Presumes that if I have a 1D array, I will want to slice origin.
            if len(key) == np.prod(self.shape[-2:]) and self.shape[-1] > 1:
                return self._slice_valuation(key)
            return self._slice_origin(key)
        elif type(key) is pd.Series:
            return self.iloc[list(self.index[key].index)]
        elif key in self.key_labels:
            # Boolean-indexing of a particular key
            return self.index[key]
        else:
            idx = self._idx_table()[key]
            idx = self._idx_table_format(idx)
            obj = _LocBase(self).get_idx(idx)
            if type(key) is not str and key != list(obj.vdims):
                # Honor order of the slice
                obj2 = obj[key[0]]
                for item in key[1:]:
                    obj2[item] = obj[item]
                return obj2
            else:
                return _LocBase(self).get_idx(idx)

    def __setitem__(self, key, value):
        ''' Function for pandas style column indexing setting '''
        idx = self._idx_table()
        idx[key] = 1
        if key in self.vdims:
            i = np.where(self.vdims == key)[0][0]
            self.values[:, i:i+1] = value.values
        else:
            self.vdims = np.array(idx.columns.unique())
            try:
                self.values = np.append(self.values, value.values, axis=1)
            except:
                # For misaligned triangle support
                self.values = np.append(
                    self.values,
                    (self.iloc[:, 0]*0+value).values, axis=1)


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
            obj.valuation[key].max().to_timestamp(how='e'), obj.valuation_date)
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
        obj.values = (obj.values*nan_tri)
        obj.values = np.take(np.take(obj.values, o_idx, -2), d_idx, -1)
        return self._cleanup_slice(obj)

    def _slice_development(self, key):
        ''' private method for handling of development slicing '''
        obj = copy.deepcopy(self)
        obj.ddims = obj.ddims[key]
        obj.values = obj.values[..., key]
        return self._cleanup_slice(obj)

    def _cleanup_slice(self, obj):
        ''' private method with common post-slicing functionality'''
        obj.valuation = obj._valuation_triangle()
        if hasattr(obj, '_nan_triangle_'):
            # Force update on _nan_triangle at next access.
            del obj._nan_triangle_
            obj._nan_triangle_ = obj._nan_triangle()
        return obj

    def _set_slicers(self):
        ''' Call any time the shape of index or column changes '''
        self.iloc, self.loc = Ilocation(self), Location(self)
