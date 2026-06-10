"""
Support pandas-style slicing to the Triangle class.
"""
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import importlib
import numpy as np
import pandas as pd

from chainladder.core.typing import (
    _AxisKey,
    _LabelKey
)

from chainladder.utils.utility_functions import num_to_nan

from typing import (
    cast,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from chainladder import Triangle
    from chainladder.core.typing import IndexExpression
    from sparse import COO
    from typing import Literal


_slicing = importlib.import_module("sparse._slicing")

class _LocBase:
    """
    Base class for pandas style loc/iloc indexing.
    """

    def __init__(self, obj):
        self.obj = obj

    def get_idx(self, idx: tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]) -> Triangle:
        """
        Returns a slice of the original Triangle

        Parameters
        ----------
        idx: tuple
            The index to slice on.

        Returns
        -------
        The requested slice of the original Triangle.

        """
        obj: Triangle = self.obj.copy()
        idx = cast(tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey], idx)
        i_idx: slice | np.ndarray = _LocBase._contig_slice(idx[0])
        c_idx: slice | np.ndarray = _LocBase._contig_slice(idx[1])
        o_idx: slice | np.ndarray = _LocBase._contig_slice(idx[2])
        d_idx: slice | np.ndarray = _LocBase._contig_slice(idx[3])
        if type(o_idx) != slice or type(d_idx) != slice:
            raise ValueError("Fancy indexing on origin/development is not supported.")
        if type(i_idx) is slice or type(c_idx) is slice:
            obj.values = obj.values[i_idx, c_idx, o_idx, d_idx]
        # Case when index and column indexing expressions are arrays.
        else:
            obj.values = obj.values[i_idx, :, o_idx, d_idx][:, c_idx, ...]
        # Set the new dimension values.
        obj.kdims = obj.kdims[i_idx]
        obj.vdims = obj.vdims[c_idx]
        obj.odims, obj.ddims = obj.odims[o_idx], obj.ddims[d_idx]
        # Set indexers.
        obj.iloc, obj.loc = Ilocation(obj), Location(obj)
        obj.valuation_date = cast(pd.Timestamp, np.minimum(obj.valuation.max(), obj.valuation_date))
        return obj

    @staticmethod
    def _contig_slice(arr: _AxisKey) -> slice | np.ndarray:
        """
        Try to make a contiguous slicer from an _AxisKey.

        Parameters
        ----------
        arr: _AxisKey
            The _AxisKey to be transformed into a contiguous slicer.

        Returns
        -------
        slice | np.ndarray
            A contiguous slice, if it can be constructed, otherwise, the original arr.
        """

        # If arr is already a slice, return it.
        if isinstance(arr, slice):
            return arr
        # If arr is int, construct a contiguous slice and return it.
        elif isinstance(arr, (int, np.int32, np.int64)):
            return slice(arr, arr + 1)
        arr = cast(np.ndarray, arr)
        # For single-element array, add 1 to return a contiguous slice.
        if len(arr) == 1:
            return slice(arr[0], arr[0] + 1)
        diff = np.diff(arr)
        if len(diff) == 0:
            raise ValueError("Slice returns empty Triangle.")
        # Case when each step is the same.
        if max(diff) == min(diff):
            step = max(diff)
        # Return the original array if no conversion is possible.
        else:
            return arr
        # Normalize the boundaries and step.
        step = None if step == 1 else step
        min_arr = None if min(arr) == 0 else min(arr)
        max_arr = max(arr) + 1
        if step and step < 0:
            min_arr, max_arr = max_arr - 1, min_arr - 1 if min_arr else min_arr
        return slice(min_arr, max_arr, step)

    def __setitem__(
            self,
            key: tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey],
            values: int | float | TriangleSlicer
    ) -> None:
        """
        Supports the square bracket syntax [] for setting Triangle values. Only supported for numpy backend.

        Parameters
        ----------
        key: tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]
            Indicates the slice of the Triangle you want to set values for.
        values: int | float | TriangleSlicer
            The value(s) you want to assign to the slice of the Triangle.

        Returns
        -------
        None

        """
        if self.obj.array_backend == "sparse":
            raise ValueError('Setting values with sparse backend requires .at or .iat')
        if isinstance(values, TriangleSlicer):
            values = values.values
        # Create a slice for any key elements that are integers, otherwise preserve the slice or array.
        key = tuple(
            [slice(item, item + 1) if isinstance(item, int) else item for item in key]
        )
        self.obj.values.__setitem__(self._normalize_index(key), values)

    def _normalize_index(self, key: IndexExpression) -> tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]:
        """
        Converts an indexing expression into a standard 4-D format. When the indexing has fewer dimensions than 4,
        slices for the remaining dimensions are added.

        Parameters
        ----------
        key: IndexExpression
            The indexing expression passed to the calling indexer.

        Returns
        -------
        tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]
            A normalized, 4-D indexing expression.

        """
        # Apply sparse normalization, fills out the rest of the dimensions using the shape of the Triangle.
        key: tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey] = _slicing.normalize_index(key, self.obj.shape)
        l = []
        # Preserve start/stop/step boundaries if the user has specified them, otherwise replace them with None.
        # None indicates "go-to-boundary" for the slice.
        for n, i in enumerate(key):
            if isinstance(i, slice):
                start: int | None= i.start if i.start > 0 else None
                stop: int | None = i.stop if i.stop > -1 else None
                stop: int | None = None if stop == self.obj.shape[n] else stop
                step: int | None = None if start is None and stop is None else i.step
                l.append(slice(start, stop, step))
            else:
                l.append(i)
        key = tuple(l)
        return key

    def _sparse_setitem(
            self,
            key: tuple[int, int, int, int],
            values: int | float
    ) -> None:
        """
        Set slice of Triangle when backend is sparse.

        Parameters
        ----------
        key: tuple[int, int, int, int]
        values: int | float

        Returns
        -------
        None

        """
        # Enforce sparse array type.
        arr: COO = cast("COO", cast(object, self.obj.values))
        key: tuple = cast("tuple", cast(object, key))
        # Check if a stored value exists at the coordinate point.
        check = (
            (arr.coords[0] == key[0]) *
            (arr.coords[1] == key[1]) *
            (arr.coords[2] == key[2]) *
            (arr.coords[3] == key[3]))
        # If it does, index the location and assign the value directly.
        if check.max():
            data_index = np.where(check == True)[0][0]
            arr.data[data_index] = values
        # Otherwise, create a new sparse array with the updated coordinates and data.
        else:
            # Append the new coordinate.
            arr.coords = np.concatenate(
                (arr.coords, np.array(key)[:, None]), axis=1)
            # Append the new data element.
            arr.data = np.concatenate(
                (arr.data, np.array([values])), axis=0)
            # Construct the new sparse array and assign to Triangle.
            self.obj.values = self.obj.get_array_module().COO(
                coords=arr.coords,
                data=arr.data,
                prune=True,
                has_duplicates=False,
                shape=self.obj.shape,
                fill_value=arr.fill_value
            )


class Location(_LocBase):
    """ class to generate .loc[] functionality """

    def __getitem__(
            self,
            key: _LabelKey
    ) -> Triangle:
        """
        Support square bracket indexing of Triangle.loc[] to extract data.

        Parameters
        ----------
        key: _LabelKey
            The pandas-style slice that you wish to extract from the Triangle.

        Returns
        -------
        Triangle
            The desired slice of a Triangle, specified by key.

        """
        # Extract the desired slice.
        obj: Triangle = self.get_idx(self.key_to_slice(key))
        # Drop the top-level index if unique, otherwise, return the slice.
        if len(obj) > 1 and obj.index.iloc[:, 0].nunique() == 1:
            obj.set_index(obj.index.iloc[:, 1:], inplace=True)
        return obj

    def format_key(self, key: _LabelKey) -> tuple[_LabelKey, _LabelKey, _LabelKey, _LabelKey]:
        """
        Aligns a user-supplied label-based key to the Triangle's 4 axes, leaving each
        element as a label-based selector for index_key/other_key to resolve later.

        Parameters
        ----------
        key: _LabelKey
            The user-supplied label-based key.

        Returns
        -------
        tuple[_LabelKey, _LabelKey, _LabelKey, _LabelKey]
            A 4-tuple of per-axis label-based selectors.
        """
        # Parse the user-supplied key and determine data type and purpose.
        # Preprocess into a common tuple-format prior to standardizing the dimensions.

        # Case when key is a tuple representing an index row.
        if (isinstance(key, tuple) and len(key) > 1
            and len(self.obj.key_labels) > 1 and type(key[1]) is str
            and key[1] in self.obj.index[self.obj.key_labels[1]].unique()):
            key = (key,)
        # Case when tuple elements represent separate dimensions, keep as-is.
        elif isinstance(key, tuple):
            pass
        # Otherwise, convert to tuple.
        else:
            key = (key,)

        # Create a placeholder normalized key, with 0 for mapping to user-supplied dimensions, and
        # a full slice otherwise.
        key_mask = tuple([i if i is Ellipsis else 0 for i in key])
        if len(key_mask) < len(self.obj.shape) and Ellipsis not in key_mask:
            key_mask = tuple(list(key_mask) + [Ellipsis])
        normalized: tuple = self._normalize_index(key_mask)
        key = [item for item in key if item is not Ellipsis]
        # Populate a new formatted key by replacing the 0s with the user-supplied key elements.
        key_mask = []
        for i in range(len(self.obj.shape)):
            if normalized[i] == 0:
                key_mask.append(key.pop(0))
            else:
                key_mask.append(normalized[i])
        return tuple(key_mask)

    def index_key(self, key: _LabelKey) -> np.ndarray:
        """
        Converts a label-based key into an integer-based one. Intended to be used for the index axis.

        Parameters
        ----------
        key: _LabelKey
            A label-based to be converted into an integer-based key.

        Returns
        -------
        np.ndarray
            An integer-based key.

        """
        # Case when key is a single index row and not a boolean mask, preprocess into a DataFrame of labels.
        if isinstance(key, pd.Series) and len(key) != len(self.obj):
                key = key.to_frame().T
        # Case boolean mask. Extract the positions where True.
        if isinstance(key, pd.Series):
            idx = np.where(key)[0]
        # Case DataFrame of labels, find positions in index.
        elif isinstance(key, pd.DataFrame):
            idx = (self.obj.index.reset_index().set_index(self.obj.key_labels)
                       .loc[key.set_index(list(key.columns)).index]).values.flatten()
        # Case Pandas-style label selectors, extract positions from index.
        elif type(key) in [slice, list, tuple]:
            idx = (self.obj.index.reset_index()
                       .set_index(self.obj.key_labels).loc[key]).values.flatten()
        # Case scalar, locate position in first level of index.
        else:
            idx = np.where(self.obj.kdims[:, 0]==key)[0]
        return idx

    def other_key(
            self,
            key: _LabelKey,
            idx: Literal['columns', 'origin', 'development']
    ) -> np.ndarray | slice:
        """
        Converts a label-based key into an integer-based one. Intended to be used for axes other than the index.

        Parameters
        ----------
        key: _LabelKey
            A label-based to be converted into an integer-based key.
        idx: Literal['columns', 'origin', 'development']
            The axis to which the key applies.

        Returns
        -------
        np.ndarray | slice
            An integer-based key.

        """
        # Case boolean mask, return positions.
        if isinstance(key, np.ndarray):
            return np.where(key)[0]
        # Case full-axis slice, simply return it.
        if isinstance(key, slice) and (key == slice(None, None, None)):
            return cast(slice, key)
        # Otherwise, extract the index and then find the positions.
        s = getattr(self.obj, idx)
        obj_idx = pd.Series(range(len(s)), index=s)
        if type(key) in [slice, list]:
            return obj_idx.loc[key].values
        if not hasattr(key, '__iter__') or type(key) is str:
            return np.array([obj_idx.loc[key]])
        else:
            raise AttributeError("Unable to slice.")


    def key_to_slice(self, key: _LabelKey) -> tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]:
        """
        Converts keys to integer slices.

        Parameters
        ----------
        key:
            The pandas-style slice that you wish to extract from the Triangle.
        Returns
        -------
        tuple
            A 4-tuple of integer-based slices.
        """

        # Preprocess key into a normalized 4-D key.
        key = self.format_key(key)
        # Transform into integer-based slices.
        out = (self.index_key(key[0]),
                self.other_key(key[1], 'columns'),
                self.other_key(key[2], 'origin'),
                self.other_key(key[3], 'development'))
        return out

    def __setitem__(self, key: _LabelKey, values: int | float | TriangleSlicer) -> None:
        super().__setitem__(cast(tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey], self.key_to_slice(key)), values)

class Ilocation(_LocBase):
    """
    Class to generate .iloc[] functionality.
    """

    def __getitem__(self, key: IndexExpression) -> Triangle:
        return self.get_idx(self._normalize_index(key))

    def __setitem__(self, key: IndexExpression, values):
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
            key: list = [key] if not hasattr(key, '__iter__') or type(key) is str else key
            # Identify the position of each requested element within the valuation dimension.
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
                self.values = xp.COO(
                    coords, data, shape=self.shape, prune=True, fill_value=xp.COO.nan
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
    def _check_index(self, key) -> tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]:
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
    def _check_index(self, key) -> tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]:
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
