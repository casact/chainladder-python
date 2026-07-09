"""
Support pandas-style slicing to the Triangle class.
"""
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd

from chainladder.core.typing import (
    _AxisKey,
    _LabelKey,
    TriangleProtocol
)

from chainladder.utils.utility_functions import num_to_nan

from typing import (
    cast,
    overload,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from chainladder import Triangle
    from collections.abc import (
        Callable,
        Sequence
    )
    from chainladder.core.typing import (
        BackendArray,
        IndexExpression
    )
    from sparse import COO
    from types import ModuleType
    from typing import Literal

from sparse import _slicing  # noqa

class _LocBase:
    """
    Base class for pandas style loc/iloc indexing.
    """

    def __init__(self, obj: TriangleProtocol):
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
        cast(np.ndarray, cast(object, self.obj.values)).__setitem__(self._normalize_index(key), values)

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
            values: int | float,
    ) -> None:
        """
        Set slice of Triangle when backend is sparse.

        Parameters
        ----------
        key: tuple[int]
            The location in the backend array to set the value.
        values: int | float
            The value to set.

        Returns
        -------
        None

        """
        # Enforce sparse array type.
        arr: COO = cast("COO", cast(object, self.obj.values))
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

    @staticmethod
    def _to_scalar(values: int | float | TriangleSlicer) -> int | float:
        """
        Unwrap a single-element TriangleSlicer into its underlying scalar value.

        Parameters
        ----------
        values: int | float | TriangleSlicer
            The value to unwrap. Non-TriangleSlicer values are returned unchanged.

        Returns
        -------
        int | float
            The scalar value.

        """
        if isinstance(values, TriangleSlicer):
            backend_values = cast("BackendArray", cast(object, values.values))
            return cast("int | float", backend_values[0, 0, 0, 0])
        return values


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
    """
    Mixin class to provide square bracket [] slicing functionality to the Triangle class.
    """

    @overload
    def __getitem__(self: TriangleProtocol, key: pd.Series | np.ndarray | list[str]) -> Triangle: ...
    @overload
    def __getitem__(self: TriangleProtocol, key: str | int) -> Triangle | pd.Series: ...
    def __getitem__(self: TriangleProtocol, key: pd.Series | np.ndarray | str | list[str] | int) -> Triangle | pd.Series:
        """
        Boolean Slicer functionality.

        Parameters
        ----------
        key: pd.Series | np.ndarray | str | list[str] | int
            The key that you want to slice on.
        Returns
        -------
        Triangle | pd.Series
            The requested slice of the Triangle, or in the case of an index, a series of index labels.
        """

        # Determine the axis to which the key applies, and slice accordingly.

        # Case development axis.
        if isinstance(key, pd.Series) and key.name == "development":
            return self._slice(key, "ddims")
        # Case ndarray, could be valuation or origin.
        if isinstance(key, np.ndarray):
            # Case valuation, inferred by size of ndarray obtained by filtering on valuation date.
            if len(key) == np.prod(self.shape[-2:]) and self.shape[-1] > 1:
                return self._slice_valuation(key)
            # Otherwise, assume case origin.
            return self._slice(key, "odims")
        # Case index.
        if isinstance(key, pd.Series):
            return self.iloc[self.index[key].index]
        elif key in self.key_labels:
            return self.index[key]
        # Case columns.
        else:
            # Access virtual columns created by lazy evaluation.
            if isinstance(key, str) and self.virtual_columns.columns.get(key, None):
                out: Triangle = self.virtual_columns[key].copy()
                out.virtual_columns.columns = {}
                return out
            keys: Sequence[str | int] = [key] if isinstance(key, (str, int, float, np.generic)) else key
            # Identify the position of each requested element within the valuation dimension.
            idx = [list(self.vdims).index(item) for item in keys]
            return self.iloc[:, idx]

    def __setitem__(
            self: TriangleProtocol,
            key: str | int,
            value: int | float | TriangleSlicer | Callable[[Triangle], TriangleSlicer]
    ) -> None:
        """
        Function for pandas-style column setting, i.e., Triangle[...] = value.

        Parameters
        ----------
        key: str | int
            The vdims label of the column to set.
        value: int | float | TriangleSlicer | Callable[[Triangle], TriangleSlicer]
            The value(s) to assign to the column. A callable defines a virtual
            (lazily-computed) column.

        Returns
        -------
        None
        """
        xp: ModuleType = self.get_array_module()
        # Case callable, create lazy-eval virtual columns, but do not compute.
        if callable(value):
            self.virtual_columns[key] = value
            if self.array_backend == "sparse":
                if key not in self.vdims:
                    k, v, o, d = self.values.shape
                    self.values.shape = k, v + 1, o, d
                    self.vdims = np.append(self.vdims, key)
                return
            # Create a placeholder column.
            value = (self.iloc[:, 0].copy() * xp.nan).set_backend(self.array_backend)
        # Otherwise, remove column from virtual columns if previously reserved for lazy evaluation.
        else:
            self.virtual_columns.pop(key)
        if isinstance(value, TriangleSlicer):
            value = cast("TriangleProtocol", cast(object, value))
            if value.array_backend != self.array_backend:
                value = value.set_backend(self.array_backend)
        # Key exists in columns, replace data.
        if key in self.vdims:
            i = np.where(self.vdims == key)[0][0]
            # Case sparse backend.
            if self.array_backend == "sparse":
                # Cast value to sparse backend.
                value = cast("Triangle", value)
                after = cast("COO", value.values)

                # Drop existing data where key matches, reassign coordinates.
                before = self.drop(key).values
                before = cast("COO", before)
                bc = before.coords[1, :]
                before.coords[1] = np.where(bc >= i, bc + 1, bc,)

                # Append assigned data and new coordinates.
                after.coords[1] = i
                coords = np.concatenate((before.coords, after.coords), axis=1)
                data = np.concatenate((before.data, after.data))

                # Create new sparse matrix with updated coords and data, assign to backend array.
                self.values = xp.COO(
                    coords=coords,
                    data=data,
                    shape=self.shape,
                    prune=True,
                    fill_value=xp.COO.nan
                )
            # Case numpy backend.
            else:
                if isinstance(value, TriangleSlicer):
                    value = value.values
                cast(np.ndarray, self.values)[:, i : i + 1] = value
        # Key is new, create a column and update data.
        else:
            self.vdims = np.append(self.vdims, key)
            if isinstance(value, (int, float, np.number)):
                # Broadcast scalar across the Triangle's shape.
                value = self.iloc[:, 0] * 0 + value
            try:
                self.values = xp.concatenate((self.values, value.values), axis=1)
            except (ValueError, AttributeError):
                # For misaligned triangle support.
                conc = (self.values, (self.iloc[:, 0] * 0 + cast("Triangle", value)).values)
                self.values = xp.concatenate(conc, axis=1)

    def _slice_valuation(self: TriangleProtocol, key: np.ndarray) -> Triangle:
        """
        Private method for handling of valuation slicing.

        Parameters
        ----------
        key: np.ndarray
            An array the size of the number of cells in origin x development, typically obtained by
            a comparison against the Triangle's valuation date.

        Returns
        -------
        Triangle
        """
        obj = self.copy()
        # Update the triangle's valuation date by computing the max valuation on the remaining cells,
        # limited by the current valuation date in case the valuation periods extend beyond it, i.e.,
        # in the case of tails.
        obj.valuation_date = min(obj.valuation[key].max(), obj.valuation_date)
        # Filter out values by converting them to nan.
        key = key.reshape(self.shape[-2:], order="F")
        obj.values = cast("BackendArray", num_to_nan(obj.values * obj.get_array_module().array(key)))
        # Recalculate size of the origin and development axes and return the slice.
        return _LocBase(obj).get_idx((
            slice(None),
            slice(None),
            np.arange(obj.shape[2])[np.sum(~key, 1) != obj.shape[3]],
            np.arange(obj.shape[3])[np.sum(~key, 0) != obj.shape[2]]
         ))

    def _slice(self: TriangleProtocol, key: pd.Series | np.ndarray, axis: Literal['ddims', 'odims']) -> Triangle:
        """
        Private method for handling of origin/development slicing.

        Parameters
        ----------
        key: pd.Series | np.ndarray
            Array specifying the slice to extract.
        axis: Literal['ddims', 'odims']
            The axis to which the slice would apply, `ddims` for development, `odims` for origin.
        """

        # Update the axis, then the values.
        obj = self.copy()
        setattr(obj, axis, getattr(obj, axis)[key])
        # noinspection PyProtectedMember
        slicer = ..., _LocBase._contig_slice(np.arange(len(key))[key])
        if axis == "odims":
            slicer = tuple(list(slicer) + [slice(None)])
        obj.values = obj.values[slicer]
        return obj

    def _set_slicers(self: TriangleProtocol) -> None:
        """
        Set the indexers on the Triangle during initialization. Enables indexing functionality such as Triangle.iloc[],
        Triangle.loc[], Triangle.iat[], and Triangle.at[].

        Also, call at any time the shape of index or column changes.

        Returns
        -------
        None
        """
        self.iloc, self.loc = Ilocation(self), Location(self)
        self.iat, self.at = Iat(self), At(self)
        self.virtual_columns = VirtualColumns(cast("Triangle", self), self.virtual_columns.columns)


class At(Location):
    """
    Single-element accessor. Mirrors pandas.DataFrame.at[].
    """
    def _check_index(self, key: _LabelKey) -> tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]:
        """
        Makes sure that the requested key explicitly specifies all 4 axes
        (index, columns, origin, development) and that it will grab a single
        element.

        Parameters
        ----------
        key: _LabelKey

        Returns
        -------
        tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey]
            If the key passes validation, it's returned as a tuple of  integer slices, otherwise an error is raised.

        """
        err_msg: str = 'Invalid Index in At slicer.'
        # Unlike loc/iloc, at requires every axis to be specified explicitly.
        if not isinstance(key, tuple) or len(key) != 4 or Ellipsis in key:
            raise ValueError(err_msg)
        # Convert to integer slices.
        idx = cast(tuple[_AxisKey, _AxisKey, _AxisKey, _AxisKey], self.key_to_slice(key))
        for n, item in enumerate(idx):
            if type(item) is slice:
                # A slice here is always the full axis (slice(None, None, None)),
                # which only selects a single element if the axis itself has size 1.
                if self.obj.shape[n] != 1:
                    raise ValueError(err_msg)
            else:
                # `item` is always an np.ndarray here, since `index_key` and `other_key`
                # only ever return a slice or an np.ndarray.
                if len(cast(np.ndarray, item)) != 1:
                    raise ValueError(err_msg)
        return idx

    def __getitem__(self, key: _LabelKey) -> float:
        """
        Extracts a single element specified by `key`.

        Parameters
        ----------
        key: _LabelKey
            A key to specify the location of the element to be extracted. Must result in a single element.

        Returns
        -------
        float
            A single data element from the Triangle.

        """
        obj: Triangle = self.get_idx(self._check_index(key))
        return obj.values[0, 0, 0, 0]

    def __setitem__(self, key: _LabelKey, values: int | float | TriangleSlicer) -> None:
        """
        Sets a single-element of a Triangle to a scalar value.

        Parameters
        ----------
        key: _LabelKey
            The key specifying the element location.
        values: int | float | TriangleSlicer
            The scalar value to set the element to.

        Returns
        -------
        None
        """
        key = self._check_index(key)
        values = self._to_scalar(values)
        if self.obj.array_backend == 'sparse':
            key = tuple(0 if type(item) is slice else int(cast(np.ndarray, item)[0]) for item in key)
            self._sparse_setitem(key, values)
        else:
            cast(np.ndarray, cast(object, self.obj.values)).__setitem__(self._normalize_index(key), values)


class Iat(Ilocation):
    """
    Single-element integer-based accessor. Mirrors pandas.DataFrame.iat[].
    """
    def _check_index(self, key: IndexExpression) -> tuple[int, int, int, int]:
        """
        Make sure the requested key accesses a single element.

        Parameters
        ----------
        key: IndexExpression
            The integer-based index expression specifying the element location for lookup.

        Returns
        -------
        tuple[int]

        """
        idx = self._normalize_index(key)
        types = {type(i) for i in idx}
        if len(types) > 1 or list(types)[0] != int:
            raise ValueError('iAt based indexing can only have integer indexers')
        return cast("tuple[int, int, int, int]", idx)

    def __getitem__(self, key: IndexExpression) -> float:
        """
        Get a single-element's value given the key.

        Parameters
        ----------
        key: IndexExpression
            The integer-based index expression specifying the element location for lookup.

        Returns
        -------
        float
            The value extracted from the specified location.

        """
        return self.get_idx(self._check_index(key)).values[0, 0, 0, 0]

    def __setitem__(self, key: IndexExpression, values: int | float | TriangleSlicer) -> None:
        """
        Sets a single-element of a Triangle to a scalar value.

        Parameters
        ----------
        key: IndexExpression
            The integer-based index expression specifying the element location to set the value for.
        values: int | float | TriangleSlice
            The value to set.

        Returns
        -------

        """
        idx = self._check_index(key)
        if self.obj.array_backend == 'sparse':
            self._sparse_setitem(idx, self._to_scalar(values))
        else:
            super().__setitem__(idx, values)


class VirtualColumns:
    """
    A virtual column is a non-computed column that enables lazy evaluation. For example, a column created
    by assigning a lambda expression.
    """
    def __init__(self, triangle: Triangle, columns=None):
        self.triangle = triangle
        self.columns = {} if not columns else columns

    def __getitem__(self, value: str | int) -> Triangle:
        """
        Look up the callable associated with the virtual column, and then execute it.

        Parameters
        ----------
        value: str | int
            The name of the virtual column you want to get the value for.

        Returns
        -------
        Triangle

        """
        return self.columns[value](self.triangle).rename("columns", [value])

    def __setitem__(self, name: str | int, value: Callable[[Triangle], TriangleSlicer]) -> None:
        """
        Set a new Callable for the requested virtual column.

        Parameters
        ----------
        name: str | int
            The name of the virtual column.
        value: Callable[[Triangle], TriangleSlicer]
            The callable to set.

        Returns
        -------
        None

        """
        self.columns[name] = value

    def __repr__(self) -> str:
        """
        Virtual column string representation.

        Returns
        -------
        A string representing the names of the virtual columns.
        """
        return str(pd.Index(self.columns.keys()))

    def pop(self, key: str | int) -> None:
        """
        Removes a virtual column.

        Parameters
        ----------
        key: str | int
            The name of the column to remove.

        Returns
        -------
        None

        """
        if key in list(self.columns.keys()):
            self.columns.pop(key)
