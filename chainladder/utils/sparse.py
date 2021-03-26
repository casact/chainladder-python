# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import sparse
from chainladder import ARRAY_BACKEND
from sparse import COO as sp
from sparse import elemwise
import pandas as pd
import copy

sp.isnan = np.isnan
sp.newaxis = np.newaxis
sp.nan = np.array([1.0, np.nan])[-1]
sp.testing = np.testing
sp.nansum = sparse.nansum
sp.nanmin = sparse.nanmin
sp.nanmax = sparse.nanmax
sp.concatenate = sparse.concatenate
sp.diagonal = sparse.diagonal
sp.zeros = sparse.zeros
sp.testing.assert_array_equal = np.testing.assert_equal
sp.sqrt = np.sqrt
sp.log = np.log
sp.exp = np.exp
sp.abs = np.abs


def nan_to_num(a):
    if type(a) in [int, float, np.int64, np.float64]:
        return np.nan_to_num(a)
    if hasattr(a, "fill_value"):
        a = a.copy()
        a.data[np.isnan(a.data)] = 0.0
    return sp(coords=a.coords, data=a.data, fill_value=0.0, shape=a.shape)


def ones(*args, **kwargs):
    return sp(np.ones(*args, **kwargs), fill_value=sp.nan)


def nanquantile(a, q, axis=None, keepdims=None, *args, **kwargs):
    new_a = np.nanquantile(
        a.todense(), q=q, axis=axis, keepdims=keepdims, *args, **kwargs
    )
    return sp(np.nan_to_num(new_a), fill_value=np.nan)


def nanmedian(a, axis=None, keepdims=None, *args, **kwargs):
    new_a = np.nanmedian(a.todense(), axis=axis, keepdims=keepdims, *args, **kwargs)
    return sp(np.nan_to_num(new_a), fill_value=np.nan)


def nanmean(a, axis=None, keepdims=None, *args, **kwargs):
    n = sp.nansum(a, axis=axis, keepdims=keepdims)
    d = sp.nansum(sp.nan_to_num(a) != 0, axis=axis, keepdims=keepdims).astype(n.dtype)
    n = sp(data=n.data, coords=n.coords, fill_value=np.nan, shape=n.shape)
    d = sp(data=d.data, coords=d.coords, fill_value=np.nan, shape=d.shape)
    out = n / d
    return sp(data=out.data, coords=out.coords, fill_value=0, shape=out.shape)


def array(a, *args, **kwargs):
    if kwargs.get("fill_value", None) is not None:
        fill_value = kwargs.pop("fill_value")
    else:
        fill_value = sp.nan
    if type(a) == sp:
        return sp(a, *args, **kwargs, fill_value=fill_value)
    else:
        return sp(np.array(a, *args, **kwargs), fill_value=fill_value)


def arange(*args, **kwargs):
    return sparse.COO.from_numpy(np.arange(*args, **kwargs))


def where(*args, **kwargs):
    return elemwise(np.where, *args, **kwargs)


def swapaxes(a, axis1, axis2):
    ax = np.arange(a.ndim)
    axis1 = ax[axis1]
    axis2 = ax[axis2]
    fv = a.fill_value
    l = []
    for item in range(a.ndim):
        if item == axis1:
            l.append(axis2)
        elif item == axis2:
            l.append(axis1)
        else:
            l.append(item)
    coords = a.coords[l, :]
    return sp(
        coords,
        a.data,
        shape=tuple([a.shape[item] for item in l]),
        prune=True,
        fill_value=fv,
    )


def repeat(a, repeats, axis):
    """Repeat elements of an array"""
    ax = np.arange(a.ndim)
    axis = ax[axis]
    r = []
    for item in range(repeats):
        coords = a.coords.copy()
        coords[axis] = coords[axis] + item
        r.append(coords)
    v = np.tile(a.data, repeats)
    a.coords = np.concatenate(r, axis=1)
    a.data = v
    a.shape = tuple(
        [item if num != axis else item * repeats for num, item in enumerate(a.shape)]
    )
    return a


def allclose(a, b, *args, **kwargs):
    return np.allclose(
        np.nan_to_num(a.todense()), np.nan_to_num(b.todense()), *args, **kwargs
    )


def unique(a, *args, **kwargs):
    fv = a.fill_value
    a = np.unique(a.todense(), *args, **kwargs)
    return sp(a, fill_value=fv)


def around(a, *args, **kwargs):
    fv = a.fill_value
    a = np.around(a.todense(), *args, **kwargs)
    return sp(a, fill_value=fv)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    fv = arr.fill_value
    arr = np.apply_along_axis(func1d, axis, arr.todense(), *args, **kwargs)
    return sp(arr, fill_value=fv)


def floor(x, *args, **kwargs):
    x.data = np.floor(x.data)
    return x


def minimum(x1, x2):
    if type(x2) in [int, float]:
        out = x1.copy()
        out.data = np.minimum(x1.data, x2)
        return out
    elif np.all(x1.coords == x2.coords):
        out = x1.copy()
        out.data = np.minimum(x1.data, x2.data)
        return out
    if x1.shape != x2.shape:
        raise ValueError("Shapes are not equal")
    return (x1 < x2) * x1 + (x1 >= x2) * x2


def maximum(x1, x2):
    if np.all(x1.coords == x2.coords):
        out = x1.copy()
        out.data = np.maximum(x1.data, x2.data)
        return out
    if x1.shape != x2.shape:
        raise ValueError("Shapes are not equal")
    return (x1 < x2) * x2 + (x1 >= x2) * x1


sp.minimum = minimum
sp.maximum = maximum
sp.unique = unique
sp.around = around
sp.apply_along_axis = apply_along_axis
sp.floor = floor
sp.repeat = repeat
sp.where = where
sp.arange = arange
sp.array = array
sp.nan_to_num = nan_to_num
sp.ones = ones
sp.nanquantile = nanquantile
sp.nanmedian = nanmedian
sp.nanmean = nanmean
sp.swapaxes = swapaxes
sp.allclose = allclose
