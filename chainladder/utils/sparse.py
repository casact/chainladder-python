# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import sparse as sp
from sparse import COO as COO
from sparse import elemwise

def _setitem_not_supported(self, key, value) -> None: # noqa
    raise TypeError(
        """
        In-place item assignment (e.g. `triangle.values[...] = value`) is not 
        supported on the sparse backend. Use numpy backend for in-place assignment.
        """
    )


sp.isnan = np.isnan
COO.nan = np.array([1.0, np.nan])[-1]
COO.__setitem__ = _setitem_not_supported
setattr(sp, 'testing', np.testing)
sp.sqrt = np.sqrt
sp.log = np.log
sp.exp = np.exp
sp.abs = np.abs


def nan_to_num(a, nan = 0.0):
    if type(a) in [int, float, np.int64, np.float64]:
        return np.nan_to_num(a)
    if hasattr(a, "fill_value"):
        a = a.copy()
        a.data[np.isnan(a.data)] = nan
    return COO(coords=a.coords, data=a.data, fill_value = nan, shape = a.shape)


def ones(*args, **kwargs):
    return COO(np.ones(*args, **kwargs), fill_value=COO.nan)


def nansum(a, axis=None, keepdims=None, *args, **kwargs):
    return COO(data=a.data, coords=a.coords, fill_value=0.0, shape=a.shape).sum(
        axis=axis, keepdims=keepdims, *args, **kwargs
    )

def nanquantile(a: COO, q: float, axis: int = 0, keepdims: bool = False):
    """
    mimics np.nanquantile

    Parameters
    ----------
    x : COO
        input sparse array.
    q : float
        quantiles in [0, 1].
    axis : int
        Axis over which the quantile is applied.
    keepdims : bool
        Match NumPy keepdims behavior.

    Returns
    -------
    out : COO
        result cast back to COO for consistency.
    """

    # coordinates that survive the reduction
    keep_axes = tuple(i for i in range(a.ndim) if i != axis)
    keep_shape = tuple(a.shape[i] for i in keep_axes)

    if not keep_axes:
        out = np.nanquantile(a.data, q)
        if keepdims:
            out = np.asarray(out).reshape(
                tuple(1 for _ in range(a.ndim))
            )
        return COO(out)

    # map every stored value to an output location
    keep_coords = a.coords[list(keep_axes)]
    group_ids = np.ravel_multi_index(keep_coords, keep_shape)
    
    # sort by group
    order = np.argsort(group_ids)
    group_ids = group_ids[order]
    values = a.data[order]

    out = np.full(np.prod(keep_shape), np.nan)

    starts = np.r_[0, np.flatnonzero(np.diff(group_ids)) + 1]
    ends = np.r_[starts[1:], len(group_ids)]

    for start, end in zip(starts, ends):
        gid = group_ids[start]
        out[gid] = np.nanquantile(values[start:end], q)

    out = out.reshape(keep_shape)

    if keepdims:
        out = np.expand_dims(out,axis)

    return COO(out)

def nanmedian(a: COO, axis: int = 0, keepdims: bool = False):
    """
    mimics np.nanmean

    Parameters
    ----------
    x : COO
        input sparse array.
    axis : int
        Axis over which the median is applied.
    keepdims : bool
        Match NumPy keepdims behavior.

    Returns
    -------
    out : COO
        result cast back to COO for consistency.
    """
    return nanquantile(a, 0.5, axis, keepdims)

def nanmean(a, axis=None, keepdims=None):
    n = nansum(a, axis=axis, keepdims=keepdims)
    d = nansum(nan_to_num(a) != 0, axis=axis, keepdims=keepdims).astype(n.dtype)
    n = COO(data=n.data, coords=n.coords, fill_value=np.nan, shape=n.shape)
    d = COO(data=d.data, coords=d.coords, fill_value=np.nan, shape=d.shape)
    out = n / d
    return COO(data=out.data, coords=out.coords, fill_value=0, shape=out.shape)

def array(a, *args, **kwargs):
    if kwargs.get("fill_value", None) is not None:
        fill_value = kwargs.pop("fill_value")
    else:
        fill_value = COO.nan
    if type(a) == sp.COO:
        return COO(a, *args, **kwargs, fill_value=fill_value)
    else:
        return COO(np.array(a, *args, **kwargs), fill_value=fill_value)


def arange(*args, **kwargs):
    return COO.from_numpy(np.arange(*args, **kwargs))


def where(*args, **kwargs):
    return elemwise(np.where, *args, **kwargs)


def cumprod(a, axis=None, dtype=None, out=None):
    return array(np.cumprod(a.todense(), axis=axis, dtype=dtype, out=out))


def floor(x: COO) -> COO:
    x = x.copy()
    x.data = np.floor(x.data)
    return x


sp.nansum = nansum
sp.minimum = np.minimum
sp.maximum = np.maximum
sp.floor = floor
sp.where = where
sp.arange = arange
sp.array = array
sp.nan_to_num = nan_to_num
sp.ones = ones
sp.cumprod = cumprod
COO.cumprod = cumprod
sp.nanmean = nanmean
sp.sum = COO.sum
sp.nanquantile = nanquantile
sp.nanmedian = nanmedian