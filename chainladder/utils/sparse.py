# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import sparse
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


def nansum(a, axis=None, keepdims=None, *args, **kwargs):
    return sp(data=a.data, coords=a.coords, fill_value=0.0, shape=a.shape).sum(
        axis=axis, keepdims=keepdims, *args, **kwargs
    )
sp.nansum = nansum


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


def cumprod(a, axis=None, dtype=None, out=None):
    return array(np.cumprod(a.todense(), axis=axis, dtype=dtype, out=out))


def floor(x, *args, **kwargs):
    x.data = np.floor(x.data)
    return x



sp.minimum = np.minimum
sp.maximum = np.maximum
sp.floor = floor
sp.where = where
sp.arange = arange
sp.array = array
sp.nan_to_num = nan_to_num
sp.ones = ones
sp.cumprod = cumprod
sp.nanmean = nanmean
