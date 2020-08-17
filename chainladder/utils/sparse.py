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
    if hasattr(a, 'fill_value'):
        a = a.copy()
        a.data[np.isnan(a.data)]=0.0
        if a.fill_value != 0.0:
            a.fill_value = 0.0
    return sp(a)
sp.nan_to_num = nan_to_num

def ones(*args, **kwargs):
    return sp(np.ones(*args, **kwargs), fill_value=sp.nan)
sp.ones = ones

def nanmedian(a, axis=None, keepdims=None, *args, **kwargs):
    a.fill_value = np.nan
    new_a = np.nanmedian(a.todense(), axis=axis, keepdims=keepdims, *args, **kwargs)
    return sp(np.nan_to_num(new_a))
sp.nanmedian = nanmedian

def nanmean(a, axis=None, keepdims=None, *args, **kwargs):
    n = sp.nansum(a, axis=axis, keepdims=keepdims)
    d = sp.nansum(sp.nan_to_num(a)!=0, axis=axis, keepdims=keepdims).astype(n.dtype)
    n.fill_value=np.nan
    d.fill_value=np.nan
    n = sp(n)
    d = sp(d)
    out = n / d
    out.fill_value = 0
    return sp(out)
sp.nanmean = nanmean

def flip(m, axis=None):
    m = m.copy()
    if axis is not None:
        m.coords[axis] = m.shape[axis] - m.coords[axis] - 1
    else:
        for i in range(m.coords.shape[0]):
            m.coords[i] = m.shape[i] - m.coords[i] - 1
    return m
sp.flip = flip


def array(a, *args, **kwargs):
    if kwargs.get('fill_value', None) is not None:
        fill_value = kwargs.pop('fill_value')
    else:
        fill_value = sp.nan
    if type(a)==sp:
        return sp(a, *args, **kwargs, fill_value=fill_value)
    else:
        return sp(np.array(a, *args, **kwargs), fill_value=fill_value)
sp.array = array

def expand_dims(a, axis):
    a = a.copy()
    shape = [slice(None, None)] * a.ndim
    if axis == -1:
        shape.append(None)
    elif axis < -1:
        axis = axis + 1
        shape.insert(axis, None)
    else:
        shape.insert(axis, None)
    return a.__getitem__(tuple(shape))
sp.expand_dims = expand_dims

def arange(*args, **kwargs):
    return sparse.COO.from_numpy(np.arange(*args, **kwargs))
sp.arange = arange

def cumfunc(a, axis, func):
    a = copy.deepcopy(a)
    ax = np.arange(a.ndim)
    axis = ax[axis]
    x = pd.DataFrame(a.coords.T)
    x.columns = ['0', '1', '2', '3']
    cols = [item for item in x.columns if item != str(axis)]
    x['y']=a.data
    x = pd.pivot_table(
        x , columns=cols,
        index=str(axis), values='y')
    missing = pd.Int64Index(
        set(np.arange(a.shape[axis])) - set(x.index), name=str(axis))
    if len(missing) > 0:
        x = x.append(pd.DataFrame(
            np.repeat((x.iloc[0:1]*0).values, len(missing), axis=0),
            index=missing, columns=x.columns))
    x = x.unstack().reset_index().fillna(0)
    x.columns = [item for item in x.columns[:-1]] + ['y']
    x = x.set_index(list(x.columns[:-1])).groupby(level=[0,1,2])
    if func == 'cumsum':
        x = x.cumsum().reset_index()
    if func == 'cumprod':
        x = x.cumprod().reset_index()
    x = x[x['y']!=0]
    a.coords = x[['0', '1', '2', '3']].values.T
    a.data = x['y'].values
    return sp(a)

def cumsum(a, axis):
    return cumfunc(a, axis, 'cumsum')
sp.cumsum = cumsum

def cumprod(a, axis):
    return cumfunc(a, axis, 'cumprod')
sp.cumprod = cumprod

def where(*args, **kwargs):
    return elemwise(np.where, *args, **kwargs)
sp.where = where

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
    coords = a.coords[l,:]
    return sp(coords, a.data, shape=tuple([a.shape[item] for item in l]),
              prune=True, fill_value=fv)
sp.swapaxes = swapaxes

def repeat(a, repeats, axis):
    """Repeat elements of an array"""
    ax = np.arange(a.ndim)
    axis = ax[axis]
    r = []
    for item in range(repeats):
        coords = a.coords.copy()
        coords[axis] = coords[axis]+item
        r.append(coords)
    v = np.tile(a.data, repeats)
    a.coords = np.concatenate(r, axis=1)
    a.data = v
    a.shape = tuple(
        [item if num!=axis else item*repeats
         for num, item in enumerate(a.shape)])
    return a
sp.repeat = repeat

def allclose(a, b, *args, **kwargs):
    return np.allclose(np.nan_to_num(a.todense()), np.nan_to_num(b.todense()), *args, **kwargs)
sp.allclose = allclose


def unique(a, *args, **kwargs):
    fv = a.fill_value
    a = np.unique(a.todense(), *args, **kwargs)
    return sp(a, fill_value = fv)
sp.unique = unique

def around(a, *args, **kwargs):
    fv = a.fill_value
    a = np.around(a.todense(), *args, **kwargs)
    return sp(a, fill_value = fv)
sp.around = around

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    fv = arr.fill_value
    arr = np.apply_along_axis(func1d, axis, arr.todense(), *args, **kwargs)
    return sp(arr, fill_value = fv)
sp.apply_along_axis = apply_along_axis

def floor(x, *args, **kwargs):
    x.data = np.floor(x.data)
    return x
sp.floor = floor

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
        raise ValueError('Shapes are not equal')
    return ((x1<x2)*x1+(x1>=x2)*x2)
sp.minimum = minimum

def maximum(x1, x2):
    if np.all(x1.coords == x2.coords):
        out = x1.copy()
        out.data = np.maximum(x1.data, x2.data)
        return out
    if x1.shape != x2.shape:
        raise ValueError('Shapes are not equal')
    return ((x1<x2)*x2+(x1>=x2)*x1)
sp.maximum = maximum
