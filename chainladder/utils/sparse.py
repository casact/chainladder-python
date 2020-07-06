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
sp.nan = 0
sp.testing = np.testing
sp.nansum = sparse.nansum
sp.concatenate = sparse.concatenate
sp.diagonal = sparse.diagonal
sp.zeros = sparse.zeros
sp.testing.assert_array_equal = np.testing.assert_equal

def nan_to_num(a):
    return a
sp.nan_to_num = nan_to_num

def ones(*args, **kwargs):
    return sp(np.ones(*args, **kwargs))
sp.ones = ones

def array(*args, **kwargs):
    return sp(np.array(*args, **kwargs))
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

def cumsum(a, axis):
    """ specific to 3 axis array """
    a = copy.deepcopy(a)
    x = pd.DataFrame(a.coords.T)
    x.columns = ['0', '1', '2', '3']
    x['y']=a.data
    x = pd.pivot_table(x , index=['0', '1', '2'], columns=str(axis), values='y').T.unstack().reset_index().fillna(0)
    x.columns = [0,1,2,3,'y']
    x = x.set_index([0,1,2,3]).groupby(level=axis).cumsum().reset_index()
    x = x[x['y']>0]
    a.coords = x[[0,1,2,3]].values.T
    a.data = x['y'].values
    return a
sp.cumsum = cumsum

def where(*args, **kwargs):
    return elemwise(np.where, *args, **kwargs)
sp.where = where

def swapaxes(a, axis1, axis2):
    l = []
    for item in range(a.ndim):
        if item == axis1:
            l.append(axis2)
        elif item == axis2:
            l.append(axis1)
        else:
            l.append(item)
    print(l)
    a.coords = a.coords[l,:]
    return a

sp.swapaxes = swapaxes
