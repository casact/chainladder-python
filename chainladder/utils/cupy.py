# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from chainladder import ARRAY_BACKEND


try:
    import cupy as cp
    cp.array([1])
    module = 'cupy'
except:
    if ARRAY_BACKEND == 'cupy':
        import warnings
        warnings.warn('Unable to load CuPY.  Using numpy instead.')
    import numpy as cp
    module = 'numpy'

def get_array_module(*args, **kwargs):
    """ default array module when cupy is not present """
    return np

def nansum(a, *args, **kwargs):
    """ For cupy v0.6.0 compatibility """
    return cp.sum(cp.nan_to_num(a), *args, **kwargs)

def nanmean(a, *args, **kwargs):
    """ For cupy v0.6.0 compatibility """
    return cp.sum(cp.nan_to_num(a), *args, **kwargs) / \
           cp.sum(~cp.isnan(a), *args, **kwargs)

def nanmedian(a, *args, **kwargs):
    """ For cupy v0.6.0 compatibility """
    return cp.array(np.nanmedian(cp.asnumpy(a), *args, **kwargs))

def nanpercentile(a, *args, **kwargs):
    """ For cupy v0.6.0 compatibility """
    return cp.array(np.nanpercentile(cp.asnumpy(a), *args, **kwargs))

def unique(ar, axis=None, *args, **kwargs):
    """ For cupy v0.6.0 compatibility """
    return cp.array(np.unique(cp.asnumpy(ar), axis=axis, *args, **kwargs))

if module == 'cupy' and int(cp.__version__.split('.')[0]) < 7:
    cp.nansum = nansum
    cp.nanmean = nanmean
    cp.nanmedian = nanmedian
    cp.nanpercentile = nanpercentile
    cp.unique = unique

if module == 'numpy':
    cp.get_array_module = get_array_module
