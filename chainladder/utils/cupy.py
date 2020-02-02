# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import numpy as np
try:
    import cupy as cp
except:
    cp = np

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
