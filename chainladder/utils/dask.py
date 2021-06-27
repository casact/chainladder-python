# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder import ARRAY_BACKEND

try:
    import dask.array as dp
    dp.array([1])
    module = "dask"
except:
    if ARRAY_BACKEND == "dask":
        import warnings

        warnings.warn("Unable to load Dask.  Using numpy instead.")
    import numpy as dp
    module = "numpy"

dp.nan = np.nan


def expand_dims(a, axis=0):
    l = []
    for i in range(len(a.shape)):
        if i == axis:
            l.append(None)
        l.append(slice(None))
    return a.__getitem__(tuple(l))

if dp != np:
    dp.expand_dims = expand_dims
