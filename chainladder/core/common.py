# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
from chainladder.utils.dask import dp


def _get_full_expectation(cdf_, ultimate_):
    """ Private method that builds full expectation"""
    from chainladder.utils.utility_functions import concat
    full = ultimate_ / cdf_
    return concat((full, ultimate_.copy().rename('development', [9999])), axis=3)


def _get_full_triangle(X, ultimate, expectation=None, n_iters=None):
    """ Private method that builds full triangle"""
    from chainladder import ULT_VAL
    cdf = X.ldf_.copy()
    xp = cdf.get_array_module()
    cdf = cdf * (ultimate / ultimate)
    cdf = cdf[cdf.valuation<X.valuation_date] * 0 + 1 + cdf[cdf.valuation>=X.valuation_date]
    cdf.values = cdf.values.cumprod(3)
    cdf.valuation_date = pd.to_datetime(ULT_VAL)
    cdf = (1 - 1 / cdf)
    cdf.ddims = cdf.ddims + {'Y': 12, 'Q': 3, 'M':1}[cdf.development_grain]
    cdf.ddims[-1] = 9999
    ld = X.latest_diagonal
    if n_iters is not None:
        cdf_ = X.cdf_
        cdf_.ddims = cdf.ddims
        a = (X.latest_diagonal * 0 + expectation) / cdf_ * X.ldf_.values
        complement = xp.nansum(cdf.values[None] ** xp.arange(n_iters)[:, None, None, None, None], 0)
        new_run_off = ((a * (cdf ** n_iters)) + (ld * complement))
    else:
        complement = (1 / (1 - cdf))
        new_run_off = (ld * complement)
    new_run_off = new_run_off[new_run_off.valuation>X.valuation_date]
    return new_run_off + X


class Common:
    """ Class that contains common properties of a "fitted" Triangle. """

    @property
    def has_ldf(self):
        if hasattr(self, "ldf_"):
            return True
        else:
            return False

    @property
    def cdf_(self):
        if not hasattr(self, "ldf_"):
            x = self.__class__.__name__
            raise AttributeError("'" + x + "' object has no attribute 'cdf_'")
        return self.ldf_.incr_to_cum()

    @property
    def ibnr_(self):
        if not hasattr(self, "ultimate_"):
            x = self.__class__.__name__
            raise AttributeError("'" + x + "' object has no attribute 'ibnr_'")
        ibnr = self.ultimate_ - self.latest_diagonal
        ibnr.vdims = self.ultimate_.vdims
        return ibnr

    @property
    def full_expectation_(self):
        if not hasattr(self, "ultimate_"):
            x = self.__class__.__name__
            raise AttributeError(
                "'" + x + "' object has no attribute 'full_expectation_'"
            )
        return _get_full_expectation(self.cdf_, self.ultimate_)

    @property
    def full_triangle_(self):
        if not hasattr(self, "ultimate_"):
            raise AttributeError(
                "'"
                + self.__class__.__name__
                + "'"
                + " object has no attribute 'full_triangle_'"
            )
        if hasattr(self, "X_"):
            X = self.X_
        else:
            X = self
        if hasattr(self, 'n_iters'):
            return _get_full_triangle(X, self.ultimate_, self.expectation_, self.n_iters)
        else:
            return _get_full_triangle(X, self.ultimate_)

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def set_backend(self, backend, inplace=False, deep=False, **kwargs):
        """ Converts triangle array_backend.

        Parameters
        ----------
        backend : str
            Currently supported options are 'numpy', 'sparse', and 'cupy'
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Triangle with updated array_backend
        """
        if hasattr(self, "array_backend"):
            old_backend = self.array_backend
        else:
            if hasattr(self, "ldf_"):
                old_backend = self.ldf_.array_backend
            else:
                raise ValueError("Unable to determine array backend.")
        if inplace:
            # Coming from dask - compute and then recall this method
            # going to dask  -
            if old_backend == 'dask' and backend != 'dask':
                self = self.compute()
                old_backend = self.array_backend
            if backend in ["numpy", "sparse", "cupy", "dask"]:
                lookup = {
                    "numpy": {
                        "sparse": lambda x: x.todense(),
                        "cupy": lambda x: cp.asnumpy(x),
                    },
                    "cupy": {
                        "numpy": lambda x: cp.array(x),
                        "sparse": lambda x: cp.array(x.todense()),
                    },
                    "sparse": {
                        "numpy": lambda x: sp.array(x),
                        "cupy": lambda x: sp.array(cp.asnumpy(x)),
                    },
                    "dask": {
                        # should this be chunked?
                        "numpy": lambda x: dp.from_array(x, **kwargs),
                        "cupy": lambda x: dp.from_array(x, **kwargs),
                        "sparse": lambda x: dp.from_array(x, **kwargs),
                    }
                }
                if hasattr(self, "values"):
                    self.values = lookup[backend].get(old_backend, lambda x: x)(
                        self.values
                    )
                if deep:
                    for k, v in vars(self).items():
                        if isinstance(v, Common):
                            v.set_backend(backend, inplace=True, deep=True)
                if hasattr(self, "array_backend"):
                    self.array_backend = backend
            else:
                raise AttributeError(backend, "backend is not supported.")
            return self
        else:
            obj = self.copy()
            return obj.set_backend(backend=backend, inplace=True, deep=deep, **kwargs)
