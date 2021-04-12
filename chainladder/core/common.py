# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp


def _get_full_expectation(cdf_, ultimate_):
    """ Private method that builds full expectation"""
    xp = ultimate_.get_array_module()
    cdf = cdf_.copy()
    if cdf.shape[2] == 1:
        cdf.values = cdf_.get_array_module().repeat(
            cdf.values[..., 0:1, :], ultimate_.shape[-2], 2
        )
        cdf.odims = ultimate_.odims
    cdf.valuation_date = ultimate_.valuation_date
    full = ultimate_.copy()
    full.values = xp.concatenate(((full / cdf).values, full.values), -1)
    offset = {"Y": 12, "Q": 3, "M": 1}[cdf_.development_grain]
    ddims = ([cdf_.ddims[0] - offset], list(cdf_.ddims[:-1]), [9999])
    full.ddims = np.concatenate(ddims)
    return full


def _get_full_triangle(X, ultimate, expectation=None, n_iters=None):
    cdf = X.ldf_.copy()
    xp = cdf.get_array_module()
    if cdf.shape[2] == 1:
        cdf.values = xp.repeat(cdf.values, len(X.origin), 2)
    cdf.odims = X.odims
    cdf = cdf[cdf.valuation<=X.valuation_date] * 0 + 1 + cdf[cdf.valuation>X.valuation_date]
    cdf.values = cdf.values.cumprod(3)
    cdf = (1 - 1 / cdf)
    ld = X.copy()
    ld.valuation_date = ld.valuation.max()
    ld = cdf * 0 + X.latest_diagonal.set_backend(cdf.array_backend).values
    if n_iters is not None:
        a = (X.latest_diagonal * 0 + expectation.values) / X.cdf_ * X.ldf_
        complement = xp.nansum(cdf.values[None] ** xp.arange(n_iters)[:, None, None, None, None], 0)
        new_run_off = (( a * (cdf ** n_iters)) + (ld * complement).values)
    else:
        complement = (1 / (1 - cdf))
        new_run_off = (ld * complement)
    new_run_off = new_run_off[new_run_off.valuation>X.valuation_date] + X
    new_run_off.is_pattern = False
    return new_run_off


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

    def set_backend(self, backend, inplace=False, deep=False):
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
            if backend in ["numpy", "sparse", "cupy"]:
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
            return obj.set_backend(backend=backend, inplace=True, deep=deep)
