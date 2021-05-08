# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp


def _get_full_expectation(cdf_, ultimate_):
    """ Private method that builds full expectation"""
    full = ultimate_.copy()
    xp = full.get_array_module()
    full.values = xp.repeat(full.values, cdf_.shape[-1], -1)
    offset = {"Y": 12, "Q": 3, "M": 1}[cdf_.development_grain]
    ddims = ([cdf_.ddims[0] - offset], list(cdf_.ddims[:-1]), [9999])
    full.ddims = np.concatenate(ddims)
    if len(cdf_) != len(ultimate_) and len(cdf_.index) > 1:
        if hasattr(ultimate_, 'group_index'):
            group_index = ultimate_.group_index
        else:
            group_index = ultimate_.index
        level = list(
            set(group_index.columns).intersection(
            set(cdf_.key_labels)))
        idx = group_index.merge(
            cdf_.index.reset_index(),
            how='left', on=level)['index'].values.astype(int)
        cdf = cdf_.values[list(idx), ...]
    else:
        cdf = cdf_.values
    full.values = full.values / cdf
    full.values = xp.concatenate((full.values, ultimate_.set_backend(full.array_backend).values), -1)
    return full


def _get_full_triangle(X, ultimate, expectation=None, n_iters=None):
    """ Private method that builds full triangle"""
    cdf = X.ldf_.copy()
    xp = cdf.get_array_module()
    if cdf.shape[2] == 1:
        cdf.values = xp.repeat(cdf.values, len(X.origin), 2)
    cdf.odims = X.odims
    cdf = cdf[cdf.valuation<=X.valuation_date] * 0 + 1 + cdf[cdf.valuation>X.valuation_date]
    cdf.values = cdf.values.cumprod(3)

    if len(cdf) != len(ultimate) and len(cdf.index) > 1:
        if hasattr(ultimate, 'group_index'):
            group_index = ultimate.group_index
        else:
            group_index = ultimate.index
        level = list(
            set(group_index.columns).intersection(
            set(cdf.key_labels)))
        idx = group_index.merge(
            cdf.index.reset_index(),
            how='left', on=level)['index'].values.astype(int)
        cdf.values = cdf.values[list(idx), ...]
    else:
        cdf.values = cdf.values
    cdf.kdims = ultimate.kdims
    cdf.key_labels = ultimate.key_labels
    cdf = (1 - 1 / cdf)
    ld = X.latest_diagonal
    ld.valuation_date = ld.valuation.max()
    ld = cdf * 0 + ld.values
    if n_iters is not None:
        a = (X.latest_diagonal * 0 + expectation.values) / X.cdf_ * X.ldf_
        complement = xp.nansum(cdf.values[None] ** xp.arange(n_iters)[:, None, None, None, None], 0)
        new_run_off = (( a * (cdf ** n_iters)) + (ld * complement).values)
    else:
        complement = (1 / (1 - cdf))
        new_run_off = (ld * complement)
    new_run_off = new_run_off[new_run_off.valuation>X.valuation_date] + X
    new_run_off.is_pattern = False
    new_run_off.is_cumulative = True
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
