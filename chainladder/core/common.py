# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
from chainladder.utils.dask import dp
import numpy as np
from chainladder.utils.utility_functions import concat
from chainladder import options


def _get_full_expectation(cdf_, ultimate_, is_cumulative=True):
    """Private method that builds full expectation"""
    full = ultimate_ / cdf_

    if is_cumulative:
        return concat((full, ultimate_.copy().rename("development", [9999])), axis=3)

    else:
        tail_ = full.iloc[:, :, :, -1] - ultimate_

        return concat(
            (full.cum_to_incr(), tail_.copy().rename("development", [9999])), axis=3
        )


def _get_full_triangle(X, ultimate, is_cumulative=True):
    """Private method that builds full triangle"""
    # Getting the LDFs and expand for all origins
    from chainladder.utils.utility_functions import num_to_nan

    emergence = (X.ldf_.copy() * (ultimate / ultimate)).fillna(1)

    # Setting LDFs for all of the known diagonals as 1
    emergence = (
        emergence[emergence.valuation < X.valuation_date] * 0
        + 1
        + emergence[emergence.valuation >= X.valuation_date]
    )

    emergence.valuation_date = pd.to_datetime(options.ULT_VAL)
    emergence.values = emergence.values.cumprod(axis=3) - 1

    # Shifting the CDFs by development age, and renaming the last column as 9999
    emergence.ddims = (
        emergence.ddims + {"Y": 12, "Q": 3, "S": 6, "M": 1}[emergence.development_grain]
    )
    emergence.ddims[-1] = 9999
    emergence.values = emergence.values / num_to_nan(emergence.values[..., -1:])
    ld = X.incr_to_cum().latest_diagonal
    cum_run_off = ld + emergence * (ultimate - ld)
    cum_run_off = cum_run_off[cum_run_off.valuation > X.valuation_date]
    cum_run_off.is_cumulative = True

    if is_cumulative:
        return X + cum_run_off
    else:
        return (X.incr_to_cum() + cum_run_off).cum_to_incr()


class Common:
    """Class that contains common properties of a "fitted" Triangle."""

    @property
    def has_ldf(self):
        if hasattr(self, "ldf_"):
            return True
        else:
            return False

    @property
    def has_zeta(self):
        if hasattr(self, "zeta_"):
            return True
        else:
            return False

    @property
    def cdf_(self):
        if not self.has_ldf:
            x = self.__class__.__name__
            raise AttributeError("'" + x + "' object has no attribute 'cdf_'")
        return self.ldf_.incr_to_cum()

    @property
    def cum_zeta_(self):
        if not self.has_zeta:
            x = self.__class__.__name__
            raise AttributeError("'" + x + "' object has no attribute 'cum_zeta_'")
        return self.zeta_.incr_to_cum()

    @property
    def ibnr_(self):
        if not hasattr(self, "ultimate_"):
            x = self.__class__.__name__
            raise AttributeError("'" + x + "' object has no attribute 'ibnr_'")
        if hasattr(self, "X_"):
            ld = self.latest_diagonal
        else:
            ld = self.latest_diagonal if self.is_cumulative else self.sum(axis=3)
        ibnr = self.ultimate_ - ld
        ibnr.vdims = self.ultimate_.vdims
        return ibnr

    @property
    def full_expectation_(self):
        if not hasattr(self, "ultimate_"):
            raise AttributeError(
                "'"
                + self.__class__.__name__
                + "' object has no attribute 'full_expectation_'"
            )

        return _get_full_expectation(self.cdf_, self.ultimate_, self.X_.is_cumulative)

    @property
    def full_triangle_(self):
        if not hasattr(self, "ultimate_"):
            raise AttributeError(
                "'"
                + self.__class__.__name__
                + "' object has no attribute 'full_triangle_'"
            )

        if hasattr(self, "X_"):
            X = self.X_
        else:
            X = self
        return _get_full_triangle(X, self.ultimate_, X.is_cumulative)

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def set_backend(
        self, backend: str, inplace: bool = False, deep: bool = False, **kwargs
    ):
        """
        Converts triangle array_backend.

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
            old_backend: str = self.array_backend
        else:
            if hasattr(self, "ldf_"):
                old_backend: str = self.ldf_.array_backend
            else:
                raise ValueError("Unable to determine array backend.")
        if inplace:
            # Coming from dask - compute and then recall this method
            # going to dask  -
            if old_backend == "dask" and backend != "dask":
                self = self.compute()
                old_backend: str = self.array_backend
            if backend in ["numpy", "sparse", "cupy", "dask"]:
                lookup: dict = {
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
            return obj.set_backend(backend=backend, inplace=True, deep=deep, **kwargs)

    def _validate_assumption(self, triangle, value, axis):
        if type(value) in (int, float, str):
            arr = np.repeat(value, triangle.shape[axis])
        if type(value) in (list, tuple, set, np.ndarray):
            arr = np.array(value)
        if type(value) is dict:
            arr = np.array([value[a] for a in triangle._get_axis_value(axis)])
        if callable(value):
            arr = np.array([value(a) for a in triangle._get_axis_value(axis)])
        if axis == 3:
            arr = arr[None, None, None]
        if axis == 2:
            arr = arr[None, None, :, None]
        return arr
