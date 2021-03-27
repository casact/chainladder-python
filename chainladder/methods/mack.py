# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
from chainladder.methods import Chainladder


class MackChainladder(Chainladder):
    """ Basic stochastic chainladder method popularized by Thomas Mack

    Parameters
    ----------
    None

    Attributes
    ----------
    X_
        returns **X**
    ultimate_
        The ultimate losses per the method
    ibnr_
        The IBNR per the method
    full_expectation_
        The ultimates back-filled to each development period in **X** replacing
        the known data
    full_triangle_
        The ultimates back-filled to each development period in **X** retaining
        the known data
    summary_
        summary of the model
    full_std_err_
        The full standard error
    total_process_risk_
        The total process error
    total_parameter_risk_
        The total parameter error
    mack_std_err_
        The total prediction error by origin period
    total_mack_std_err_
        The total prediction error across all origin periods
    """

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Data to which the model will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, sample_weight)
        if "sigma_" not in self.X_:
            raise ValueError("Triangle not compatible with MackChainladder")
        # Caching full_triangle_ for fit as it is called a lot
        self.X_._full_triangle_ = self.full_triangle_
        self.parameter_risk_ = self._mack_recursion("parameter_risk_", self.X_)
        self.process_risk_ = self._mack_recursion("process_risk_", self.X_)
        self.total_parameter_risk_ = self._mack_recursion(
            "total_parameter_risk_", self.X_
        )
        del self.X_._full_triangle_
        self.process_variance_ = self._include_process_variance()
        return self

    def predict(self, X, sample_weight=None):
        X_new = super().predict(X, sample_weight)
        X_new.sigma_ = getattr(X_new, "sigma_", self.X_.sigma_)
        X_new.std_err_ = getattr(X_new, "std_err_", self.X_.std_err_)
        X_new.average_ = getattr(X_new, "average_", self.X_.average_)
        X_new.w_ = getattr(X_new, "w_", self.X_.w_)
        X_new._full_triangle_ = X_new.full_triangle_
        X_new.parameter_risk_ = self._mack_recursion("parameter_risk_", X_new)
        X_new.process_risk_ = self._mack_recursion("process_risk_", X_new)
        X_new.total_process_risk_ = (X_new.process_risk_ ** 2).sum(axis="origin").sqrt()
        X_new.total_parameter_risk_ = self._mack_recursion(
            "total_parameter_risk_", X_new
        )
        X_new.full_std_err_ = self._get_full_std_err_(X_new)
        X_new.total_mack_std_err_ = self._get_total_mack_std_err_(X_new)
        X_new.mack_std_err_ = (
            X_new.parameter_risk_ ** 2 + X_new.process_risk_ ** 2
        ).sqrt()
        del X_new._full_triangle_
        return X_new

    @property
    def full_std_err_(self):
        return self._get_full_std_err_(self.X_)

    def _get_full_std_err_(self, X=None):
        from chainladder.utils.utility_functions import num_to_nan

        obj = X.copy()
        xp = obj.get_array_module()
        lxp = X.ldf_.get_array_module()
        full = getattr(X, "_full_triangle_", self.full_triangle_)
        avg = {"regression": 0, "volume": 1, "simple": 2}
        avg = [avg.get(item, item) for item in X.average_]
        val = xp.broadcast_to(xp.array(avg + [avg[-1]]), X.shape)
        weight = xp.sqrt(full.values[..., : len(X.ddims)] ** (2 - val))
        obj.values = X.sigma_.values / num_to_nan(weight)
        w = lxp.concatenate((X.w_, lxp.ones((1, 1, val.shape[2], 1))), 3)
        w[xp.isnan(w)] = 1
        obj.values = xp.nan_to_num(obj.values) * xp.array(w)
        obj.valuation_date = full.valuation_date
        obj._set_slicers()
        return obj

    @property
    def total_process_risk_(self):
        return (self.process_risk_ ** 2).sum(axis="origin").sqrt()

    def _mack_recursion(self, est, X=None):
        obj = X.copy()
        xp = obj.get_array_module()
        risk_arr = xp.zeros((*X.shape[:3], 1))
        if est == "total_parameter_risk_":
            nans = None
            risk_arr = risk_arr[..., 0:1, :]
            future_std_err = (
                X._full_triangle_ - X[X.valuation < X.valuation_date]
            ).iloc[:, :, :, : X.shape[3]] * X.std_err_.values
            t1_t = xp.nan_to_num(future_std_err.sum("origin").values)
            obj.odims = obj.odims[0:1]
        else:
            nans = xp.nan_to_num(X.nan_triangle[None, None])
            nans = 1 - xp.concatenate((nans, xp.zeros((1, 1, X.shape[2], 1))), 3)
            full_tri = X._full_triangle_.values[..., : len(X.ddims)]
            if est == "parameter_risk_":
                t1_t = xp.nan_to_num(full_tri) * obj.std_err_.values
            else:
                t1_t = xp.nan_to_num(full_tri) * self._get_full_std_err_(X).values
        extend = X.ldf_.shape[-1] - X.shape[-1] + 1
        ldf = X.ldf_.values[..., : len(X.ddims) - 1]
        tail = X.cdf_.values[..., -extend : -extend + 1]
        ldf = xp.array(X.ldf_.get_array_module().concatenate((ldf, tail), -1))
        # Recursive Mack Formula
        for i in range(t1_t.shape[-1]):
            t1 = t1_t[..., i : i + 1] ** 2
            t2 = (ldf[..., i : i + 1] * risk_arr[..., i : i + 1]) ** 2
            t_tot = xp.sqrt(t1 + t2)
            if nans is not None:
                t_tot = t_tot * nans[..., i + 1 : i + 2]
            risk_arr = xp.concatenate((risk_arr, xp.nan_to_num(t_tot)), 3)
        obj.values = risk_arr
        obj.ddims = X._full_triangle_.ddims[list(range(X.shape[-1])) + [-1]]
        obj.valuation_date = X._full_triangle_.valuation_date
        obj._set_slicers()
        return obj

    @property
    def mack_std_err_(self):
        return (self.parameter_risk_ ** 2 + self.process_risk_ ** 2).sqrt()

    @property
    def total_mack_std_err_(self):
        return self._get_total_mack_std_err_(self)

    def _get_total_mack_std_err_(self, obj):
        obj = obj.total_process_risk_ ** 2 + obj.total_parameter_risk_ ** 2
        if obj.array_backend == "sparse":
            out = obj.set_backend("numpy").sqrt().values[..., 0, -1]
        else:
            out = obj.sqrt().values[..., 0, -1]
        return pd.DataFrame(out, index=obj.index, columns=obj.columns)

    @property
    def summary_(self):
        # This might be better as a dataframe
        obj = self.ultimate_.copy()
        cols = (
            self.X_.latest_diagonal.values,
            self.ibnr_.values,
            self.ultimate_.values,
            self.mack_std_err_.values[..., -1:],
        )
        obj.values = obj.get_array_module().concatenate(cols, 3)
        obj.ddims = ["Latest", "IBNR", "Ultimate", "Mack Std Err"]
        obj._set_slicers()
        return obj
