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
    X_:
        returns **X**
    ultimate_:
        The ultimate losses per the method
    ibnr_:
        The IBNR per the method
    full_expectation_:
        The ultimates back-filled to each development period in **X** replacing
        the known data
    full_triangle_:
        The ultimates back-filled to each development period in **X** retaining
        the known data
    summary_:
        summary of the model
    full_std_err_:
        The full standard error
    total_process_risk_:
        The total process error
    total_parameter_risk_:
        The total parameter error
    mack_std_err_:
        The total prediction error by origin period
    total_mack_std_err_:
        The total prediction error across all origin periods

    Examples
    --------
    Fit the Mack chainladder method and inspect the headline summary table,
    which combines the deterministic chainladder estimate with Mack's
    stochastic standard error.

    .. testsetup:

        import chainladder as cl

    .. testcode:

        tr = cl.load_sample('ukmotor')
        model = cl.MackChainladder().fit(tr)
        print(model.summary_)

    .. testoutput:

               Latest          IBNR      Ultimate  Mack Std Err
        2007  12690.0           NaN  12690.000000           NaN
        2008  12746.0    350.902024  13096.902024     27.246756
        2009  12993.0   1037.536767  14030.536767     36.524408
        2010  11093.0   2044.859861  13137.859861    144.534287
        2011  10217.0   3663.404483  13880.404483    427.634355
        2012   9650.0   7162.150646  16812.150646    693.166178
        2013   6283.0  14396.919151  20679.919151    901.408385

    The deterministic chainladder ultimates match those of
    :class:`Chainladder`. Mack's contribution is the stochastic standard error
    in the rightmost column, which can be aggregated across origins.

    .. testcode:

        print(model.total_mack_std_err_)

    .. testoutput:

        columns        values
        (Total,)  1424.531543
    """

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X: Triangle-like
            Data to which the model will be applied.
        y: Ignored
        sample_weight: Ignored

        Returns
        -------
        self: object
            Returns the instance itself.

        Examples
        --------
        Fitting attaches the ``ultimate_`` and Mack std error attributes to
        the estimator and returns the estimator itself.

        >>> tr = cl.load_sample('ukmotor')
        >>> cl.MackChainladder().fit(tr)
        MackChainladder()
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
        """Predicts the Mack chainladder ultimate on a new triangle **X**.

        The fitted age-to-age factors and sigma estimates from ``self.X_`` are
        applied to ``X`` to compute ``ultimate_`` and the Mack standard errors
        (parameter risk, process risk, ``mack_std_err_``,
        ``total_mack_std_err_``) on the predicted Triangle.

        Parameters
        ----------
        X: Triangle
            Loss data to which the fitted model will be applied. Must share
            the same shape as the Triangle used in :meth:`fit`.
        sample_weight: Triangle
            Optional exposure used in CDF alignment.

        Returns
        -------
        X_new: Triangle
            Triangle with ``ultimate_`` and Mack std error attributes
            attached.

        Examples
        --------
        Fit the model and apply it to a Triangle with the same shape, then
        read the Mack standard error off the resulting Triangle.

        >>> tr = cl.load_sample('ukmotor')
        >>> model = cl.MackChainladder().fit(tr)
        >>> predicted = model.predict(tr)
        >>> predicted.total_mack_std_err_
        columns        values
        (Total,)  1424.531543
        """
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
        """
        Per-cell standard error of the form ``sigma_ / sqrt(c**(2-alpha))``,
        where ``alpha`` reflects the development averaging method
        (regression, volume, simple). This is the building block Mack's
        recursion uses to propagate parameter and process risk forward.

        Returns
        -------
        Triangle
            Per-cell standardized error scale.

        Examples
        --------
        >>> tr = cl.load_sample('ukmotor')
        >>> model = cl.MackChainladder().fit(tr)
        >>> model.full_std_err_
                    12        24        36        48        60        72   84
        2007  0.047826  0.040745  0.031412  0.010337  0.001431  0.001523  0.0
        2008  0.044802  0.038074  0.029815  0.010123  0.001410  0.001500  0.0
        2009  0.042943  0.036708  0.029445  0.009864  0.001361  0.001449  0.0
        2010  0.043241  0.037958  0.030130  0.010154  0.001407  0.001497  0.0
        2011  0.043990  0.037603  0.029468  0.009879  0.001369  0.001457  0.0
        2012  0.039675  0.034017  0.026776  0.008976  0.001243  0.001324  0.0
        2013  0.035752  0.030671  0.024143  0.008094  0.001121  0.001193  0.0
        """
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
        """
        Across-origin process risk by development period.

        Process risk is independent across origins, so the across-origin
        total is the quadrature sum: ``sqrt(sum_i process_risk_i**2)``.
        Reported as a 1-row Triangle (the "(Total)" origin).

        Returns
        -------
        Triangle
            Single-origin Triangle of total process risk by development
            period.

        Examples
        --------
        >>> tr = cl.load_sample('ukmotor')
        >>> model = cl.MackChainladder().fit(tr)
        >>> model.total_process_risk_.iloc[..., -3:]
                     72           84           9999
        2007  1039.901929  1069.726277  1069.726277
        """
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
        """
        Per-(origin, development) Mack prediction error,
        ``sqrt(parameter_risk**2 + process_risk**2)`` (Mack 1993 eq 7).

        Returns
        -------
        Triangle
            Per-cell Mack standard error. The last development column is the
            ultimate-period prediction error per origin.

        Examples
        --------
        Showing the last three origins and last three development periods to
        keep the slice readable. The final column is the ultimate prediction
        error per origin.

        >>> tr = cl.load_sample('ukmotor')
        >>> model = cl.MackChainladder().fit(tr)
        >>> model.mack_std_err_.iloc[..., -3:, -3:]
                    72          84          9999
        2011  415.253333  427.634355  427.634355
        2012  673.828536  693.166178  693.166178
        2013  876.437914  901.408385  901.408385
        """
        return (self.parameter_risk_ ** 2 + self.process_risk_ ** 2).sqrt()

    @property
    def total_mack_std_err_(self):
        """
        Total Mack prediction error aggregated across all origin periods,
        ``sqrt(total_process_risk**2 + total_parameter_risk**2)``.

        Note that this total exceeds the quadrature sum of per-origin
        ``mack_std_err_`` values because parameter risk is correlated across
        origins (each origin's projection uses the same estimated age-to-age
        factors), introducing positive cross-origin covariances.

        Returns
        -------
        DataFrame
            One value per (index, column) combination of the fitted Triangle.

        Examples
        --------
        >>> tr = cl.load_sample('ukmotor')
        >>> model = cl.MackChainladder().fit(tr)
        >>> model.total_mack_std_err_
        columns        values
        (Total,)  1424.531543
        """
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
        """
        Headline Mack summary table by origin: latest diagonal, IBNR,
        ultimate, and Mack standard error.

        Returns
        -------
        Triangle
            Triangle whose four development columns are ``Latest``, ``IBNR``,
            ``Ultimate``, and ``Mack Std Err``.

        Examples
        --------
        >>> tr = cl.load_sample('ukmotor')
        >>> model = cl.MackChainladder().fit(tr)
        >>> model.summary_
               Latest          IBNR      Ultimate  Mack Std Err
        2007  12690.0           NaN  12690.000000           NaN
        2008  12746.0    350.902024  13096.902024     27.246756
        2009  12993.0   1037.536767  14030.536767     36.524408
        2010  11093.0   2044.859861  13137.859861    144.534287
        2011  10217.0   3663.404483  13880.404483    427.634355
        2012   9650.0   7162.150646  16812.150646    693.166178
        2013   6283.0  14396.919151  20679.919151    901.408385
        """
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
