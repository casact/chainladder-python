# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
from chainladder.methods import Chainladder


class MackChainladder(Chainladder):
    """
    Basic stochastic chainladder method popularized by Thomas Mack

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
    cdr_std_err_:
        The one-year standard error of the claims development result by
        origin period (Merz and Wüthrich 2008, ``merz2008``)
    total_cdr_std_err_:
        The one-year standard error of the claims development result across
        all origin periods

    See Also
    --------
    BootstrapODPSample : Generates ODP bootstrap samples for reserve uncertainty.

    Examples
    --------
    Use ``MackChainladder`` when the IBNR point estimate alone is not
    sufficient and a measure of reserve uncertainty is also needed.
    ``summary_`` shows the deterministic chainladder ultimate alongside
    Mack's per-origin prediction error.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        tr = cl.load_sample('ukmotor')
        model = cl.MackChainladder().fit(tr)
        print(model.summary_)

    .. testoutput::

               Latest          IBNR      Ultimate  Mack Std Err
        2007  12690.0           NaN  12690.000000           NaN
        2008  12746.0    350.902024  13096.902024     27.246756
        2009  12993.0   1037.536767  14030.536767     36.524408
        2010  11093.0   2044.859861  13137.859861    144.534287
        2011  10217.0   3663.404483  13880.404483    427.634355
        2012   9650.0   7162.150646  16812.150646    693.166178
        2013   6283.0  14396.919151  20679.919151    901.408385

    ``total_mack_std_err_`` aggregates the prediction error across all
    origins. It exceeds the quadrature sum of the per-origin errors in
    ``summary_`` because parameter risk is correlated across origins: all
    origins share the same estimated age-to-age factors.

    .. testcode::

        print(model.total_mack_std_err_)

    .. testoutput::

        columns        values
        (Total,)  1424.531543

    The Mack standard error is sensitive to how the upstream development
    factors were estimated. Using simple (unweighted) averaging in
    :class:`Development` before fitting ``MackChainladder`` gives equal
    weight to each accident year regardless of size. On a small triangle
    this raises the aggregate standard error relative to volume weighting,
    since thinner years contribute more uncertainty.

    .. testcode::

        tr = cl.load_sample("ukmotor")
        tr_simple = cl.Development(average="simple").fit_transform(tr)
        print(cl.MackChainladder().fit(tr_simple).total_mack_std_err_)

    .. testoutput::

        columns        values
        (Total,)  1591.603339
    """

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the model with X.

        Parameters
        ----------
        X: Triangle
            Data to which the model will be applied.
        y: Ignored
        sample_weight: Ignored

        Returns
        -------
        self: object
            Returns the instance itself.

        Examples
        --------
        After fitting, ``ibnr_`` holds the point estimate per origin and
        ``mack_std_err_`` holds the prediction error. The ratio of the two
        gives a coefficient of variation that shows which origins carry the
        most reserve uncertainty relative to their size.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            import numpy as np
            import pandas as pd

            tr = cl.load_sample('ukmotor')
            model = cl.MackChainladder().fit(tr)
            df = model.ibnr_.to_frame(origin_as_datetime=False).round(1)
            df.columns = ["IBNR"]
            df["Mack Std Err"] = np.round(
                model.mack_std_err_.values[0, 0, :, -1], 1
            )
            print(df)

        .. testoutput::

                     IBNR  Mack Std Err
            2007      NaN           NaN
            2008    350.9          27.2
            2009   1037.5          36.5
            2010   2044.9         144.5
            2011   3663.4         427.6
            2012   7162.2         693.2
            2013  14396.9         901.4
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
        """
        Predicts the Mack chainladder ultimate on a new triangle **X**.

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
        ``predict`` re-applies the fitted age-to-age factors and sigma
        estimates to a new triangle without refitting. A common use is
        sensitivity testing: hold the development pattern fixed and scale
        the reported losses by an adverse factor to see how the Mack
        standard error responds. Under Mack (1993), the prediction error
        scales approximately proportionally with the reserve level because
        sigma is estimated from the original fit and held fixed; applying the
        factors to a uniformly scaled triangle would give exactly proportional
        results, but scaling only the reported diagonal (as below) gives a
        slightly smaller increase.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            model = cl.MackChainladder().fit(tr)
            tr_adverse = tr * 1.05
            print(model.predict(tr).total_mack_std_err_)
            print(model.predict(tr_adverse).total_mack_std_err_)

        .. testoutput::

            columns        values
            (Total,)  1424.531543
            columns        values
            (Total,)  1475.539173
        """
        X_new = super().predict(X, sample_weight)
        X_new.sigma_ = getattr(X_new, "sigma_", self.X_.sigma_)
        X_new.std_err_ = getattr(X_new, "std_err_", self.X_.std_err_)
        X_new.average_ = getattr(X_new, "average_", self.X_.average_)
        X_new.w_ = getattr(X_new, "w_", self.X_.w_)
        X_new._full_triangle_ = X_new.full_triangle_
        X_new.parameter_risk_ = self._mack_recursion("parameter_risk_", X_new)
        X_new.process_risk_ = self._mack_recursion("process_risk_", X_new)
        X_new.total_process_risk_ = (X_new.process_risk_**2).sum(axis="origin").sqrt()
        X_new.total_parameter_risk_ = self._mack_recursion(
            "total_parameter_risk_", X_new
        )
        X_new.full_std_err_ = self._get_full_std_err_(X_new)
        X_new.total_mack_std_err_ = self._get_total_mack_std_err_(X_new)
        X_new.mack_std_err_ = (X_new.parameter_risk_**2 + X_new.process_risk_**2).sqrt()
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

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            model = cl.MackChainladder().fit(tr)
            print(model.full_std_err_)

        .. testoutput::

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
        w = lxp.concatenate(
            (X.w_, lxp.ones((val.shape[0], val.shape[1], val.shape[2], 1))), 3
        )
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

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            model = cl.MackChainladder().fit(tr)
            print(model.total_process_risk_.iloc[..., -3:])

        .. testoutput::

                         72           84           9999
            2007  1039.901929  1069.726277  1069.726277
        """
        return (self.process_risk_**2).sum(axis="origin").sqrt()

    def _mack_recursion(self, est, X=None):
        obj = X.copy()
        backend = X.array_backend
        xp = obj.get_array_module()
        risk_arr = xp.zeros((*X.shape[:3], 1))
        if est == "total_parameter_risk_":
            nans = None
            risk_arr = risk_arr[..., 0:1, :]
            future_std_err = (
                X._full_triangle_ - X[X.valuation < X.valuation_date]
            ).iloc[:, :, :, : X.shape[3]] * X.std_err_.values
            # sum applies auto_sparse, so backend needs to be forced
            t1_t = xp.nan_to_num(
                future_std_err.sum("origin").set_backend(backend).values
            )
            obj.odims = obj.odims[0:1]
        else:
            nans = xp.nan_to_num(X.nan_triangle[None, None])
            nans = 1 - xp.concatenate((nans, xp.zeros((1, 1, X.shape[2], 1))), 3)
            full_tri = X._full_triangle_.values[..., : len(X.ddims)]
            if est == "parameter_risk_":
                # std_err_ is always numpy
                t1_t = (
                    xp.nan_to_num(full_tri) * obj.std_err_.set_backend(backend).values
                )
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

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            model = cl.MackChainladder().fit(tr)
            print(model.mack_std_err_.iloc[..., -3:, -3:])

        .. testoutput::

                        72          84          9999
            2011  415.253333  427.634355  427.634355
            2012  673.828536  693.166178  693.166178
            2013  876.437914  901.408385  901.408385
        """
        return (self.parameter_risk_**2 + self.process_risk_**2).sqrt()

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

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            model = cl.MackChainladder().fit(tr)
            print(model.total_mack_std_err_)

        .. testoutput::

            columns        values
            (Total,)  1424.531543
        """
        return self._get_total_mack_std_err_(self)

    def _get_total_mack_std_err_(self, obj):
        obj = obj.total_process_risk_**2 + obj.total_parameter_risk_**2
        if obj.array_backend == "sparse":
            out = obj.set_backend("numpy").sqrt().values[..., 0, -1]
        else:
            out = obj.sqrt().values[..., 0, -1]
        return pd.DataFrame(out, index=obj.index, columns=obj.columns)

    @property
    def cdr_std_err_(self):
        """
        One-year standard error of the claims development result by origin
        period, Result 3.1 (eq. 3.9) of Merz and Wüthrich (2008, ``merz2008``).

        The claims development result is the change in an origin's estimated
        ultimate once the next diagonal is observed. ``mack_std_err_`` is the
        counterpart to ultimate. The values follow the ``sigma_interpolation``
        of :class:`Development`, and missing cells above the latest diagonal
        give NaN.

        Returns
        -------
        Triangle
            One value per origin, shaped like ``ultimate_``, NaN for the
            fully developed oldest origin.

        Raises
        ------
        ValueError
            If the triangle is not square, has a tail, or its development
            factors are not volume-weighted over every link ratio.

        Examples
        --------
        The triangle of Merz and Wüthrich (2008), with Mack's extrapolation
        for the last sigma.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('mw2008')
            dev = cl.Development(sigma_interpolation='mack').fit_transform(tr)
            model = cl.MackChainladder().fit(dev)
            print(model.cdr_std_err_)

        .. testoutput::

                          2261
            2001           NaN
            2002    566.174395
            2003   1486.560344
            2004   3923.098608
            2005   9722.859763
            2006  28442.621556
            2007  20954.286973
            2008  28119.317963
            2009  53320.821049
        """
        obj = self.ultimate_.set_backend("numpy")
        obj.values = np.sqrt(self._get_cdr_msep()[0])[..., None]
        return obj.set_backend(self.X_.array_backend)

    @property
    def total_cdr_std_err_(self):
        """
        One-year standard error of the claims development result summed over
        all origin periods, Result 3.3 (eq. 3.15) of Merz and Wüthrich (2008,
        ``merz2008``). It exceeds the quadrature sum of ``cdr_std_err_``
        because all origins share the estimated factors, and it raises
        ``ValueError`` where ``cdr_std_err_`` does.

        Returns
        -------
        DataFrame
            One value per (index, column) combination of the fitted Triangle.

        Examples
        --------
        On the paper's triangle the one-year error is about three quarters of
        Mack's error to ultimate.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('mw2008')
            dev = cl.Development(sigma_interpolation='mack').fit_transform(tr)
            model = cl.MackChainladder().fit(dev)
            print(model.total_cdr_std_err_)
            print(model.total_mack_std_err_)

        .. testoutput::

            columns         values
            (Total,)  81080.546787
            columns          values
            (Total,)  108401.387451
        """
        obj = self.X_
        out = np.sqrt(self._get_cdr_msep()[1])
        return pd.DataFrame(out, index=obj.index, columns=obj.columns)

    def _get_cdr_msep(self):
        """
        Mean squared errors of the one-year claims development result.

        Returns the per-origin values, shape (index, columns, origin), and
        the totals over origins, shape (index, columns), from Results 3.1 and
        3.3 of ``merz2008`` with I = J = n - 1. Origins i and development
        periods j are 0-based, as in the paper, and the equation numbers in
        the comments are the paper's.
        """
        from chainladder.development import Development

        X = self.X_.set_backend("numpy")
        C = X if X.is_cumulative else X.incr_to_cum()
        a, b, n, m = C.shape
        msg = "cdr_std_err_ and total_cdr_std_err_ need "
        upper = np.add.outer(np.arange(n), np.arange(m)) <= n - 1
        if n != m or not np.array_equal(~np.isnan(X.nan_triangle), upper):
            raise ValueError(
                msg + "a square triangle valued at its latest diagonal; this "
                f"one has {n} origins and {m} development periods."
            )
        J = m - 1
        ldf = X.ldf_.set_backend("numpy").values[..., 0, :]
        sigma = X.sigma_.set_backend("numpy").values[..., 0, :]
        if not (
            np.all(np.nan_to_num(ldf[..., J:], nan=1.0) == 1)
            and np.all(np.nan_to_num(sigma[..., J:]) == 0)
        ):
            raise ValueError(
                msg + "a pattern without a tail, as Merz and Wüthrich (2008) "
                "assume; fit without a tail estimator or with TailConstant(1.0)."
            )
        average = set(np.asarray(X.average_).ravel())
        if average != {"volume"}:
            raise ValueError(
                msg + "volume-weighted development factors; this pattern uses "
                f"{sorted(str(v) for v in average)}."
            )
        # A default Development on C gives the paper's factors (2.9), and the
        # fitted w_ and ldf_ must match it.
        fresh = Development().fit(C)
        w, w_fresh = np.broadcast_arrays(np.asarray(X.w_), np.asarray(fresh.w_))
        if not np.array_equal(w, w_fresh, equal_nan=True):
            raise ValueError(
                msg + "every link ratio of the triangle; this pattern excludes "
                "some (drop, drop_high, drop_low, drop_valuation, n_periods)."
            )
        f = ldf[..., :J]  # f_j^I, (2.9)
        if not np.allclose(
            f, fresh.ldf_.values[..., 0, :], rtol=1e-8, atol=0, equal_nan=True
        ):
            raise ValueError(
                msg + "development factors estimated from this triangle's "
                "own link ratios; ldf_ differs from them, as after "
                "Development(fillna=...)."
            )
        latest = C.latest_diagonal.values[..., 0]  # C_{i, I-i}
        diag = latest[..., ::-1][..., :J]  # C_{I-j, j}
        S1 = C.sum(axis="origin").values[..., 0, :J]  # S_j^{I+1}, (2.10)
        S = S1 - diag  # S_j^I, (2.9)
        ult = self.ultimate_.set_backend("numpy").values[..., 0]  # C_{i,J}^I, (2.11)
        zero = np.zeros((a, b, 1))

        def later(t):
            """Sum of t over the positions after each position, last axis."""
            return np.concatenate((np.cumsum(t[..., ::-1], -1)[..., -2::-1], zero), -1)

        with np.errstate(divide="ignore", invalid="ignore"):
            q = sigma[..., :J] ** 2 / f**2  # sigma_j^2 (3.3) over (f_j^I)^2
            i = np.arange(1, n)  # origins with a next diagonal
            d = J - i  # their next development period, I - i
            qd, Sd, S1d, Ci = q[..., d], S[..., d], S1[..., d], latest[..., i]
            # Sums over j = I-i+1 .. J-1: the summands of Phi (3.5), and those
            # of the sum that Delta (3.4) and Lambda (3.13) share.
            Phi = later(diag * q / S1**2)[..., d]
            shared = later((diag / S1) ** 2 * q / S)[..., d]
            Psi = qd / Ci  # (3.6)
            Delta = qd / Sd + shared  # (3.4)
            Lam = Ci / S1d * qd / Sd + shared  # Lambda, (3.13)
            Xi = Phi + qd / S1d  # (3.14)
            msep = np.full((a, b, n), np.nan)
            # (3.9), with Gamma = Phi + Psi (3.7)
            msep[..., 1:] = ult[..., i] ** 2 * (Phi + Psi + Delta)
            # (3.15): 2 * sum over k > i > 0 of C_{i,J} C_{k,J} (Xi + Lambda)
            cross = ult[..., i] * later(ult[..., 1:]) * (Xi + Lam)
            total = msep[..., 1:].sum(-1) + 2 * cross.sum(-1)
        complete = ~np.isnan(C.values[..., upper]).any(-1)
        msep[~complete] = np.nan
        total[~complete] = np.nan
        return msep, total

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

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            model = cl.MackChainladder().fit(tr)
            print(model.summary_)

        .. testoutput::

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
        backend = obj.array_backend
        # forcing these four triangles to the same backend
        cols = (
            self.X_.latest_diagonal.set_backend(backend).values,
            self.ibnr_.set_backend(backend).values,
            self.ultimate_.set_backend(backend).values,
            self.mack_std_err_.set_backend(backend).values[..., -1:],
        )
        obj.values = obj.get_array_module().concatenate(cols, 3)
        obj.ddims = ["Latest", "IBNR", "Ultimate", "Mack Std Err"]
        obj._set_slicers()
        return obj
