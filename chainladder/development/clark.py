# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations
from chainladder.development.base import DevelopmentBase
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy import ndarray
    from types import ModuleType


class ClarkLDF(DevelopmentBase):
    """An Estimator that allows for curve fitting development patterns according
    to Clark 2003.

    The method fits incremental triangle amounts to one of
    "loglogistic" or "weibull" growth curves.  Both of Clark's methods, LDF and Cape
    Cod, can be estimated.  To invoke the Cape Cod method, include
    "sample_weight" in when fitting the estimator.


    Parameters
    ----------
    growth: {'loglogistic', 'weibull'}
        The growth function to be used in curve fitting development patterns.
        Options are 'loglogistic' and 'weibull'
    groupby:
        An option to group levels of the triangle index together for the purposes
        of estimating patterns.  If omitted, each level of the triangle
        index will receive its own patterns.

    Attributes
    ----------
    ldf_: Triangle
        The estimated loss development patterns.
    cdf_: Triangle
        The estimated cumulative development patterns.
    incremental_fits_: Triangle
        The fitted incrementals of the model.
    theta_: DataFrame
        Estimates of the theta parameter of the growth curve.
    omega_: DataFrame
        Estimates of the omega parameter of the growth curve.
    elr_: DataFrame
        The Expected Loss Ratio parameter. This only exists when a ``sample_weight``
        is provided to the Estimator.
    scale_: DataFrame
        The scale parameter of the model.
    norm_resid_: Triangle
        The "Normalized" Residuals of the model according to Clark.

    Examples
    --------
    An actuary fitting Clark's method must choose between the two growth
    curve families. Within the triangle the two curves often fit similar
    development patterns, but the loglogistic curve carries a heavier tail,
    so the choice drives how much development the model projects beyond the
    observed data. Fitting both curves to the same triangle and comparing
    the full fitted pattern, along with the percent of ultimate each curve
    implies has emerged at a mature age (here, 240 months), shows how much
    of the estimate rides on the curve selection.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        import numpy as np

        tri = cl.load_sample("ukmotor")
        m_log = cl.ClarkLDF(growth="loglogistic").fit(tri)
        m_wei = cl.ClarkLDF(growth="weibull").fit(tri)
        print(np.round(m_log.ldf_.values[0, 0, 0, :], 3))
        print(np.round(m_wei.ldf_.values[0, 0, 0, :], 3))
        print(float(np.round(m_log.G_(240.0).values[0, 0, 0, 0], 3)))
        print(float(np.round(m_wei.G_(240.0).values[0, 0, 0, 0], 3)))

    .. testoutput::

        [1.917 1.268 1.141 1.089 1.063 1.047]
        [1.912 1.276 1.143 1.087 1.058 1.04 ]
        0.85
        0.993

    The in-triangle factors are nearly identical, but the curves disagree
    about the tail: the loglogistic fit implies only 85% of ultimate has
    emerged by 240 months versus more than 99% for the weibull fit. The
    actuary should validate the implied tail against other benchmarks
    before relying on either curve.

    To use Clark's Cape Cod method instead of the LDF method, pass a
    premium vector as ``sample_weight`` when fitting. No other parameters
    are needed. The fitted estimator records the method in ``method_`` and
    exposes the fitted expected loss ratios in ``elr_``.

    .. testcode::

        clrd = cl.load_sample("clrd").groupby("LOB").sum()
        m = cl.ClarkLDF().fit(
            clrd["CumPaidLoss"],
            sample_weight=clrd["EarnedPremDIR"].latest_diagonal,
        )
        print(m.method_)
        print(m.elr_.round(3))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

        cape_cod
                  CumPaidLoss
        LOB
        comauto         0.680
        medmal          0.701
        othliab         0.624
        ppauto          0.826
        prodliab        0.671
        wkcomp          0.698

    The ``clrd`` sample holds 775 company-level triangles, and most
    individual companies are too small to support their own curve fit.
    Rather than running a separate optimization on each thin triangle, the
    actuary can pass ``groupby`` to pool the companies into one fit per
    line of business, producing a single ``(omega_, theta_)`` parameter
    pair for each ``LOB``.

    .. testcode::

        clrd = cl.load_sample("clrd")[["CumPaidLoss"]]
        print(len(clrd.index))
        m = cl.ClarkLDF(groupby="LOB").fit(clrd)
        print(m.omega_.round(2))
        print(m.theta_.round(2))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

        775
                  CumPaidLoss
        LOB                  
        comauto         1.08
        medmal          1.89
        othliab         1.47
        ppauto          1.15
        prodliab        1.44
        wkcomp          1.11
                  CumPaidLoss
        LOB                  
        comauto        20.48
        medmal         35.13
        othliab        37.75
        ppauto         10.02
        prodliab       64.35
        wkcomp         20.11

    """

    def __init__(
            self,
            growth: str = "loglogistic",
            groupby=None
    ):
        self.growth: str = growth
        self.groupby = groupby

    def _G(
            self,
            age,
            theta: float = None,
            omega: float = None
    ):
        """Growth function.

        Parameters
        ----------

        theta: float
            The scale parameter of the growth function.
        omega: float
            The shape, or "warp" parameter of the growth function.

        """
        xp: ModuleType = self.incremental_act_.get_array_module()
        if theta is None:
            theta = self.theta_.values[..., None, None]
        if omega is None:
            omega = self.omega_.values[..., None, None]
        age[age == 0.0] = xp.nan
        if self.growth == "loglogistic":
            out = 1 + (theta ** omega) * (age ** (-omega))
        elif self.growth == "weibull":
            out = 1 / (1 - xp.exp(-((age / theta) ** omega)))
        else:
            ValueError(str(self.growth) + "is an invalid growth curve.")
        out[xp.isnan(out)] = xp.inf # noqa
        return out

    def G_(self, age):
        """
        Growth function of the estimator.

        Parameters
        ----------
        age : int, float or array
            The age(s) at which to compute the value of the growth curve.

        Returns
        -------
        Triangle
            A Triangle object with growth curve values
        """
        xp = self.incremental_act_.get_array_module()
        if type(age) in [int, float, xp.int64, xp.float64]:
            age = xp.array([age]).astype("float64")
        if type(age) == list:
            age = xp.array([age]).astype("float64")
        obj = self.incremental_act_.copy()
        obj.odims = obj.odims[0:1]
        obj.values = 1 / self._G(age)
        obj.ddims = age
        return obj

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the munich adjustment will be applied.
        y : Ignored
        sample_weight : Triangle-like
            Exposure vector used to invoke the Cape Cod method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        from chainladder import options

        backend = X.array_backend
        if backend != "numpy":
            obj = X.set_backend("numpy", deep=True)
        else:
            obj = X.copy()
        obj = self._set_fit_groups(obj)
        X = self._set_fit_groups(X)
        xp = obj.get_array_module()
        nan_triangle = obj.nan_triangle
        ld = obj.latest_diagonal
        self.incremental_act_ = obj.cum_to_incr()
        if sample_weight:
            self.method_ = "cape_cod"
            sample_weight = self._set_fit_groups(sample_weight)
        else:
            self.method_ = "ldf"
        age_offset = {"Y": 6.0, "S": 3, "Q": 1.5, "M": 0.5}[X.development_grain]
        age_interval = {"Y": 12.0, "S": 6.0, "Q": 3.0, "M": 1.0}[X.development_grain]
        nans = nan_triangle.reshape(1, -1)[0]
        age = xp.tile(X.ddims, len(X.odims))[~xp.isnan(nans)].astype("float64")
        age_end = age - age_offset
        age_start = xp.maximum(age_end - age_interval, 0).astype("float64")
        origin = np.repeat(X.odims, len(X.ddims))[~xp.isnan(nans)]
        latest_diagonal = obj[X.valuation == X.valuation_date].sum("origin")
        latest_age = latest_diagonal.ddims.astype("float64")
        latest_origin = X.odims
        latest_diagonal = latest_diagonal.values[..., 0, :]

        index = xp.argsort(latest_origin)
        sorted_latest_origin = latest_origin[index]
        sorted_index = xp.searchsorted(sorted_latest_origin, origin)
        map_index = xp.take(index, sorted_index, mode="clip")

        increments = obj.cum_to_incr().values.reshape(X.shape[0], X.shape[1], -1, 1)[
            :, :, ~xp.isnan(nans), 0
        ]

        params = []
        for idx in range(len(X.index)):
            idx_params = []
            for col in range(len(X.columns)):

                def solver(x: ndarray):
                    """ Solve Loglogistic MLE"""
                    ldf = lambda age: self._G(age, theta=x[..., 1], omega=x[..., 0])
                    if sample_weight:
                        ult = (
                            sample_weight.values[idx : idx + 1, col : col + 1, ::-1, 0]
                            * x[..., 2]
                        )
                    else:
                        ult = (
                            ldf(age=latest_age - age_offset)
                            * latest_diagonal[idx : idx + 1, col : col + 1]
                        )
                    increment_fit = ult[..., ::-1][..., map_index] * (
                        1 / ldf(age=age_end) - 1 / ldf(age=age_start)
                    )
                    mle = (
                        increments[idx : idx + 1, col : col + 1] * xp.log(increment_fit)
                        - increment_fit
                    )
                    return -xp.sum((xp.nan_to_num(mle.flatten())))

                if sample_weight:
                    x0 = xp.array([[[[1.0, age_interval, 1.0]]]])
                    bounds = ((1e-6, None), (1e-6, None), (1e-6, None))
                else:
                    x0 = xp.array([[[[1.0, age_interval]]]])
                    bounds = ((1e-6, None), (1e-6, None))
                idx_params.append(
                    minimize(fun=solver, x0=x0.flatten(), bounds=bounds).x.reshape(1, 1, 1, -1)
                )
            params.append(xp.concatenate(idx_params, axis=1))
        params = xp.concatenate(params, axis=0)
        cdf = self._G(
            latest_age - age_offset, theta=params[..., 1:2], omega=params[..., 0:1]
        )
        obj.values = cdf[..., :-1] / cdf[..., 1:]
        obj.ddims = X.link_ratio.ddims
        obj.odims = obj.odims[0:1]
        obj.is_pattern = True
        obj.is_cumulative = False
        obj._set_slicers()
        self.ldf_ = obj
        self.ldf_.valuation_date = pd.to_datetime(options.ULT_VAL)
        rows = X.index.set_index(X.key_labels).index
        self.omega_ = pd.DataFrame(params[..., 0, 0], index=rows, columns=X.vdims)
        self.theta_ = pd.DataFrame(params[..., 0, 1], index=rows, columns=X.vdims)
        if sample_weight:
            self.elr_ = pd.DataFrame(params[..., 0, 2], index=rows, columns=X.vdims)
        ultimate_ = (
            self._G(age=(latest_age - age_offset)[::-1]).swapaxes(-1, -2)
            * ld.values
        )
        self.incremental_fits_ = X.copy()
        self.incremental_fits_.array_backend = "numpy"
        self.incremental_fits_.values = (
            (
                1 / self._G(X.ddims - age_offset)
                - 1 / self._G(xp.maximum(X.ddims - age_offset - age_interval, 0))
            )
            * ultimate_[..., ::-1]
            * nan_triangle
        )
        self.incremental_fits_.is_cumulative = False
        if backend == "cupy":
            self.set_backend("cupy", inplace=True)
        return self

    def transform(self, X):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = X.copy()
        triangles = [
            "ldf_",
            "omega_",
            "theta_",
            "incremental_fits_",
            "G_",
            "_G",
            "growth",
            "scale_",
        ]
        X_new.group_index = self._set_transform_groups(X_new)
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        if hasattr(self, "elr_"):
            X_new.elr_ = self.elr_
        return X_new

    @property
    def scale_(self):
        xp = self.incremental_fits_.get_array_module()
        scale = (
            (
                ((self.incremental_fits_ - self.incremental_act_) ** 2)
                / self.incremental_fits_
            )
            .sum("origin")
            .sum("development")
        )
        df = xp.nansum(self.incremental_fits_.nan_triangle) - 2
        if self.method_ == "ldf":
            df = df - len(self.incremental_fits_.odims)
        else:
            df = df - 1
        if scale.shape != ():
            scale = scale.values[..., 0, 0] / df
        else:
            scale = [[scale / df]]
        return pd.DataFrame(
            scale,
            index=self.incremental_fits_.index.set_index(
                self.incremental_fits_.key_labels
            ).index,
            columns=self.incremental_fits_.columns,
        )

    @property
    def norm_resid_(self):
        resid = (self.incremental_act_.values - self.incremental_fits_.values) / (
            np.sqrt(self.scale_.values[..., None, None] * self.incremental_fits_.values)
        )
        obj = self.incremental_fits_.copy()
        obj.values = resid
        return obj
