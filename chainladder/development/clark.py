# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development.base import DevelopmentBase
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ClarkLDF(DevelopmentBase):
    """ A Estimator that allows for curve fitting development pattterns according
    to Clark 2003.

    The method fits incremental triangle amounts to one of
    ``loglogistic`` or ``weibull`` growth curves.  Both Clarks methods, LDF and Cape
    Cod, can be estimated.  To invoke the Cape Cod method, include
    ``sample_weight`` in when fitting the estimator.


    Parameters
    ----------
    growth : {'loglogistic', 'weibull'}
        The growth function to be used in curve fitting development patterns.
        Options are 'loglogistic' and 'weibull'
    groupby :
        An option to group levels of the triangle index together for the purposes
        of estimating patterns.  If omitted, each level of the triangle
        index will receive its own patterns.


    Attributes
    ----------
    ldf_ : Triangle
        The estimated loss development patterns.
    cdf_ : Triangle
        The estimated cumulative development patterns.
    incremental_fits_ : Triangle
        The fitted incrementals of the model.
    theta_ : DataFrame
        Estimates of the theta parameter of the growth curve.
    omega_ : DataFrame
        Estimates of the omega parameter of the growth curve.
    elr_ : DataFrame
        The Expected Loss Ratio parameter. This only exists when a ``sample_weight``
        is provided to the Estimator.
    scale_ : DataFrame
        The scale parameter of the model.
    norm_resid_ : Triangle
        The "Normalized" Residuals of the model according to Clark.

    Todos
    -----
    1. Add stochastic functionality
    3. Allow for dropping elements from the fit


    """

    def __init__(self, growth="loglogistic", groupby=None):
        self.growth = growth
        self.groupby = groupby

    def _G(self, age, theta=None, omega=None):
        """ Growth function """
        xp = self.incremental_act_.get_array_module()
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
        out[xp.isnan(out)] = xp.inf
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
        from chainladder import ULT_VAL

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
        age_offset = {"Y": 6.0, "Q": 1.5, "M": 0.5}[X.development_grain]
        age_interval = {"Y": 12.0, "Q": 3.0, "M": 1.0}[X.development_grain]
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

                def solver(x):
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
                    minimize(fun=solver, x0=x0, bounds=bounds).x.reshape(1, 1, 1, -1)
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
        self.ldf_.valuation_date = pd.to_datetime(ULT_VAL)
        rows = X.index.set_index(X.key_labels).index
        self.omega_ = pd.DataFrame(params[..., 0, 0], index=rows, columns=X.vdims)
        self.theta_ = pd.DataFrame(params[..., 0, 1], index=rows, columns=X.vdims)
        if sample_weight:
            self.elr_ = pd.DataFrame(params[..., 0, 2], index=rows, columns=X.vdims)
        ultimate_ = (
            xp.swapaxes(self._G(age=(latest_age - age_offset)[::-1]), -1, -2)
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
