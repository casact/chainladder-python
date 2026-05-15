# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.core.io import EstimatorIO
import pandas as pd


class Trend(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Estimator to create and apply trend factors to a Triangle object.  Allows
    for compound trends as well as storage of the trend matrix to be used in
    other estimators, such as `CapeCod`.

    Parameters
    ----------

    trends: list-like
        The list containing the annual trends expressed as a decimal. For example,
        5% decrease should be stated as -0.05
    dates: list of date-likes
        A list-like of (start, end) dates to correspond to the `trend` list.
    axis: str (options: [‘origin’, ‘valuation’])
        The axis on which to apply the trend

    Attributes
    ----------

    trend_:
        A triangle representation of the trend factors

    Examples
    --------
    The same annual decimal trend is applied along ``origin`` or
    ``valuation`` axes, producing different factor surfaces.

    .. testsetup::

        import chainladder as cl
        import numpy as np

    .. testcode::

        tri = cl.load_sample("raa")
        origin = cl.Trend(0.05, axis="origin").fit(tri)
        val = cl.Trend(0.05, axis="valuation").fit(tri)
        print(round(float(origin.trend_.values[0, 0, 2, 3]), 6))
        print(round(float(val.trend_.values[0, 0, 2, 3]), 6))

    .. testoutput::

        1.4071
        1.215506

    Multiple ``trends`` with paired ``dates`` compound only across the
    windows you specify, so the factors need not match a single flat trend.

    .. testcode::

        tri = cl.load_sample("raa")
        flat = cl.Trend(0.10, axis="origin").fit(tri)
        piece = cl.Trend(
            trends=[0.05, 0.05],
            dates=[(None, "1985"), ("1985", None)],
            axis="origin",
        ).fit(tri)
        print(round(float(flat.trend_.values[0, 0, 0, 0]), 6))
        print(round(float(piece.trend_.values[0, 0, 0, 0]), 6))

    .. testoutput::

        2.357948
        1.551328

    ``trend_`` holds the compounded factor surface; ``transform`` applies it
    so a downstream ``CapeCod`` can be run with ``trend=0`` while still
    reflecting the staged annual assumptions.

    .. testcode::

        tr = cl.load_sample("clrd")[["CumPaidLoss", "EarnedPremDIR"]].sum()
        t_step = cl.Trend(
            trends=[0.04, 0.02],
            dates=[(None, "1995"), ("1995", None)],
            axis="origin",
        ).fit(tr["CumPaidLoss"])
        paid_leveled = t_step.transform(tr["CumPaidLoss"])
        ibnr = (
            cl.CapeCod()
            .fit(
                paid_leveled,
                sample_weight=tr["EarnedPremDIR"].latest_diagonal,
            )
            .ibnr_
        )
        print(round(float(t_step.trend_.values[0, 0, 2, 3]), 6))
        print(int(round(float(np.nansum(ibnr.values)), 0)))

    .. testoutput::

        1.21562
        29278236

    """

    def __init__(self, trends=0.0, dates=None, axis="origin"):
        self.trends = trends
        self.dates = dates
        self.axis = axis

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
        """
        trends = self.trends if type(self.trends) is list else [self.trends]
        dates = [(None, None)] if self.dates is None else self.dates
        dates = [dates] if type(dates) is not list else dates
        if type(dates[0]) is not tuple:
            raise AttributeError(
                "Dates must be specified as a tuple of start and end dates"
            )
        self.trend_ = X.copy()
        for i, trend in enumerate(trends):
            self.trend_ = self.trend_.trend(
                trend, self.axis, start=dates[i][0], end=dates[i][1]
            )
        self.trend_ = self.trend_ / X
        return self

    def transform(self, X, y=None, sample_weight=None):
        """If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X: Triangle
            The triangle to be transformed

        Returns
        -------
            X_new: New triangle with transformed attributes.
        """
        X_new = X.copy()
        triangles = ["trend_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new


class TrendConstant(BaseEstimator, TransformerMixin, EstimatorIO):
    # """
    # Estimator to create and apply trend factors to a Triangle object.  Allows
    # for compound trends as well as storage of the trend matrix to be used in
    # other estimators, such as `CapeCod`.

    # Parameters
    # ----------

    # trends: list-like
    #     The list containing the annual trends expressed as a decimal. For example,
    #     5% decrease should be stated as -0.05
    # dates: list of date-likes
    #     A list-like of (start, end) dates to correspond to the `trend` list.
    # axis: str (options: [‘origin’, ‘valuation’])
    #     The axis on which to apply the trend

    # Attributes
    # ----------

    # trend_:
    #     A triangle representation of the trend factors

    # """

    def __init__(
        self,
        base_trend=0.0,
        trend_from="mid",
        trend_to_date=None,
        # dates=None,
        axis="origin",
    ):
        self.base_trend = base_trend
        self.trend_from = trend_from
        self.trend_to_date = trend_to_date
        self.axis = axis

    def fit(self, X, y=None, sample_weight=None):
        # """Fit the model with X.

        # Parameters
        # ----------
        # X: Triangle-like
        #     Data to which the model will be applied.
        # y: Ignored
        # sample_weight: Ignored

        # Returns
        # -------
        # self: object
        #     Returns the instance itself.
        # """
        print("IN TrendConstant FIT")
        print("base_trend", self.base_trend)

        self.trendedvalues_ = X.copy().trend(
            self.base_trend, self.axis  # , start=dates[i][0], end=dates[i][1]
        )
        print("self.trendedvalues_\n", self.trendedvalues_)

        # if type(dates[0]) is not tuple:
        #     raise AttributeError(
        #         "Dates must be specified as a tuple of start and end dates"
        #     )
        # self.trend_ = X.copy()
        # for i, trend in enumerate(trends):
        #     self.trend_ = self.trend_.trend(
        #         trend, self.axis, start=dates[i][0], end=dates[i][1]
        #     )
        self.trendfactor_ = self.trendedvalues_ / X
        print("self.trendfactor_\n", self.trendfactor_)
        return self

    def transform(self, X, y=None, sample_weight=None):
        # """ If X and self are of different shapes, align self to X, else
        # return self.

        # Parameters
        # ----------
        # X: Triangle
        #     The triangle to be transformed

        # Returns
        # -------
        #     X_new: New triangle with transformed attributes.
        # """
        X_new = X.copy()
        # triangles = ["trend_"]
        # for item in triangles:
        #     setattr(X_new, item, getattr(self, item))
        # X_new._set_slicers()
        return X_new
