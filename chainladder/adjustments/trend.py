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
        print(np.round(origin.trend_, 4))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                 12      24      36      48      60      72      84      96      108     120
        1981  1.5513  1.5513  1.5513  1.5513  1.5513  1.5513  1.5513  1.5513  1.5513  1.5513
        1982  1.4775  1.4775  1.4775  1.4775  1.4775  1.4775  1.4775  1.4775  1.4775     NaN
        1983  1.4071  1.4071  1.4071  1.4071  1.4071  1.4071  1.4071  1.4071     NaN     NaN
        1984  1.3401  1.3401  1.3401  1.3401  1.3401  1.3401  1.3401     NaN     NaN     NaN
        1985  1.2763  1.2763  1.2763  1.2763  1.2763  1.2763     NaN     NaN     NaN     NaN
        1986  1.2155  1.2155  1.2155  1.2155  1.2155     NaN     NaN     NaN     NaN     NaN
        1987  1.1576  1.1576  1.1576  1.1576     NaN     NaN     NaN     NaN     NaN     NaN
        1988  1.1025  1.1025  1.1025     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        1989  1.0500  1.0500     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        1990  1.0000     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN

    .. testcode::

        print(np.round(val.trend_, 4))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                 12      24      36      48      60      72      84      96    108  120
        1981  1.5513  1.4775  1.4071  1.3401  1.2763  1.2155  1.1576  1.1025  1.05  1.0
        1982  1.4775  1.4071  1.3401  1.2763  1.2155  1.1576  1.1025  1.0500  1.00  NaN
        1983  1.4071  1.3401  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000   NaN  NaN
        1984  1.3401  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000     NaN   NaN  NaN
        1985  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000     NaN     NaN   NaN  NaN
        1986  1.2155  1.1576  1.1025  1.0500  1.0000     NaN     NaN     NaN   NaN  NaN
        1987  1.1576  1.1025  1.0500  1.0000     NaN     NaN     NaN     NaN   NaN  NaN
        1988  1.1025  1.0500  1.0000     NaN     NaN     NaN     NaN     NaN   NaN  NaN
        1989  1.0500  1.0000     NaN     NaN     NaN     NaN     NaN     NaN   NaN  NaN
        1990  1.0000     NaN     NaN     NaN     NaN     NaN     NaN     NaN   NaN  NaN

    Multiple ``trends`` with paired ``dates`` apply a different annual trend
    to each window, producing a surface that matches neither single flat
    trend. Each tuple runs from the recent anchor back to the older bound, so
    here a 10% trend covers the recent origins (1985 and later) and a 5% trend
    the earlier ones.

    .. testcode::

        tri = cl.load_sample("raa")
        piece = cl.Trend(
            trends=[0.10, 0.05],
            dates=[(None, "1985"), ("1985", None)],
            axis="origin",
        ).fit(tri)
        print(np.round(piece.trend_, 4))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                 12      24      36      48      60      72      84      96      108     120
        1981  2.0429  2.0429  2.0429  2.0429  2.0429  2.0429  2.0429  2.0429  2.0429  2.0429
        1982  1.9456  1.9456  1.9456  1.9456  1.9456  1.9456  1.9456  1.9456  1.9456     NaN
        1983  1.8529  1.8529  1.8529  1.8529  1.8529  1.8529  1.8529  1.8529     NaN     NaN
        1984  1.7647  1.7647  1.7647  1.7647  1.7647  1.7647  1.7647     NaN     NaN     NaN
        1985  1.6105  1.6105  1.6105  1.6105  1.6105  1.6105     NaN     NaN     NaN     NaN
        1986  1.4641  1.4641  1.4641  1.4641  1.4641     NaN     NaN     NaN     NaN     NaN
        1987  1.3310  1.3310  1.3310  1.3310     NaN     NaN     NaN     NaN     NaN     NaN
        1988  1.2100  1.2100  1.2100     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        1989  1.1000  1.1000     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        1990  1.0000     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN

    The recent origins (1985-1990) follow the 10% path exactly, matching a
    single flat 10% trend, while the earlier origins compound the extra 5% on
    top and rise above a flat 5% surface, so the staged factors track neither
    flat trend alone.

    Because ``trend_`` is a stored factor surface, ``transform`` can pre-level
    a triangle so a downstream ``CapeCod`` reflects staged annual assumptions
    that a single scalar ``trend`` could not express. Leveling the losses lifts
    the a-priori for the older years, so total IBNR rises against an unleveled
    fit (29.3M versus 26.4M):

    .. testcode::

        tr = cl.load_sample("clrd")[["CumPaidLoss", "EarnedPremDIR"]].sum()
        sample_weight = tr["EarnedPremDIR"].latest_diagonal
        t_step = cl.Trend(
            trends=[0.04, 0.02],
            dates=[(None, "1995"), ("1995", None)],
            axis="origin",
        ).fit(tr["CumPaidLoss"])
        paid_leveled = t_step.transform(tr["CumPaidLoss"])
        leveled = cl.CapeCod(trend=0).fit(paid_leveled, sample_weight=sample_weight)
        unleveled = cl.CapeCod(trend=0).fit(tr["CumPaidLoss"], sample_weight=sample_weight)
        print(np.round(t_step.trend_, 4))
        print(int(round(float(np.nansum(leveled.ibnr_.values)), 0)))
        print(int(round(float(np.nansum(unleveled.ibnr_.values)), 0)))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                 12      24      36      48      60      72      84      96      108     120
        1988  1.2647  1.2647  1.2647  1.2647  1.2647  1.2647  1.2647  1.2647  1.2647  1.2647
        1989  1.2399  1.2399  1.2399  1.2399  1.2399  1.2399  1.2399  1.2399  1.2399     NaN
        1990  1.2156  1.2156  1.2156  1.2156  1.2156  1.2156  1.2156  1.2156     NaN     NaN
        1991  1.1918  1.1918  1.1918  1.1918  1.1918  1.1918  1.1918     NaN     NaN     NaN
        1992  1.1684  1.1684  1.1684  1.1684  1.1684  1.1684     NaN     NaN     NaN     NaN
        1993  1.1455  1.1455  1.1455  1.1455  1.1455     NaN     NaN     NaN     NaN     NaN
        1994  1.1230  1.1230  1.1230  1.1230     NaN     NaN     NaN     NaN     NaN     NaN
        1995  1.0816  1.0816  1.0816     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        1996  1.0400  1.0400     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        1997  1.0000     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
        29278236
        26370689

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
