# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.core.io import EstimatorIO

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from chainladder import Triangle
    from collections.abc import Sequence
    from pandas import Period, Timestamp


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
    base_period: int or str, optional
        The period whose factor is set to 1.0, so that ``trend_`` states every
        other period relative to it. Defaults to the latest period of ``axis``.
        A ``base_period`` coarser than the Triangle's grain -- a bare year against
        a quarterly axis, say -- resolves to the earliest period it spans. Note
        that ``trend_`` remains a multiplier *to* the base period's cost level; a
        cost level index rising with time is its reciprocal.
    full_triangle: bool (default=False)
        By default ``trend_`` is shaped like the Triangle it was fit on, so cells
        past the valuation date, and any the Triangle is missing internally, come
        back as NaN. When True, ``trend_`` is instead the factor surface over the
        whole origin x development rectangle, which methods needing an n x n trend
        matrix require. Because factors then run past the valuation date rather
        than being clipped at it, a Triangle whose valuation date falls part-way
        through an origin period will not reproduce the default exactly on the
        cells the two share.

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

    By default, trend factors are set relative the most recent origin or valuation period.
    This can be changed via ``base_period``, which sets the trend factors to be relative
    to that period.

    .. testcode::

        tri = cl.load_sample("raa")
        rebased = cl.Trend(0.05, axis="origin", base_period=1981).fit(tri)
        print(np.round(rebased.trend_, 4))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                 12      24      36      48      60      72      84      96      108  120
        1981  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0
        1982  0.9524  0.9524  0.9524  0.9524  0.9524  0.9524  0.9524  0.9524  0.9524  NaN
        1983  0.9070  0.9070  0.9070  0.9070  0.9070  0.9070  0.9070  0.9070     NaN  NaN
        1984  0.8638  0.8638  0.8638  0.8638  0.8638  0.8638  0.8638     NaN     NaN  NaN
        1985  0.8227  0.8227  0.8227  0.8227  0.8227  0.8227     NaN     NaN     NaN  NaN
        1986  0.7835  0.7835  0.7835  0.7835  0.7835     NaN     NaN     NaN     NaN  NaN
        1987  0.7462  0.7462  0.7462  0.7462     NaN     NaN     NaN     NaN     NaN  NaN
        1988  0.7107  0.7107  0.7107     NaN     NaN     NaN     NaN     NaN     NaN  NaN
        1989  0.6768  0.6768     NaN     NaN     NaN     NaN     NaN     NaN     NaN  NaN
        1990  0.6446     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  NaN

    Toggle ``full_triangle=True`` to retrun a full triangle of trend factors.

    .. testcode::

        print(np.round(cl.Trend(0.05, axis="valuation", full_triangle=True).fit(tri).trend_, 4))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                 12      24      36      48      60      72      84      96      108     120
        1981  1.5513  1.4775  1.4071  1.3401  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000
        1982  1.4775  1.4071  1.3401  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000  0.9524
        1983  1.4071  1.3401  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000  0.9524  0.9070
        1984  1.3401  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000  0.9524  0.9070  0.8638
        1985  1.2763  1.2155  1.1576  1.1025  1.0500  1.0000  0.9524  0.9070  0.8638  0.8227
        1986  1.2155  1.1576  1.1025  1.0500  1.0000  0.9524  0.9070  0.8638  0.8227  0.7835
        1987  1.1576  1.1025  1.0500  1.0000  0.9524  0.9070  0.8638  0.8227  0.7835  0.7462
        1988  1.1025  1.0500  1.0000  0.9524  0.9070  0.8638  0.8227  0.7835  0.7462  0.7107
        1989  1.0500  1.0000  0.9524  0.9070  0.8638  0.8227  0.7835  0.7462  0.7107  0.6768
        1990  1.0000  0.9524  0.9070  0.8638  0.8227  0.7835  0.7462  0.7107  0.6768  0.6446

    """

    def __init__(
        self,
        trends: float | int | list[float | int] = 0.0,
        dates: tuple | list[tuple] | None = None,
        axis: Literal["origin", "valuation", 2, -2] = "origin",
        base_period: int | str | None = None,
        full_triangle: bool = False,
    ):
        self.trends = trends
        self.dates = dates
        self.axis = axis
        self.base_period = base_period
        self.full_triangle = full_triangle

    def _accumulate(
        self,
        obj: Triangle,
        trends: Sequence[float | int],
        dates: Sequence[tuple],
        default_start: Timestamp,
    ) -> Triangle:
        """
        Apply each trend segment to ``obj`` in turn, compounding the segments.

        Parameters
        ----------
        obj: Triangle
            The Triangle the segments are applied to. Passing a Triangle of 1s
            yields the factors themselves; passing data yields trended data.
        trends: sequence of float
            The annual trend of each segment, expressed as a decimal.
        dates: sequence of tuple
            The ``(start, end)`` bounds of each segment, positionally paired with
            ``trends``. Either bound may be None.
        default_start: Timestamp
            The default starting date of a segment, if the segment has no starting date.

        Returns
        -------
        Triangle
            ``obj`` multiplied by the compounded factors of every segment.
        """
        for i, trend in enumerate(trends):
            start = default_start if dates[i][0] is None else dates[i][0]
            obj = obj.trend(trend=trend, axis=self.axis, start=start, end=dates[i][1])
        return obj

    @staticmethod
    def _grid(X: Triangle) -> Triangle:
        """
        Fill X with 1s, including lower triangle NaNs, creating a full triangle of 1s.

        Parameters
        ----------
        X : Triangle,
            The triangle to fill.

        Returns
        -------
        Triangle
            A full triangle of 1s, on the numpy backend.

        Notes
        -----
        The grid is densified because it is a full rectangle: every cell is
        occupied, so a COO array would store a coordinate per cell and gain
        nothing. Trending it on the sparse backend measures several times slower
        than trending it dense.
        """
        grid = X.copy().set_backend("numpy")
        grid.valuation_date = grid.valuation.max()
        return (grid * 0 + 1).fillna(1)

    def _latest_period(self, X: Triangle) -> Period:
        """
        The latest period of the trended axis, which the estimator normalizes on
        when no ``base_period`` is given.

        Parameters
        ----------
        X: Triangle
            The Triangle being fit.

        Returns
        -------
        Period
            The last origin when trending on origin, otherwise the period of X's
            valuation date.
        """
        if self.axis in ["origin", 2, -2]:
            return X.origin[-1]
        return pd.Timestamp(X.valuation_date).to_period("M")

    def _get_rebasing_factor(
        self,
        factors: Triangle,
        base_period: int | str | Period,
    ) -> float | int:
        """
        Calculate a scalar used to adjust a triangle of trend factors to the period
        specified by base_period.

        Parameters
        ----------
        factors: Triangle
            A set of trend factors, prior to base period adjustment. Must be a full triangle.
        base_period: int, str or Period
            The period which the trend factors are relative to.

        Returns
        -------
        float | int
            The factor at ``base_period``.

        Raises
        ------
        ValueError
            If ``base_period`` matches no period on the axis being trended.
        """
        period = pd.Period(str(base_period))
        lo, hi = period.to_timestamp(how="s"), period.to_timestamp(how="e")
        values = np.asarray(factors.values)[0, 0]
        if self.axis in ["origin", 2, -2]:
            starts = factors.origin.to_timestamp(how="s")
            matches = np.where((starts >= lo) & (starts <= hi))[0]
            position = (int(matches[0]), 0) if len(matches) else None
            axis_label = "origin"
            first, last = factors.origin[0], factors.origin[-1]
        # Case valuation.
        else:
            valuation = pd.DatetimeIndex(np.array(factors.valuation))
            matches = np.argwhere(
                ((valuation >= lo) & (valuation <= hi)).reshape(
                    factors.shape[-2:], order="f"
                )
            )
            position = tuple(matches[0]) if len(matches) else None
            axis_label = "valuation"
            first, last = f"{valuation.min():%Y-%m}", f"{valuation.max():%Y-%m}"

        if position is None:
            raise ValueError(
                f"base_period {base_period!r} does not match any {axis_label} period. "
                f"{axis_label.capitalize()}s run {first} through {last}."
            )
        return float(values[position])

    def fit(
        self,
        X: Triangle,
        y: None = None,  # noqa, needed for Pipeline
        sample_weight: Triangle | None = None,  # noqa
    ) -> Trend:
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
        """
        trends = self.trends if isinstance(self.trends, list) else [self.trends]
        dates = [(None, None)] if self.dates is None else self.dates
        dates = dates if isinstance(dates, list) else [dates]
        if type(dates[0]) is not tuple:
            raise AttributeError(
                "Dates must be specified as a tuple of start and end dates"
            )
        grid = self._grid(X)
        factors = self._accumulate(
            obj=grid,
            trends=trends,
            dates=dates,
            default_start=(
                grid.valuation_date if self.full_triangle else X.valuation_date
            ),
        )
        anchor = (
            self._latest_period(X) if self.base_period is None else self.base_period
        )
        self.trend_ = factors / self._get_rebasing_factor(factors, anchor)
        if not self.full_triangle:
            self.trend_ = self.trend_ * (X / X)
            self.trend_.valuation_date = X.valuation_date
        if X.array_backend != self.trend_.array_backend:
            # _grid densifies, so hand back whatever backend came in. Only
            # full_triangle reaches here: masking by `X / X` above already
            # carries the default path back to X's backend.
            self.trend_ = self.trend_.set_backend(X.array_backend)
        return self

    def transform(self, X, y=None, sample_weight=None):
        """
        If X and self are of different shapes, align self to X, else
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
            self.base_trend,
            self.axis,  # , start=dates[i][0], end=dates[i][1]
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
