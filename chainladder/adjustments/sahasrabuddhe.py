"""
Implement the Sahasrabuddhe layer adjustment.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from chainladder.adjustments.trend import Trend
from chainladder.core.io import EstimatorIO
from chainladder.development import DevelopmentConstant
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from chainladder import Triangle


class LEV(BaseEstimator, TransformerMixin, EstimatorIO):
    r"""
    Limited expected values of an exponential claim size model.

    For a claim size distribution with mean :math:`\theta`, the limited expected
    value at a limit :math:`d` is the average claim once losses are capped at
    that limit. Under an exponential model that is available in closed form:

    .. math::

        \mathrm{LEV}(d) = \theta \left(1 - e^{-d / \theta}\right)

    Sahasrabuddhe (2010) uses these to convert claims development patterns
    between layers: the limit adjustment factor between two layers is a ratio
    of limited expected values. A layer is named by ``attachment`` and
    ``limit``, and its expected loss comes back as ``lev_``.

    Parameters
    ----------
    means: Triangle or dict
        The output of the claim size analysis -- the mean claim size by
        development age, expressed at a single cost level. Either a Triangle
        whose development axis matches the Triangle being fit, or a mapping of
        development age to mean, e.g. ``{12: 28138, 24: 84242, ...}``.

        A mapping is expanded onto the latest origin period of ``X``, giving a
        one-origin-row Triangle. A Triangle is taken as given, which is what
        lets a caller pass means that have already been restated to a cost
        level that varies by origin as well as by development age.
    trend: Triangle or Trend, optional
        Either a Triangle, or a :class:`Trend` to be fitted.

        A Triangle is a cost level index over the origin x development
        rectangle, stating each cell's cost level relative to a common base.
        Build it from a fitted :class:`Trend` with ``full_triangle=True``, which
        is required since the index has to cover cells the Triangle itself does
        not reach. A cost level index runs the other way from ``trend_``, so it
        is the reciprocal: ``1 / trend.trend_``.

        A Trend need not be fitted. ``fit`` fits a copy of it on ``X`` with
        ``full_triangle=True`` and takes the reciprocal of its ``trend_``, so
        the caller's object is left untouched. The result is available as
        ``trend_``. A Trend carries one trend and one axis; a compound index,
        such as an exposure period trend combined with a calendar period trend,
        has to be built by hand and passed as a Triangle.

        ``means`` is taken to be stated at the cost level of the latest origin
        period. Given a trend, ``fit`` restates it onto every cell, which is
        equation 3.1 of the paper:

        .. math::

            \Phi_{i,j} = \theta_j \, \frac{T_{i,j}}{T_{n,j}}

        where :math:`\theta_j` is ``means`` at development age :math:`j`,
        :math:`T` is the cost level index, :math:`n` is the latest origin
        period, and :math:`\Phi_{i,j}` is the resulting ``means_``.

        Only ratios of the index are used, so its own base period is
        immaterial: rescaling it by any constant leaves the result unchanged.
        Left as None, ``means`` is used as given and no restatement happens.
    attachment: float (default=0.0)
        The limit at which the layer attaches.
    limit: float (default=np.inf)
        The limit at which the layer exhausts, i.e. where claims are capped.
        The default leaves the layer unlimited, so ``lev_`` is the unlimited
        mean -- ``M(Phi)`` in the paper's notation.

    Attributes
    ----------
    means_: Triangle
        ``means`` resolved against the Triangle that was fit, restated onto
        every cell's own cost level when a ``trend`` was given. These are the
        claim size model's parameters, ``Phi`` in the paper's notation.
    trend_: Triangle
        The cost level index ``means`` was restated with: ``trend`` as given
        when it was a Triangle, or the reciprocal of the fitted Trend's
        ``trend_``. None when no ``trend`` was supplied.
    lev_: Triangle
        The expected loss in the layer, ``LEV(limit) - LEV(attachment)`` --
        equation 2.4. With the default layer this is the unlimited mean, which
        for an exponential equals ``means_``: the parameter of an exponential
        is its mean. The paper prints the two as separate exhibits (C2 and C3)
        because a model with more than one parameter would distinguish them.

    Examples
    --------

    .. testsetup::

        import chainladder as cl

    .. testcode::

        genins = cl.load_sample("genins")
        thetas = {
            12: 28138, 24: 84242, 36: 133998, 48: 182460, 60: 204649,
            72: 228245, 84: 252830, 96: 265063, 108: 275707, 120: 280000,
        }
        lev = cl.LEV(means=thetas, limit=2000000).fit(genins)
        print(lev.lev_.round(0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                  12       24        36        48        60        72        84        96        108       120
        2010  28138.0  84242.0  133998.0  182457.0  204637.0  228209.0  252737.0  264923.0  275512.0  279779.0

    Attaching at the narrower limit of the data triangle gives the layer
    between the two:

    .. testcode::

        layer = cl.LEV(means=thetas, attachment=1000000, limit=2000000)
        print(layer.fit(genins).lev_.round(0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

              12   24    36     48      60      72      84      96      108     120
        2010  0.0  1.0  77.0  757.0  1533.0  2820.0  4750.0  5954.0  7137.0  7651.0

    """

    # Fitted attributes.
    means_: Triangle
    trend_: Triangle | None
    lev_: Triangle

    def __init__(
        self,
        means: Triangle | dict[int, float | int] | None = None,
        trend: Triangle | Trend | None = None,
        attachment: float | int = 0.0,
        limit: float | int = np.inf,
    ):
        self.means = means
        self.trend = trend
        self.attachment = attachment
        self.limit = limit

    @staticmethod
    def _base_row(
        X: Triangle,  # noqa - sklearn convention
    ) -> slice:
        """
        The latest origin period's row of X's origin axis.

        Parameters
        ----------
        X: Triangle
            The Triangle being fit.

        Returns
        -------
        slice
            A one-row slice of the origin axis. ``means`` and ``trend`` are both
            stated at this period's cost level.
        """
        latest = X.shape[-2] - 1
        return slice(latest, latest + 1)

    @staticmethod
    def _limited_expected_value(
        means: Triangle,
        limit: float | int,
    ) -> Triangle:
        """
        The limited expected value of an exponential model at a single limit.

        Parameters
        ----------
        means: Triangle
            The claim size model's parameters, which for an exponential are also
            its means.
        limit: float
            The limit at which claims are capped. ``np.inf`` gives the unlimited
            mean, which for this model is the parameter itself.

        Returns
        -------
        Triangle
            A triangle of limited expected values.

        Raises
        ------
        ValueError
            If ``limit`` is negative.
        """
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}.")
        if np.isinf(limit):
            return means.copy()
        if limit == 0:
            return means * 0.0
        return means * (1 - (-limit / means).exp())

    def _resolve_means(
        self,
        X: Triangle,  # noqa - sklearn convention
    ) -> Triangle:
        """
        Resolve ``means`` against ``X`` into a Triangle of mean claim sizes.

        Parameters
        ----------
        X: Triangle
            The Triangle being fit, which supplies the development axis a
            mapping is expanded onto.

        Returns
        -------
        Triangle
            A Triangle of means. One origin row when ``means`` is a mapping,
            otherwise whatever shape ``means`` already had.

        Raises
        ------
        ValueError
            If ``means`` is None, or a mapping that does not cover every
            development age of ``X``.
        """
        if self.means is None:
            raise ValueError(
                "means is required. Supply the mean claim size by development "
                "age, either as a Triangle or as a dictionary mapping such as "
                "{12: 28138, 24: 84242, ...}."
            )
        if not isinstance(self.means, dict):
            return self.means.copy()
        ages = list(X.development)
        missing = [age for age in ages if age not in self.means]
        if missing:
            raise ValueError(
                f"means is missing development age(s) {missing}. It must cover "
                f"every development age of the Triangle: {ages}."
            )
        row = X.iloc[0, 0, self._base_row(X), :].copy().set_backend("numpy")
        row.valuation_date = row.valuation.max()
        row = (row * 0 + 1).fillna(1)
        return row * np.array([self.means[age] for age in ages], dtype="float64")

    def _resolve_trend(
        self,
        X: Triangle,  # noqa - sklearn convention
    ) -> Triangle | None:
        """
        Resolve ``trend`` against ``X`` into a cost level index.

        Parameters
        ----------
        X: Triangle
            The Triangle being fit, which an unfitted Trend is fitted on.

        Returns
        -------
        Triangle or None
            The index as given when ``trend`` is a Triangle, the reciprocal of
            the fitted ``trend_`` when it is a Trend, or None without a trend.
        """
        if not isinstance(self.trend, Trend):
            return self.trend
        # The index has to cover cells past the valuation date, so the full
        # rectangle is not optional. Fit a clone so that the caller's object is
        # not changed, and so that fitting twice does not share state.
        fitted = clone(self.trend).set_params(full_triangle=True).fit(X)
        # `trend_` is the multiplier TO the base period's cost level, which is
        # the reciprocal of an index of cost levels.
        return 1 / fitted.trend_

    def fit(
        self,
        X: Triangle,  # noqa - sklearn convention
        y=None,  # noqa - expected by sklearn API
        sample_weight=None,  # noqa - expected by sklearn API
    ) -> LEV:
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
            Returns the instance itself, with fitted limited expected value parameters.

        Raises
        ------
        ValueError
            If the layer exhausts at or below its attachment point, if
            ``trend`` does not share X's origin and development axes, or if it
            is not defined across the whole latest origin period. Also if
            ``means`` is a one-origin Triangle with gaps.
        """
        if self.limit <= self.attachment:
            raise ValueError(
                f"limit ({self.limit}) must exceed attachment ({self.attachment})."
            )

        means_at_base = self._resolve_means(X)
        self._validate_means()
        self.trend_ = self._resolve_trend(X)
        if self.trend_ is None:
            self.means_ = means_at_base
        else:
            base_row = self._base_row(X)
            self._validate_trend(X, base_row)
            # `means` describes the latest origin period. Spreading it over the
            # rectangle is a ratio of the index, so the index's own base period
            # cancels.
            self.means_ = (
                means_at_base * self.trend_ / self.trend_.iloc[..., base_row, :]
            )

        self.lev_ = self._limited_expected_value(
            self.means_, self.limit
        ) - self._limited_expected_value(self.means_, self.attachment)
        return self

    def _validate_means(self) -> None:
        """
        Check that a one-origin Triangle of ``means`` has no gaps.

        Raises
        ------
        ValueError
            If the row has gaps.
        """
        if (
            self.means is None
            or isinstance(self.means, dict)
            or self.means.shape[-2] != 1
        ):
            return

        values = np.asarray(self.means.set_backend("numpy").values)
        if np.isnan(values).any():
            raise ValueError(
                "means has gaps in its single origin row. Every development "
                "age must be populated; a row sliced from a Triangle is "
                "masked past its latest age."
            )

    def _validate_trend(
        self,
        X: Triangle,  # noqa - sklearn convention
        base_row: slice,
    ) -> None:
        """
        Check that ``trend`` can restate ``means`` across the whole rectangle.

        Parameters
        ----------
        X: Triangle
            The Triangle being fit.
        base_row: slice
            The latest origin period's row of the origin axis.

        Raises
        ------
        ValueError
            If the axes do not match, or the latest period's row has a gap.
        """
        # Compare labels rather than shapes: a trend built against a different
        # Triangle of the same dimensions would otherwise pass silently and
        # restate every cell by the wrong factor.
        for axis in ("origin", "development"):
            mine = list(getattr(self.trend_, axis))
            theirs = list(getattr(X, axis))
            if mine != theirs:
                raise ValueError(
                    f"trend does not share X's {axis} axis: trend runs "
                    f"{mine[0]} through {mine[-1]}, X runs {theirs[0]} through "
                    f"{theirs[-1]}. Build the trend against X, with "
                    "full_triangle=True so it covers the whole rectangle."
                )

        # Everything is divided by the latest period's row of the index, so a gap
        # anywhere in it propagates NaN across the whole result. A trend fitted
        # without full_triangle=True is shaped like the Triangle, which leaves
        # that row almost entirely empty -- and the failure is silent, so it is
        # worth catching here rather than letting it surface as missing data.
        base_factors = np.asarray(
            self.trend_.iloc[..., base_row, :].set_backend("numpy").values
        )
        if np.isnan(base_factors).any():
            raise ValueError(
                f"trend is not defined across the whole {X.origin[base_row][0]} "
                "origin period, which every cell is restated against. Refit the "
                "trend with full_triangle=True so it covers cells past the "
                "valuation date."
            )

    def transform(self, X: Triangle, y=None, sample_weight=None) -> Triangle:
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
        triangles = ["means_", "lev_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new


class Sahasrabuddhe(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Restate claims data, or a development pattern, onto another claim size
    layer and a common cost level.

    Claims development is specific to the limit the data is reported at and to
    the cost level of each cell. Sahasrabuddhe (2010) relates the two, so one
    basis can be converted into another without refitting. What ``fit``
    receives decides which conversion it does.

    **Given a Triangle of claims**, every observation is restated to the basic
    limit at the latest origin period's cost level -- equation 3.2::

        triangle_(i,j) = X(i,j) * LEV(basic_limit; Phi(base,j))
                                / LEV(data_limit;  Phi(i,j))

    The numerator is one row, the basis everything moves to; the denominator
    varies cell by cell, which is what makes the ratio a combined trend and
    limit adjustment. Ordinary development then follows in a Pipeline, with
    nothing left in the data for the factors to be contaminated by.

    **Given a development pattern** fitted at ``basic_limit``, the pattern for
    the target layer follows from a ratio of layer expectations -- equations
    3.8 and 3.9 -- with no refitting and no second triangle::

        factor(i,j) = cdf(j) * [ L(i,last) / A(last) ] / [ L(i,j) / A(j) ]

    where ``L`` is the target layer's expected loss and ``A`` is
    ``basic_limit``'s. ``L(i,last) / L(i,j)`` is how much of exposure period
    i's ultimate loss in the new layer has emerged by age j, and the ``A``
    terms restate that onto the pattern's own basis.

    Parameters
    ----------
    means: Triangle or dict
        The mean claim size by development age at a single cost level, as
        :class:`LEV` takes it. This is the output of the claim size analysis,
        stated at the latest origin period's cost level.
    trend: Triangle or Trend
        Either a cost level index as a Triangle, or a :class:`Trend` to be
        fitted, as :class:`LEV` takes it. A Triangle index covers the origin x
        development rectangle and states each cell's cost level relative to a
        common base; build it from a fitted :class:`Trend` with
        ``full_triangle=True`` and take the reciprocal of ``trend_``.

        Only ratios of this index are used, so its own base period is
        immaterial: rescaling it by any constant leaves the result unchanged.

        When a pattern is fitted rather than a Triangle, this is also what
        supplies the origin axis: a pattern has only one row, so the rectangle
        has nowhere else to come from. That needs a Triangle: an unfitted Trend
        has no Triangle to be fitted on, so it is only accepted when fitting
        claims.
    data_limit: float
        The claim size limit the observed data is reported at. Used only when
        fitting a Triangle, since a pattern carries no data to restate.
    basic_limit: float
        The claim size limit that forms the common basis -- ``B`` in the
        paper. A Triangle is restated onto it, and a pattern is assumed to
        have been fitted on it.
    target_layer: tuple of float
        The layer to restate to, as ``(attachment, exhaustion)``. Use
        ``np.inf`` for the exhaustion of an unlimited top layer, e.g.
        ``(2_000_000, np.inf)``. A ground-up layer attaches at zero:
        ``(0, 500_000)``.

        When fitting a Triangle this must be ``(0, basic_limit)``: restating
        claims is a move onto the basis, and any other layer would be a
        silently different calculation.

    Attributes
    ----------
    means_: Triangle
        The claim size model parameters restated to every cell's own cost
        level -- ``means`` spread over the rectangle by ``trend``.
    triangle_: Triangle
        Set when a Triangle was fitted: the restated claims, shaped like the
        Triangle that was fit.
    cdf_: Triangle
        Set when a pattern was fitted: cumulative development factors for the
        target layer, taken along the latest diagonal so that each exposure
        period develops from its own age at its own cost level.
    ldf_: Triangle
        Set when a pattern was fitted: ``cdf_`` as age-to-age factors.
    full_cdf_: Triangle
        Set when a pattern was fitted: cumulative factors over the whole origin
        x development rectangle, before collapsing to a diagonal. Each row is
        one exposure period's complete pattern at its own cost level, so a
        column shows how the same factor varies with cost level. The cells past
        the latest diagonal are the factors a future valuation would use.

        The development axis keeps the Triangle's own ages rather than pattern
        labels, so that valuation arithmetic still works on the rectangle.
        Column ``j`` holds the factor from age ``j`` to ultimate.
    full_ldf_: Triangle
        Set when a pattern was fitted: ``full_cdf_`` as age-to-age factors,
        ``full_cdf_(i,j) / full_cdf_(i,j+1)``. Column ``j`` holds the factor
        from age ``j`` to the next age, and the last column is 1.0. A layer
        with no expected loss at an age gives a non-finite cumulative factor
        there, and the ratio of two such factors is NaN.

        Note that its diagonal is **not** ``ldf_``, and the two answer
        different questions. Each row here divides within itself, so it is one
        exposure period's own pattern, entirely at that period's cost level.
        ``ldf_`` divides along the diagonal of ``full_cdf_``, which steps up a
        row with every age, so consecutive factors come from different
        exposure periods -- which is what makes it the pattern to apply to a
        Triangle, where each period sits at a different age.

    """

    def __init__(
        self,
        means: Triangle | dict[int, float] | None = None,
        trend: Triangle | Trend | None = None,
        data_limit: float | None = None,
        basic_limit: float | None = None,
        target_layer: tuple[float, float] | None = None,
    ):
        self.means = means
        self.trend = trend
        self.data_limit = data_limit
        self.basic_limit = basic_limit
        self.target_layer = target_layer

    def fit(
        self,
        X: Triangle,
        y=None,
        sample_weight: Triangle | None = None,
    ) -> Sahasrabuddhe:
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
        if self.trend is None:
            raise ValueError("trend is required. Build one with cl.Trend(...).")
        for name in ("basic_limit", "target_layer"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required.")
        # A JSON round trip turns the tuple into a list, so normalise rather
        # than compare against whatever shape it came back as.
        try:
            attachment, exhaustion = (float(x) for x in self.target_layer)
        except (TypeError, ValueError):
            raise ValueError(
                "target_layer must be a pair of limits, (attachment, "
                f"exhaustion), e.g. (0, {self.basic_limit}). Got "
                f"{self.target_layer!r}."
            ) from None
        if exhaustion <= attachment:
            raise ValueError(
                f"target_layer exhausts at {exhaustion} and attaches at "
                f"{attachment}; the exhaustion must be the larger of the two."
            )
        self._target = (attachment, exhaustion)

        pattern = self._as_cdf(X)
        if pattern is not None and isinstance(self.trend, Trend):
            raise ValueError(
                "a development pattern has a single origin row, so there is no "
                "Triangle to fit trend on. Pass trend as a Triangle, e.g. "
                "1 / cl.Trend(..., full_triangle=True).fit(triangle).trend_."
            )
        # A pattern has a single origin row, so it cannot supply the rectangle
        # the claim size model is spread over. The trend can, and has to cover
        # it anyway.
        grid = self.trend if pattern is not None else X

        base_row = self._base_slice = LEV._base_row(grid)

        # LEV owns the claim size model: given the trend it restates `means`
        # from the latest origin period onto every cell's own cost level, and
        # validates that the trend can do so.
        self.lev_ = LEV(means=self.means, trend=self.trend).fit(grid)
        self.means_ = self.lev_.means_

        if pattern is None:
            self._fit_triangle(X, base_row)
        else:
            self._fit_pattern(pattern, base_row)
        return self

    @staticmethod
    def _as_cdf(X) -> Triangle | None:
        """
        The cumulative factors ``X`` carries, or None if it carries claims.

        Parameters
        ----------
        X: Triangle or estimator
            A Triangle of claims, a Triangle of cumulative factors, or a
            fitted development estimator.

        Returns
        -------
        Triangle or None
            The pattern, or None when ``X`` is claims data to be restated.
        """
        # Order matters. A Triangle that has been through a development step
        # carries a cdf_ of its own, so asking for cdf_ first would read claims
        # data as a pattern. is_pattern is what actually distinguishes them,
        # and only a Triangle has it.
        if hasattr(X, "is_pattern"):
            return X if X.is_pattern else None
        # A fitted development estimator. Anything without cdf_ is neither a
        # pattern nor a Triangle, and fails on use rather than being guessed at.
        return getattr(X, "cdf_", None)

    def _fit_triangle(self, X: Triangle, base_row: slice) -> None:
        """
        Restate claims onto ``basic_limit`` -- equation 3.2.

        Parameters
        ----------
        X: Triangle
            The claims to restate.
        base_row: slice
            The latest origin period's row of the origin axis.

        Raises
        ------
        ValueError
            If ``data_limit`` is missing, or ``target_layer`` is a layer
            other than ``(0, basic_limit)``.
        """
        if self.data_limit is None:
            raise ValueError("data_limit is required to restate a Triangle.")
        if self._target != (0.0, float(self.basic_limit)):
            raise ValueError(
                "restating a Triangle moves it onto basic_limit, so the target "
                f"layer must be (0, {self.basic_limit}). Got "
                f"{self._target}. Fit a development pattern instead to restate "
                "onto another layer."
            )
        target = LEV._limited_expected_value(self.means_, self.basic_limit).iloc[
            ..., base_row, :
        ]
        self.triangle_ = (
            X * target / LEV._limited_expected_value(self.means_, self.data_limit)
        )
        # The limited expected values cover the whole rectangle, so the product
        # inherits their valuation date rather than X's. The values are already
        # masked correctly -- it is the metadata that is wrong -- but leaving it
        # would give the result a nan_triangle and a latest diagonal belonging
        # to a Triangle that runs years past the data.
        self.triangle_.valuation_date = X.valuation_date

    def _fit_pattern(self, cdf: Triangle, base_row: slice) -> None:
        """
        Restate a pattern onto the target layer -- equations 3.8 and 3.9.

        Parameters
        ----------
        cdf: Triangle
            Cumulative development factors fitted at ``basic_limit``.
        base_row: slice
            The latest origin period's row of the origin axis.
        """
        shape = self.trend.set_backend("numpy").copy()
        ages = list(shape.development)

        aligned = self._cdf_by_age(cdf, ages)
        layer = LEV._limited_expected_value(
            self.means_, self._target[1]
        ) - LEV._limited_expected_value(self.means_, self._target[0])
        anchor = LEV._limited_expected_value(self.means_, self.basic_limit).iloc[
            ..., base_row, :
        ]

        layer_values = np.asarray(layer.set_backend("numpy").values)
        anchor_values = np.asarray(anchor.set_backend("numpy").values)
        emerged = layer_values / anchor_values
        factors = aligned * (layer_values[..., -1:] / anchor_values[..., -1:]) / emerged

        surface = shape.copy()
        surface.valuation_date = surface.valuation.max()
        surface.values = factors
        self.full_cdf_ = surface

        # Age-to-age is the ratio of neighbouring cumulative factors, which is
        # the same rule the diagonal pattern follows. The last column has no
        # next age to divide by and is the identity, matching how `ldf_` ends.
        ratios = np.ones_like(factors)
        ratios[..., :-1] = factors[..., :-1] / factors[..., 1:]
        full_ldf = surface.copy()
        full_ldf.values = ratios
        self.full_ldf_ = full_ldf

        # `cdf_` is the diagonal of that surface -- each exposure period at its
        # own age and its own cost level. A pattern carries no valuation date to
        # take the diagonal at, and the trend's own runs to the end of the
        # rectangle, so the diagonal is anchored on the newest origin period's
        # first development age. That is the latest diagonal whenever the trend
        # was built against the data, which is what `trend` is for.
        valuation = np.array(shape.valuation).reshape(shape.shape[-2:], order="F")
        shape.valuation_date = pd.Timestamp(valuation[-1, 0])

        try:
            fitted = self._as_pattern(shape, factors, ages)
        except ValueError as error:
            # An incomplete diagonal makes no usable pattern, but it takes
            # nothing away from the rectangle -- defined for every cell.
            # Warning rather than raising keeps full_cdf_/full_ldf_ reachable.
            warnings.warn(f"{error} cdf_ and ldf_ are not set.", UserWarning)
            return
        self.cdf_, self.ldf_ = fitted.cdf_, fitted.ldf_

    def transform(self, X: Triangle, y=None, sample_weight=None) -> Triangle:
        """
        Return X restated onto the common basis.

        Parameters
        ----------
        X: Triangle
            The triangle to be transformed.

        Returns
        -------
            X_new: New triangle with adjusted values.
        """
        if hasattr(self, "cdf_"):
            X_new = X.copy()
            # cdf_ is a read-only property of a Triangle, derived from ldf_.
            for item in ("means_", "ldf_", "full_cdf_", "full_ldf_"):
                setattr(X_new, item, getattr(self, item))
            X_new._set_slicers()
            return X_new

        # Restate the Triangle that was handed in, not the one fit() happened
        # to see -- the claim size model is the fitted state, the data is not.
        target = LEV._limited_expected_value(self.means_, self.basic_limit).iloc[
            ..., self._base_slice, :
        ]
        X_new = X * target / LEV._limited_expected_value(self.means_, self.data_limit)
        X_new.valuation_date = X.valuation_date
        X_new.means_ = self.means_
        X_new._set_slicers()
        return X_new

    def _cdf_by_age(self, development, ages: list) -> np.ndarray:
        """
        Align a development pattern to ``ages``, by age rather than by position.

        Parameters
        ----------
        development: estimator or Triangle
            Anything carrying a ``cdf_`` -- a fitted development estimator or a
            Triangle it has transformed -- or a Triangle of cumulative factors.
        ages: list
            The development ages to align to.

        Returns
        -------
        ndarray
            Factors shaped ``(..., 1, len(ages))``. An age the pattern does not
            reach takes 1.0, which is the identity a missing tail implies.

        Raises
        ------
        ValueError
            If ``development`` is not a development pattern.
        """
        cdf = getattr(development, "cdf_", development)
        if not hasattr(cdf, "ddims") or cdf.shape[-2] != 1:
            raise ValueError(
                "development must be a fitted development estimator or a "
                "Triangle of cumulative development factors."
            )
        values = np.asarray(cdf.set_backend("numpy").values)
        ddims = list(cdf.ddims)
        out = np.ones(values.shape[:-1] + (len(ages),), dtype="float64")
        for position, age in enumerate(ages):
            if age in ddims:
                out[..., position] = values[..., ddims.index(age)]
        return out

    @staticmethod
    def _as_pattern(shape: Triangle, factors, ages: list) -> DevelopmentConstant:
        """
        Collapse a factor surface to its latest diagonal, as a pattern.

        Parameters
        ----------
        shape: Triangle
            The Triangle the factors are shaped like, numpy backed.
        factors: ndarray
            The factor surface, over origin x development.
        ages: list
            The development ages of ``shape``.
        style: {'ldf', 'cdf'}
            The form to return.

        Returns
        -------
        Triangle
            A development pattern carrying the diagonal's factors.

        Notes
        -----
        The diagonal is located by valuation rather than by counting back from
        the last row, because a Triangle whose development axis outruns its
        origin axis has ages that no exposure period has reached -- a quarterly
        development grain on annual origins reaches only every fourth age. Those
        ages are left out of the pattern.

        NaN is not a usable signal here either: a high excess layer genuinely
        has non-finite factors at early ages, and those cells sit on the
        diagonal like any other.
        """
        valuation = pd.DatetimeIndex(np.array(shape.valuation))
        on_diagonal = np.asarray(valuation == shape.valuation_date).reshape(
            shape.shape[-2:], order="f"
        )
        reached = on_diagonal.any(axis=0)
        if not reached.all():
            missing = [age for age, hit in zip(ages, reached) if not hit]
            raise ValueError(
                f"No exposure period has reached development age(s) {missing}, "
                "so the latest diagonal does not cover the whole development "
                "axis and the result would not be a usable pattern. This "
                "happens when the development grain is finer than the origin "
                "grain -- quarterly development on annual origins reaches only "
                "every fourth age. Re-grain the Triangle so the two align, or "
                "read `full_cdf_` instead."
            )
        rows = np.argmax(on_diagonal, axis=0)
        pattern = {
            age: float(factors[..., rows[position], position].flat[0])
            for position, age in enumerate(ages)
        }
        return DevelopmentConstant(patterns=pattern, style="cdf").fit(shape)
