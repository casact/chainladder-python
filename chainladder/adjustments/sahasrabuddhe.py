"""
Implement the Sahasrabuddhe layer adjustment.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from chainladder.adjustments.trend import Trend
from chainladder.core.io import EstimatorIO
from chainladder.development import DevelopmentConstant
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from chainladder import Triangle
    from chainladder.development.base import DevelopmentBase
    from collections.abc import Iterable
    from typing import Any


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
        whose development axis matches the Triangle being fit, or a dictionary
        mapping of development age to mean, e.g. ``{12: 28138, 24: 84242, ...}``.

        A dictionary mapping is assumed to be at the cost level of the latest origin
        period in ``X``. A Triangle is taken as given, which is what
        lets a caller pass means that have already been restated to a cost
        level that varies by origin as well as by development age.
    trend: Triangle or Trend, optional
        Pass a fitted trend as a A Triangle.

        A compound index, such as an exposure period trend
        combined with a calendar period trend, must be passed as a Triangle
        as the result of multiplying its fitted component trends.

        Simple trends can be passed as unfitted ``Trend`` objects.
    attachment: float (default=0.0)
        The attachment point of the layer.
    limit: float (default=np.inf)
        The limit of the layer, i.e. where claims are capped.
        The default leaves the layer unlimited, so ``lev_`` is the unlimited
        mean -- ``M(Phi)`` in the paper's notation.

    Attributes
    ----------
    means_: Triangle
        ``means`` resolved against the Triangle that was fit, restated onto
        every cell's own cost level when a ``trend`` was given. These are the
        claim size model's parameters, ``Phi`` in the paper's notation.
    trend_: Triangle
        The cost level index ``means`` was restated with, relative to the latest
        origin period: ``trend`` as given when it was a Triangle, or the
        reciprocal of the fitted Trend's ``trend_`` rebased to the latest origin
        period. None when no ``trend`` was supplied.
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

    Attaching at the lower limit, that of the data triangle, gives the layer
    between the two:

    .. testcode::

        layer = cl.LEV(means=thetas, attachment=1000000, limit=2000000)
        print(layer.fit(genins).lev_.round(0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

              12   24    36     48      60      72      84      96      108     120
        2010  0.0  1.0  77.0  757.0  1533.0  2820.0  4750.0  5954.0  7137.0  7651.0

    The means are stated at the latest origin period's cost level. Passing a
    ``trend`` restates them onto every other origin period. With a 2% annual
    trend, each earlier origin period's means are 2% lower than the next:

    .. testcode::

        trended = cl.LEV(means=thetas, trend=cl.Trend(0.02, axis="origin"))
        print(trended.fit(genins).means_.round(0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                  12       24        36        48        60        72        84        96        108       120
        2001  23545.0  70490.0  112124.0  152674.0  171241.0  190985.0  211557.0  221793.0  230699.0  234291.0
        2002  24016.0  71900.0  114366.0  155728.0  174666.0  194805.0  215788.0  226229.0  235313.0  238977.0
        2003  24496.0  73338.0  116653.0  158842.0  178159.0  198701.0  220104.0  230753.0  240020.0  243757.0
        2004  24986.0  74804.0  118986.0  162019.0  181722.0  202675.0  224506.0  235368.0  244820.0  248632.0
        2005  25485.0  76301.0  121366.0  165260.0  185357.0  206729.0  228996.0  240076.0  249716.0  253605.0
        2006  25995.0  77827.0  123793.0  168565.0  189064.0  210863.0  233576.0  244877.0  254711.0  258677.0
        2007  26515.0  79383.0  126269.0  171936.0  192845.0  215080.0  238247.0  249775.0  259805.0  263850.0
        2008  27045.0  80971.0  128795.0  175375.0  196702.0  219382.0  243012.0  254770.0  265001.0  269127.0
        2009  27586.0  82590.0  131371.0  178882.0  200636.0  223770.0  247873.0  259866.0  270301.0  274510.0
        2010  28138.0  84242.0  133998.0  182460.0  204649.0  228245.0  252830.0  265063.0  275707.0  280000.0

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
        Resolve ``means`` against ``X`` into a Triangle of mean claim sizes. Used to handle different data types
        that can be passed in as ``means``. If ``means`` is a Triangle, return it. Otherwise, transform it into
        a Triangle.

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
        row = row.fill(1)
        sizes = np.array([self.means[age] for age in ages], dtype="float64")
        row.values = row.values * sizes
        return row

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
        # We need a full triangle index, so fit it against a dummy triangle of ones.
        ones = X.copy().set_backend("numpy")
        ones.valuation_date = ones.valuation.max()
        ones = ones.fill(1)
        trend = deepcopy(self.trend)
        trend.fit(ones)
        # Reciprocal is taken to restate means from the latest year to prior year levels.
        index = 1 / trend.trend_
        return index / index.iloc[..., self._base_row(X), :]

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
        self: LEV
            Returns the instance itself, with fitted limited expected value parameters.

        Raises
        ------
        ValueError
            If the layer exhausts at or below its attachment point, if
            ``trend`` does not share X's origin and development axes, or if it
            has NaN anywhere in its latest origin row. Also if
            ``means`` is a one-origin Triangle with NaNs.
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
            self._validate_trend(
                X=X,
                trend=self.trend_,
                base_row=base_row,
            )
            self.means_ = means_at_base * self.trend_

        self.lev_ = self._limited_expected_value(
            means=self.means_,
            limit=self.limit,
        ) - self._limited_expected_value(
            means=self.means_,
            limit=self.attachment,
        )
        return self

    def _validate_means(self) -> None:
        """
        Check that a one-origin Triangle of ``means`` has no NaNs.

        Raises
        ------
        ValueError
            If the row has NaNs.
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

    @staticmethod
    def _validate_trend(
        X: Triangle,  # noqa - sklearn convention
        trend: Triangle,
        base_row: slice,
    ) -> None:
        """
        Check that ``trend`` can restate ``means`` across the whole triangle.

        Parameters
        ----------
        X: Triangle
            The Triangle being fit.
        trend: Triangle
            The resolved trend to check against X.
        base_row: slice
            The latest origin period's row of the origin axis.

        Raises
        ------
        ValueError
            If the axes do not match, or the latest period's row has a NaN.
        """
        # Make sure the axis labels are the same.
        for axis in ("origin", "development"):
            trend_labels = list(getattr(trend, axis))
            triangle_labels = list(getattr(X, axis))
            if trend_labels != triangle_labels:
                raise ValueError(
                    f"trend does not share X's {axis} axis: trend runs "
                    f"{trend_labels[0]} through {trend_labels[-1]}, X runs {triangle_labels[0]} through "
                    f"{triangle_labels[-1]}. Build the trend against X, fitted on a "
                    "full triangle of ones so it covers the whole triangle."
                )

        # Check that trend is defined across the base row origin period.
        base_factors = np.asarray(
            trend.iloc[..., base_row, :].set_backend("numpy").values
        )
        if np.isnan(base_factors).any():
            raise ValueError(
                f"trend is not defined across the whole {X.origin[base_row][0]} "
                "origin period, which every cell is restated against. Refit the "
                "trend on a full triangle of ones so it covers cells past the "
                "valuation date."
            )

    def transform(
        self,
        X: Triangle,  # noqa - sklearn convention
        y=None,  # noqa - expected by sklearn API
        sample_weight=None,  # noqa - expected by sklearn API
    ) -> Triangle:
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
        X_new = X.copy()  # noqa - sklearn convention
        triangles = ["means_", "lev_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        # noinspection PyProtectedMember
        X_new._set_slicers()
        return X_new


class Sahasrabuddhe(BaseEstimator, TransformerMixin, EstimatorIO):
    r"""
    Restate claims data, or a development pattern, onto another claim size
    layer and a common cost level.

    Claims development is specific to the limit the data is reported at and to
    the cost level of each cell. Sahasrabuddhe (2010) relates the two, so one
    basis can be converted into another without refitting. What ``fit``
    receives decides which conversion it does.

    **Given a Triangle of claims**, every observation is restated to the target
    layer at the latest origin period's cost level -- equation 3.2, which is
    the case of a ground-up target layer :math:`(0, B)`:

    .. math::

        C_{i,j} = X_{i,j} \, \frac{\mathrm{LEV}(p;\, \Phi_{n,j})
                                   - \mathrm{LEV}(a;\, \Phi_{n,j})}
                                  {\mathrm{LEV}(d;\, \Phi_{i,j})}

    where :math:`X` is the observed claims at ``observed_limit`` :math:`d`,
    :math:`a` and :math:`p` are the attachment and limit of ``target_layer``,
    :math:`n` is the latest origin period, and :math:`C` is ``triangle_``.
    An excess layer holds almost no loss at the youngest ages, so development
    fitted directly on it is unstable. The paper restates to the basic limit
    instead and converts the fitted pattern, as below.

    **Given a development pattern** fitted at ``basic_limit``, the pattern for
    the target layer follows from a ratio of layer expectations -- equations
    3.8 and 3.9 -- with no refitting and no second triangle:

    .. math::

        F_{i,j} = F^{B}_{j} \,
            \frac{L_{i,\infty} / A_{\infty}}{L_{i,j} / A_{j}}

    where :math:`F^{B}_{j}` is the ``basic_limit`` pattern's cumulative factor
    at age :math:`j`, :math:`L_{i,j}` is the target layer's expected loss
    and :math:`A_j` is ``basic_limit``'s, at the latest origin period's cost
    level. :math:`L_{i,j} / L_{i,\infty}` is how much of exposure period
    :math:`i`'s ultimate loss in the new layer has emerged by age :math:`j`,
    and the :math:`A` terms restate that onto the pattern's own basis.

    Parameters
    ----------
    means: Triangle or dict
        The mean claim size by development age at a single cost level, as
        :class:`LEV` takes it. This is the output of the claim size analysis,
        stated at the latest origin period's cost level.
    trend: Triangle or Trend, optional
        Either a cost level index as a Triangle, or a :class:`Trend` to be
        fitted, as :class:`LEV` takes it. A Triangle index is in the form of a full
        triangle and is taken to be stated relative to the latest
        origin period.

        Leaving it out assumes no trend.

    observed_limit: float
        The claim size limit of the observed data.
    basic_limit: float
        The claim size limit that a development pattern is assumed to have
        been fitted on -- ``B`` in the paper. Required when fitting a pattern.
        A Triangle is restated onto ``target_layer`` and does not use it.
    target_layer: tuple[float | int, float | int] | None
        The layer to restate to, as ``(attachment, limit)``. Use
        ``np.inf`` for an unlimited top layer, e.g.
        ``(2_000_000, np.inf)``. A ground-up layer attaches at zero:
        ``(0, 500_000)``.

    Attributes
    ----------
    means_: Triangle
        The claim size model parameters restated to every cell's own cost
        level -- ``means`` spread over the full triangle by ``trend``.
    triangle_: Triangle
        Set when a Triangle was fitted: the restated claims at target layer.
    cdf_: Triangle
        Set when a pattern was fitted: cumulative development factors for the
        target layer, taken along the latest diagonal of ``full_cdf_`` so that each exposure
        period develops from its own age at its own cost level.
    ldf_: Triangle
        Set when a pattern was fitted: ``cdf_`` as age-to-age factors.
    full_cdf_: Triangle
        Set when a pattern was fitted: cumulative factors over the full triangle,
        before collapsing to a diagonal. Each row is one exposure period's complete
        pattern at its own cost level, so a column shows how the same factor varies
        with cost level. The cells past the latest diagonal are the factors a future
        valuation would use.
    full_ldf_: Triangle
        Set when a pattern was fitted: ``full_cdf_`` as age-to-age factors,
        ``full_cdf_(i,j) / full_cdf_(i,j+1)``. Column ``j`` holds the factor
        from age ``j`` to the next age, and the last column is 1.0. A layer
        with no expected loss at an age gives a non-finite cumulative factor
        there, and the ratio of two such factors is NaN.

    Examples
    --------

    .. testsetup::

        import chainladder as cl

    Given a 2% origin trend and claim size model, restate the triangle limited to 500k, and calculate the factors.

    .. testcode::

        genins = cl.load_sample("genins")
        thetas = {
            12: 28138, 24: 84242, 36: 133998, 48: 182460, 60: 204649,
            72: 228245, 84: 252830, 96: 265063, 108: 275707, 120: 280000,
        }
        restated = cl.Sahasrabuddhe(
            means=thetas,
            trend=cl.Trend(0.02, axis="origin"),
            observed_limit=1000000,
            target_layer=(0, 500000),
        ).fit(genins)
        development = cl.Development(average="volume").fit(restated.triangle_)
        print(development.cdf_.round(3))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

               12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult
        (All)  12.199   3.503   2.054   1.467   1.278   1.186   1.121   1.078    1.013

    Restate a pattern from one layer to another by supplying it to ``fit()``. When no ``trend`` is given,
    so every origin period is at the same cost level, and the pattern applies to
    all of them. Pass a ``trend`` to give each origin period its own pattern:

    .. testcode::

        layer = cl.Sahasrabuddhe(
            means=thetas,
            basic_limit=500000,
            target_layer=(0, 1000000),
        ).fit(development)
        print(layer.cdf_.round(3))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

               12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
        (All)  14.245   4.079   2.342   1.609   1.373   1.246    1.15   1.093    1.017      1.0
    """

    # Fitted attributes.
    lev_: LEV
    means_: Triangle
    triangle_: Triangle
    cdf_: Triangle
    ldf_: Triangle
    full_cdf_: Triangle
    full_ldf_: Triangle
    _target: tuple[float | int, float | int]
    _base_slice: slice

    def __init__(
        self,
        means: Triangle | dict[int, float | int] | None = None,
        trend: Triangle | Trend | None = None,
        observed_limit: float | int | None = None,
        basic_limit: float | int | None = None,
        target_layer: tuple[float | int, float | int] | None = None,
    ):
        self.means = means
        self.trend = trend
        self.observed_limit = observed_limit
        self.basic_limit = basic_limit
        self.target_layer = target_layer

    def fit(
        self,
        X: Triangle | DevelopmentBase,  # noqa - sklearn convention
        y=None,  # noqa - expected by sklearn API
        sample_weight: Triangle | None = None,  # noqa - expected by sklearn API
    ) -> Sahasrabuddhe:
        """
        Fit the model with X.

        Parameters
        ----------
        X: Triangle or DevelopmentBase
            A Triangle of claims to restate, or a development pattern to
            restate: a Triangle of cumulative factors or a fitted development
            estimator.
        y: Ignored
        sample_weight: Ignored

        Returns
        -------
        self: Sahasrabuddhe
            Returns the instance itself.
        """
        if self.target_layer is None:
            raise ValueError("target_layer is required.")
        try:
            attachment, limit = (float(x) for x in self.target_layer)
        except (TypeError, ValueError):
            raise ValueError(
                "target_layer must be a pair of limits, (attachment, "
                "limit), e.g. (0, 500_000). Got "
                f"{self.target_layer!r}."
            ) from None
        if limit <= attachment:
            raise ValueError(
                f"target_layer exhausts at {limit} and attaches at "
                f"{attachment}; the limit must be the larger of the two."
            )
        self._target = (attachment, limit)

        # Imported here because chainladder.core imports this module.
        from chainladder.core.triangle import Triangle

        pattern = self._as_cdf(X)
        if pattern is None:
            if not isinstance(X, Triangle):
                raise ValueError(
                    "X must be a Triangle of claims, a Triangle of cumulative "
                    "development factors, or a fitted development estimator."
                )
            base_row = self._fit_claim_size_model(X)
            self._fit_triangle(X, base_row)
        else:
            if self.basic_limit is None:
                raise ValueError("basic_limit is required to restate a pattern.")
            if isinstance(self.trend, Trend):
                raise ValueError(
                    "a development pattern has a single origin row, so there is no "
                    "Triangle to fit trend on. Pass trend as a Triangle: a Trend "
                    "fitted on a full triangle of ones, then inverted."
                )
            # A pattern has a single origin row, so the rectangle comes from the
            # trend, or from the ages of the means when there is none.
            if self.trend is None:
                grid = self._constant_cost_level(pattern)
            else:
                grid = self.trend
            base_row = self._fit_claim_size_model(grid)
            self._fit_pattern(pattern, base_row)
        return self

    def _fit_claim_size_model(self, grid: Triangle) -> slice:
        """
        Restate ``means`` onto every cell of ``grid``.

        Parameters
        ----------
        grid: Triangle
            The origin by development rectangle the claim size model covers.

        Returns
        -------
        slice
            The latest origin period's row of the origin axis.
        """
        # noinspection PyProtectedMember
        base_row = self._base_slice = LEV._base_row(grid)

        # Restate `means` from the latest origin period onto every cell's own cost level.
        self.lev_ = LEV(
            means=self.means,
            trend=self.trend,
        ).fit(grid)
        self.means_ = self.lev_.means_
        return base_row

    def _constant_cost_level(self, pattern: Triangle) -> Triangle:
        """
        The one origin row a pattern is spread over when there is no trend.

        Without a ``trend`` every origin period sits at the same cost level.

        Parameters
        ----------
        pattern: Triangle
            The development pattern being restated.

        Returns
        -------
        Triangle
            One origin row of ones over the ages of ``means``.

        Raises
        ------
        ValueError
            If ``means`` does not say which ages it is stated at, or leaves a
            gap among the pattern's ages up to its own last age. Pattern ages
            beyond that, such as a tail, are ignored, as they are with a trend.
        """
        if isinstance(self.means, dict):
            ages = sorted(self.means)
        elif self.means is not None and hasattr(self.means, "ddims"):
            ages = [int(age) for age in np.asarray(self.means.ddims)]
        else:
            raise ValueError("means is required.")
        pattern_ages = [int(age) for age in np.asarray(pattern.ddims)]
        last = max(ages)
        missing = [age for age in pattern_ages if age not in ages and age < last]
        if missing:
            raise ValueError(
                f"means is missing development age(s) {missing} of the pattern. "
                f"It must be stated at every age the pattern has up to {last}."
            )
        origin_row = pattern.copy().set_backend("numpy")
        origin_row.values = np.ones(pattern.shape[:3] + (len(ages),))
        origin_row.ddims = np.array(ages)
        origin_row.is_pattern = False
        # noinspection PyProtectedMember
        origin_row._set_slicers()
        return origin_row

    @staticmethod
    def _as_cdf(
        X: Triangle | Any,  # noqa - sklearn convention
    ) -> Triangle | None:
        """
        Extract the cumulative factors ``X`` carries, or return ``None`` if it carries claims.

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
        # Case Triangle of values or patterns.
        if hasattr(X, "is_pattern"):
            return X if X.is_pattern else None
        # Case fitted development estimator.
        return getattr(X, "cdf_", None)

    def _target_at_latest(self, base_row: slice) -> Triangle:
        """
        The target layer's expected loss at the latest origin period's cost level.

        Parameters
        ----------
        base_row: slice
            The latest origin period's row of the origin axis.

        Returns
        -------
        Triangle
            One origin row, over every development age.
        """
        # noinspection PyProtectedMember
        layer = LEV._limited_expected_value(
            means=self.means_,
            limit=self._target[1],
        ) - LEV._limited_expected_value(
            means=self.means_,
            limit=self._target[0],
        )
        if layer.shape[-2] == 1:
            # No trend: the means are already a single origin row.
            return layer
        return layer.iloc[..., base_row, :]

    def _fit_triangle(
        self,
        X: Triangle,  # noqa - sklearn convention
        base_row: slice,
    ) -> None:
        """
        Restate claims onto ``target_layer`` -- equation 3.2.

        Parameters
        ----------
        X: Triangle
            The claims to restate.
        base_row: slice
            The latest origin period's row of the origin axis.

        Raises
        ------
        ValueError
            If ``observed_limit`` is missing.
        """
        if self.observed_limit is None:
            raise ValueError("observed_limit is required to restate a Triangle.")
        # Restate the claims to the target layer at the most recent cost level.
        # noinspection PyProtectedMember
        self.triangle_ = (
            X
            * self._target_at_latest(base_row)
            / LEV._limited_expected_value(means=self.means_, limit=self.observed_limit)
        )
        # Restore the valuation date.
        self.triangle_.valuation_date = X.valuation_date

    def _fit_pattern(
        self,
        cdf: Triangle,
        base_row: slice,
    ) -> None:
        """
        Restate a pattern onto the target layer -- equations 3.8 and 3.9.

        Parameters
        ----------
        cdf: Triangle
            Cumulative development factors fitted at ``basic_limit``.
        base_row: slice
            The latest origin period's row of the origin axis.
        """
        shape = self.means_.set_backend("numpy").copy()
        ages = list(shape.development)

        aligned = self._cdf_by_age(cdf, ages)
        # noinspection PyProtectedMember
        layer = LEV._limited_expected_value(
            means=self.means_,
            limit=self._target[1],
        ) - LEV._limited_expected_value(
            means=self.means_,
            limit=self._target[0],
        )
        # noinspection PyProtectedMember
        anchor = LEV._limited_expected_value(
            means=self.means_,
            limit=self.basic_limit,
        ).iloc[..., base_row, :]

        layer_values = np.asarray(layer.set_backend("numpy").values)
        anchor_values = np.asarray(anchor.set_backend("numpy").values)
        emerged = layer_values / anchor_values
        factors = aligned * (layer_values[..., -1:] / anchor_values[..., -1:]) / emerged

        # The factors take their index and columns from whichever input carries
        # them: the pattern, the trend, or the means.
        full_cdf = self._labeled(shape, factors, (cdf, shape, self.means_))
        full_cdf.valuation_date = full_cdf.valuation.max()
        self.full_cdf_ = full_cdf

        # Calculate the LDFs.
        ratios = np.ones_like(factors)
        ratios[..., :-1] = factors[..., :-1] / factors[..., 1:]
        full_ldf = full_cdf.copy()
        full_ldf.values = ratios
        self.full_ldf_ = full_ldf

        # Take the diagonal at the newest origin period's first development age.
        diagonal = full_cdf.copy()
        diagonal.valuation_date = diagonal.valuation[diagonal.shape[-2] - 1]

        try:
            fitted = self._as_pattern(
                shape=diagonal,
                factors=factors,
                ages=ages,
            )
        except ValueError as error:
            # An incomplete diagonal makes no usable pattern, but it takes
            # nothing away from the triangle -- defined for every cell.
            # Warning rather than raising keeps full_cdf_/full_ldf_ reachable.
            warnings.warn(f"{error} cdf_ and ldf_ are not set.", UserWarning)
            return
        self.cdf_, self.ldf_ = fitted.cdf_, fitted.ldf_

    def transform(
        self,
        X: Triangle,  # noqa - sklearn convention
        y=None,  # noqa - expected by sklearn API
        sample_weight=None,  # noqa - expected by sklearn API
    ) -> Triangle:
        """
        Return X restated onto the target layer.

        Parameters
        ----------
        X: Triangle
            The triangle to be transformed.

        Returns
        -------
            X_new: New triangle with adjusted values.
        """
        if hasattr(self, "cdf_"):
            X_new = X.copy()  # noqa - sklearn convention
            for item in ("means_", "ldf_", "full_cdf_", "full_ldf_"):
                setattr(X_new, item, getattr(self, item))
            # noinspection PyProtectedMember
            X_new._set_slicers()
            return X_new

        # Restate the Triangle to the target layer at the cost level of the latest origin period.
        # noinspection PyProtectedMember
        X_new = (  # noqa sklearn convention
            X
            * self._target_at_latest(self._base_slice)
            / LEV._limited_expected_value(  # noqa - sklearn convention
                means=self.means_, limit=self.observed_limit
            )
        )
        X_new.valuation_date = X.valuation_date
        X_new.means_ = self.means_
        # noinspection PyProtectedMember
        X_new._set_slicers()
        return X_new

    @staticmethod
    def _labeled(
        triangle: Triangle,
        cdf_values: np.ndarray,
        labeled_by: tuple,
    ) -> Triangle:
        """
        Give ``cdf_values`` the axes of ``triangle``, labeled by the right source. Helper function used to
        support multidimensionality.

        Parameters
        ----------
        triangle: Triangle
            Supplies the origin and development axes.
        cdf_values: ndarray
            The array to hold, shaped ``(index, columns, origin, development)``.
        labeled_by: tuple of Triangle
            Candidates to take the index and the columns from, in order of
            preference. The first one whose length on an axis matches
            ``cdf_values`` supplies that axis, so the index and the columns can
            come from different places.

        Returns
        -------
        Triangle
            A copy of ``triangle`` holding ``cdf_values``.
        """
        full_cdf = triangle.copy()
        full_cdf.values = cdf_values
        # A broadcast axis is as long as the longest input along it, so that input
        # labels it. Ties go to the first, in order of preference.
        index_from = max(labeled_by, key=lambda candidate: candidate.shape[0])
        columns_from = max(labeled_by, key=lambda candidate: candidate.shape[1])
        full_cdf.kdims, full_cdf.key_labels = (
            index_from.kdims,
            index_from.key_labels,
        )
        full_cdf.vdims = columns_from.vdims
        # noinspection PyProtectedMember
        full_cdf._set_slicers()
        return full_cdf

    @staticmethod
    def _cdf_by_age(
        development: Triangle | Any,
        ages: Iterable[int],
    ) -> np.ndarray:
        """
        Align a development pattern to ``ages``, by age rather than by position.

        Parameters
        ----------
        development: estimator or Triangle
            Anything carrying a ``cdf_`` -- a fitted development estimator or a
            Triangle it has transformed -- or a Triangle of cumulative factors.
        ages: iterable of int
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
        ages = list(ages)
        out = np.ones(values.shape[:-1] + (len(ages),), dtype="float64")
        for position, age in enumerate(ages):
            if age in ddims:
                out[..., position] = values[..., ddims.index(age)]
        return out

    @staticmethod
    def _as_pattern(
        shape: Triangle,
        factors,
        ages: list,
    ) -> DevelopmentConstant:
        """
        Collapse an array of factors to its latest diagonal, as a pattern.

        Parameters
        ----------
        shape: Triangle
            The Triangle the factors are shaped like, numpy backed.
        factors: ndarray
            The factor array, over origin x development.
        ages: list
            The development ages of ``shape``.

        Returns
        -------
        DevelopmentConstant
            A fitted development pattern carrying the diagonal's factors.

        Notes
        -----
        The diagonal is located by valuation rather than by counting back from
        the last row, because a Triangle whose development axis outruns its
        origin axis has ages that no exposure period has reached -- a quarterly
        development grain on annual origins reaches only every fourth age. Those
        ages are left out of the pattern.
        """
        if shape.shape[-2] == 1:
            # A single origin period is its own diagonal.
            on_diagonal = np.ones(shape.shape[-2:], dtype=bool)
        else:
            valuation = pd.DatetimeIndex(np.array(shape.valuation))
            on_diagonal = np.asarray(valuation == shape.valuation_date).reshape(
                shape.shape[-2:], order="F"
            )
        reached = np.asarray(on_diagonal.any(axis=0))
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
        # One diagonal per index and column: (index, columns, 1, development).
        diagonal = np.stack(
            [
                factors[..., rows[position], position]
                for position in range(factors.shape[-1])
            ],
            axis=-1,
        )[..., None, :]
        ldf = np.concatenate(
            (diagonal[..., :-1] / diagonal[..., 1:], diagonal[..., -1:]), axis=-1
        )
        # DevelopmentConstant takes a single pattern, so it supplies the
        # structure from the first slice and the values are set for every slice.
        pattern = {age: float(diagonal[0, 0, 0, i]) for i, age in enumerate(ages)}
        development = DevelopmentConstant(
            patterns=pattern,
            style="cdf",
        )
        development.fit(shape)
        development.ldf_.values = ldf
        return development
