# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.core.io import EstimatorIO
from chainladder.development import DevelopmentConstant

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from chainladder import Triangle


class LEV(BaseEstimator, EstimatorIO):
    """
    Limited expected values of an exponential claim size model.

    For a claim size distribution with mean ``theta``, the limited expected
    value at a limit ``d`` is the average claim once losses are capped at that
    limit. Under an exponential model that is available in closed form::

        LEV(d) = theta * (1 - exp(-d / theta))

    Sahasrabuddhe (2010) uses these to convert claims development patterns
    between layers: the limit adjustment factor between two layers is a ratio
    of limited expected values, which is what :meth:`layer` returns.

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

    Attributes
    ----------
    means_: Triangle
        ``means`` resolved against the Triangle that was fit. For an
        exponential model this doubles as the unlimited mean, since the mean of
        an exponential distribution is its only parameter.

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
        lev = cl.LEV(means=thetas).fit(genins)
        print(lev.at(2000000).round(0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                  12       24        36        48        60        72        84        96        108       120
        2010  28138.0  84242.0  133998.0  182457.0  204637.0  228209.0  252737.0  264923.0  275512.0  279779.0

    Capping at the policy limit and at the narrower limit of the data triangle
    gives the layer between them:

    .. testcode::

        print(lev.layer(1000000, 2000000).round(0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

              12   24    36     48      60      72      84      96      108     120
        2010  0.0  1.0  77.0  757.0  1533.0  2820.0  4750.0  5954.0  7137.0  7651.0

    """

    def __init__(self, means: Triangle | dict[int, float] | None = None):
        self.means = means

    def _resolve_means(self, X: Triangle) -> Triangle:
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
                "age, either as a Triangle or as a mapping such as "
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
        # The latest origin is a single cell of data, so its row is mostly NaN.
        # Push the valuation date out to unmask it before laying the means on
        # top -- otherwise every age past the first would come back NaN.
        row = X.iloc[0, 0, -1:, :].copy().set_backend("numpy")
        row.valuation_date = row.valuation.max()
        row = (row * 0 + 1).fillna(1)
        return row * np.array([self.means[age] for age in ages], dtype="float64")

    def fit(self, X: Triangle, y=None, sample_weight=None) -> LEV:
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
        self.means_ = self._resolve_means(X)
        return self

    def at(self, limit: float) -> Triangle:
        """
        The limited expected value at a single limit.

        Parameters
        ----------
        limit: float
            The limit at which claims are capped. ``np.inf`` returns the
            unlimited mean.

        Returns
        -------
        Triangle
            ``means_ * (1 - exp(-limit / means_))``, shaped like ``means_``.
        """
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}.")
        if np.isinf(limit):
            return self.means_.copy()
        if limit == 0:
            # Not just an optimisation. `0 / means_` is NaN rather than 0 in
            # Triangle arithmetic, and `1 - NaN` is then 1 rather than NaN, so
            # the general expression below would hand back the means untouched.
            return self.means_ * 0.0
        return self.means_ * (1 - np.exp(-limit / self.means_))

    def layer(self, attachment: float, exhaustion: float) -> Triangle:
        """
        The expected loss in the layer between two limits.

        This is ``LEV(p) - LEV(d)`` for the layer attaching at ``d`` and
        exhausting at ``p`` -- equation 2.4 of the paper.

        Parameters
        ----------
        attachment: float
            The limit at which the layer attaches.
        exhaustion: float
            The limit at which the layer exhausts. Use ``np.inf`` for an
            unlimited top layer.

        Returns
        -------
        Triangle
            The limited expected value of the layer.

        Raises
        ------
        ValueError
            If the layer exhausts at or below its attachment point.
        """
        if exhaustion <= attachment:
            raise ValueError(
                f"exhaustion ({exhaustion}) must exceed attachment ({attachment})."
            )
        return self.at(exhaustion) - self.at(attachment)


class Sahasrabuddhe(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Restate a Triangle to a common cost level and claim size limit.

    Claims development patterns are specific to the limit the data is reported
    at and to the cost level of each cell. Sahasrabuddhe (2010) shows that a
    Triangle can be moved onto a single basis before any development is fitted,
    using a claim size model and a cost level index::

        adjusted(i,j) = X(i,j) * LEV(target_limit; Phi(base,j))
                               / LEV(data_limit;   Phi(i,j))

    where ``Phi(i,j)`` is the claim size model restated to cell (i, j)'s cost
    level. The numerator is one row -- the basis everything is moved to -- and
    the denominator varies cell by cell, which is what makes the ratio a
    combined trend and limit adjustment.

    The transformed Triangle carries adjusted values, so ordinary development
    follows in a Pipeline: there is nothing left in the data for the factors to
    be contaminated by.

    Parameters
    ----------
    means: Triangle or dict
        The mean claim size by development age at a single cost level, as
        :class:`LEV` takes it. This is the output of the claim size analysis,
        stated at ``base_period``'s cost level.
    trend: Triangle
        A cost level index over the origin x development rectangle, stating
        each cell's cost level relative to a common base. Build it with
        :class:`Trend` -- ``full_triangle=True`` is required, since the index
        has to cover cells the Triangle itself does not reach.

        Only ratios of this index are used, so its own base period is
        immaterial: rescaling it by any constant leaves the result unchanged.
    data_limit: float
        The claim size limit the observed data is reported at.
    target_limit: float
        The claim size limit to restate to. ``np.inf`` restates to unlimited.
    base_period: int or str, optional
        The origin period whose cost level everything is restated to, and the
        period ``means`` is stated at. Defaults to the latest origin.

    Attributes
    ----------
    means_: Triangle
        The claim size model parameters restated to every cell's own cost
        level -- ``means`` spread over the rectangle by ``trend``.
    lev_: LEV
        The limited expected value model fitted on ``means_``. Its ``at`` and
        ``layer`` methods give the expectations behind the adjustment.
    adjusted_: Triangle
        The restated Triangle, shaped like the Triangle that was fit.

    """

    def __init__(
        self,
        means: Triangle | dict[int, float] | None = None,
        trend: Triangle | None = None,
        data_limit: float | None = None,
        target_limit: float | None = None,
        base_period: int | str | None = None,
    ):
        self.means = means
        self.trend = trend
        self.data_limit = data_limit
        self.target_limit = target_limit
        self.base_period = base_period

    def _base_index(self, X: Triangle) -> int:
        """
        The position of ``base_period`` on X's origin axis.

        Parameters
        ----------
        X: Triangle
            The Triangle being fit.

        Returns
        -------
        int
            The origin index to anchor on; the latest origin when
            ``base_period`` is None.

        Raises
        ------
        ValueError
            If ``base_period`` matches no origin period of X.
        """
        if self.base_period is None:
            return X.shape[-2] - 1
        period = pd.Period(str(self.base_period))
        lo, hi = period.to_timestamp(how="s"), period.to_timestamp(how="e")
        starts = X.origin.to_timestamp(how="s")
        matches = np.where((starts >= lo) & (starts <= hi))[0]
        if not len(matches):
            raise ValueError(
                f"base_period {self.base_period!r} does not match any origin "
                f"period. Origins run {X.origin[0]} through {X.origin[-1]}."
            )
        return int(matches[0])

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
        for name in ("data_limit", "target_limit"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required.")
        # Compare labels rather than shapes: a trend built against a different
        # Triangle of the same dimensions would otherwise pass silently and
        # restate every cell by the wrong factor.
        for axis in ("origin", "development"):
            mine = list(getattr(self.trend, axis))
            theirs = list(getattr(X, axis))
            if mine != theirs:
                raise ValueError(
                    f"trend does not share X's {axis} axis: trend runs "
                    f"{mine[0]} through {mine[-1]}, X runs {theirs[0]} through "
                    f"{theirs[-1]}. Build the trend against X, with "
                    "full_triangle=True so it covers the whole rectangle."
                )
        base = self._base_index(X)
        base_row = self._base_slice = slice(base, base + 1)

        # Everything is divided by the base period's row of the index, so a gap
        # anywhere in it propagates NaN across the whole result. A trend fitted
        # without full_triangle=True is shaped like the Triangle, which leaves
        # that row almost entirely empty -- and the failure is silent, so it is
        # worth catching here rather than letting it surface as missing data.
        base_factors = np.asarray(
            self.trend.iloc[..., base_row, :].set_backend("numpy").values
        )
        if np.isnan(base_factors).any():
            raise ValueError(
                f"trend is not defined across the whole {X.origin[base]} origin "
                "period, which every cell is restated against. Refit the trend "
                "with full_triangle=True so it covers cells past the valuation "
                "date."
            )

        # `means` describes the base period. Spreading it over the rectangle is
        # a ratio of the index, so the index's own base period cancels.
        means_at_base = LEV(means=self.means).fit(X).means_
        self.means_ = means_at_base * self.trend / self.trend.iloc[..., base_row, :]
        self.lev_ = LEV(means=self.means_).fit(X)

        target = self.lev_.at(self.target_limit).iloc[..., base_row, :]
        self.adjusted_ = X * target / self.lev_.at(self.data_limit)
        # The limited expected values cover the whole rectangle, so the product
        # inherits their valuation date rather than X's. The values are already
        # masked correctly -- it is the metadata that is wrong -- but leaving it
        # would give the result a nan_triangle and a latest diagonal belonging
        # to a Triangle that runs years past the data.
        self.adjusted_.valuation_date = X.valuation_date
        return self

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
        X_new = self.adjusted_.copy()
        X_new.means_ = self.means_
        X_new.lev_ = self.lev_
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

    def by_layer(
        self,
        development,
        attachment: float = 0.0,
        exhaustion: float | None = None,
        full_triangle: bool = False,
        style: Literal["ldf", "cdf"] | None = None,
    ) -> Triangle:
        """
        Restate a development pattern onto a different layer.

        A pattern fitted on one layer implies a pattern for any other, through
        the ratio of the two layers' expected losses -- no refitting and no
        second triangle. For a layer whose expected loss is ``L(i,j)``::

            factor(i,j) = cdf(j) * [ L(i,last) / A(last) ] / [ L(i,j) / A(j) ]

        where ``A`` is the expected loss of the layer the pattern was fitted
        on, taken at ``base_period``. ``L(i,last) / L(i,j)`` is how much of
        exposure period i's ultimate loss in the new layer has emerged by age
        j, and the ``A`` terms restate that onto the pattern's own basis.

        Parameters
        ----------
        development: estimator or Triangle
            The pattern to restate, fitted on the Triangle this estimator
            transformed. A fitted development estimator, a Triangle it has
            transformed, or a Triangle of cumulative factors.
        attachment: float (default=0.0)
            The limit at which the target layer attaches.
        exhaustion: float, optional
            The limit at which the target layer exhausts. Defaults to
            ``target_limit``, which reproduces the pattern's own layer.
            ``np.inf`` gives an unlimited top layer.
        full_triangle: bool (default=False)
            By default the factors are shaped like the Triangle that was fit.
            When True they cover the whole origin x development rectangle
            instead -- nothing in the calculation needs the data, so the cells
            past the latest diagonal are the factors a future valuation uses.
            Cannot be combined with ``style``.
        style: {'ldf', 'cdf'}, optional
            By default the full origin x development surface is returned. Set
            this to collapse it to the **latest diagonal** -- one factor per
            development age, taken from the exposure period that has actually
            reached that age -- shaped as an ordinary development pattern.

            These are the factors you would apply: each exposure period
            develops from its own current age at its own cost level, which is
            one cell per row along the last diagonal. Reading a single row of
            the surface instead would apply one period's cost level to them
            all.

            ``'cdf'`` labels them ``12-Ult``, ``24-Ult``, ...; ``'ldf'``
            converts to age-to-age, labelled ``12-24``, ``24-36``, ...

        Returns
        -------
        Triangle
            Cumulative development factors for the layer, over the origin x
            development surface -- or, when ``style`` is given, the latest
            diagonal as a development pattern.

        Notes
        -----
        Layers that have almost no expected loss at early ages produce very
        large or non-finite factors. That is the arithmetic being honest: the
        factor required to develop nothing to ultimate is undefined, and a high
        excess layer genuinely has nothing to develop at age one.
        """
        if style is not None:
            if style not in ("ldf", "cdf"):
                raise ValueError(f"style must be 'ldf', 'cdf' or None, got {style!r}.")
            if full_triangle:
                raise ValueError(
                    "style and full_triangle cannot be combined. The latest "
                    "diagonal is a property of the observed Triangle, so there "
                    "is no diagonal to take on the full rectangle."
                )
        exhaustion = self.target_limit if exhaustion is None else exhaustion
        shape = self.adjusted_
        ages = list(shape.development)

        cdf = self._cdf_by_age(development, ages)
        layer = self.lev_.layer(attachment, exhaustion)
        anchor = self.lev_.at(self.target_limit).iloc[..., self._base_slice, :]

        layer_values = np.asarray(layer.set_backend("numpy").values)
        anchor_values = np.asarray(anchor.set_backend("numpy").values)
        emerged = layer_values / anchor_values
        factors = cdf * (layer_values[..., -1:] / anchor_values[..., -1:]) / emerged

        out = shape.set_backend("numpy").copy()
        if style is not None:
            return self._as_pattern(out, factors, ages, style)
        if full_triangle:
            # Push the valuation date past the data so nan_triangle stops masking.
            out.valuation_date = out.valuation.max()
        else:
            factors = factors * out.nan_triangle
        out.values = factors
        return out.set_backend(shape.array_backend)

    @staticmethod
    def _as_pattern(shape: Triangle, factors, ages: list, style: str) -> Triangle:
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
                "omit `style` and take the factor surface instead."
            )
        rows = np.argmax(on_diagonal, axis=0)
        pattern = {
            age: float(factors[..., rows[position], position].flat[0])
            for position, age in enumerate(ages)
        }
        fitted = DevelopmentConstant(patterns=pattern, style="cdf").fit(shape)
        return fitted.cdf_ if style == "cdf" else fitted.ldf_
