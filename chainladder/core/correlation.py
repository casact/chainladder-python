# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scipy.special import comb
from scipy.stats import binom, norm, rankdata

if TYPE_CHECKING:
    from chainladder.core.triangle import Triangle


class DevelopmentCorrelation:
    """
    Mack (1994) test for correlations between subsequent development
    factors. Results should be within confidence interval range
    otherwise too much correlation

    Parameters
    ----------
    triangle: Triangle
        Triangle on which to estimate correlation between subsequent development
        factors.
    p_critical: float (default=0.5)
        Value between 0 and 1 giving the significance level of the test. The
        confidence level is ``1 - p_critical``, so a value of 0.5 implies 50%
        confidence and a value of 0.1 implies 90% confidence. The default value
        follows Appendix G of Mack (1994), which proposes a 50% confidence
        interval because the test is only approximate and because the aim is
        to detect correlations already in a substantial part of the triangle.

    Attributes
    ----------
    t_critical: DataFrame
        Boolean value for whether correlation is too high at significance level
        ``p_critical``. False where ``t_expectation`` is NaN.
    t_expectation: DataFrame
        Values representing the Spearman rank correlation. NaN for a triangle
        with no column of at least two factor pairs.
    t_variance: DataFrame
        Variance measure of Spearman rank correlation, one value per triangle.
    confidence_interval: tuple of DataFrame
        Range around zero that contains ``t_expectation`` with probability of
        about ``1 - p_critical`` when the development factors are uncorrelated.
        ``t_critical`` is True when ``t_expectation`` falls outside it.

    References
    ----------
    Mack, T. (1994). Measuring the Variability of Chain Ladder Reserve
    Estimates. Casualty Actuarial Society Forum, Spring 1994, Appendix G.
    https://www.casact.org/sites/default/files/2021-02/pubs_forum_94spforum_94spf101.pdf

    Examples
    --------

    Mack (1994) shows that the chain-ladder assumptions imply that
    subsequent development factors are uncorrelated. Before relying on a
    ``Chainladder`` or ``MackChainladder`` ultimate it is good practice to
    test this on the triangle at hand.
    ``DevelopmentCorrelation`` performs Mack's weighted Spearman rank test
    across consecutive development columns and exposes both the test
    statistic ``t_expectation`` and the no-correlation
    ``confidence_interval``, so the decision is visible rather than reduced
    to a single boolean.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        raa = cl.load_sample('raa')
        dc = cl.DevelopmentCorrelation(raa, p_critical=0.5)
        print(round(float(dc.t_expectation.iloc[0, 0]), 4))
        print(round(float(dc.confidence_interval[0].iloc[0, 0]), 4))
        print(round(float(dc.confidence_interval[1].iloc[0, 0]), 4))
        print(bool(dc.t_critical.iloc[0, 0]))

    .. testoutput::

        0.0696
        -0.1275
        0.1275
        False

    The Spearman statistic ``0.0696`` lies inside the 50% confidence band
    ``(-0.1275, 0.1275)`` derived from ``t_variance = 2 / ((I - 2)(I - 3))``,
    so the test does not reject independence and chain-ladder is appropriate
    for RAA on this dimension. See the Mack chain-ladder section of the user
    guide for the full assumption set.

    ``p_critical`` sets how wide that acceptance band is. Raising it
    tightens the band, so the same Spearman statistic can flip
    ``t_critical`` even though the point estimate is unchanged:

    .. testcode::

        tight = cl.DevelopmentCorrelation(raa, p_critical=0.99)
        print(round(float(tight.confidence_interval[0].iloc[0, 0]), 4))
        print(round(float(tight.confidence_interval[1].iloc[0, 0]), 4))
        print(bool(tight.t_critical.iloc[0, 0]))

    .. testoutput::

        -0.0024
        0.0024
        True
    """

    def __init__(self, triangle, p_critical: float = 0.5):
        self.p_critical = p_critical

        # Check that critical value is a probability
        validate_critical(p_critical=p_critical)

        if triangle.array_backend != "numpy":
            triangle = triangle.set_backend("numpy")
        xp = triangle.get_array_module()

        link_ratio = triangle.link_ratio.values

        # T_k compares each factor F_ik of column k with F_i,k-1 of the same
        # origin. A complete triangle has I - k such pairs in column k. The
        # pairs present are counted per triangle, so a missing factor or an
        # origin without data lowers the count.
        current = link_ratio[..., 1:]
        previous = link_ratio[..., :-1]
        paired = ~xp.isnan(current) & ~xp.isnan(previous)
        n_pairs = paired.sum(axis=2)

        # Rank both columns among the paired factors only, 1 for the lowest,
        # and sum the squared rank differences (numerator of formula G4)
        squared_diff = xp.zeros(n_pairs.shape)
        if paired.size:
            ranks = [
                xp.apply_along_axis(
                    func1d=rankdata,
                    axis=2,
                    arr=xp.where(paired, column, xp.nan),
                    nan_policy="omit",
                )
                for column in (current, previous)
            ]
            ranks = [xp.where(paired, rank, xp.nan) for rank in ranks]
            squared_diff = xp.nansum((ranks[0] - ranks[1]) ** 2, axis=2)

        # Formula G4 for each column with at least two pairs, weighted by one
        # less than its number of pairs, as Mack proposes
        usable = n_pairs >= 2
        weight = xp.where(usable, n_pairs - 1, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_k = xp.where(
                usable, 1 - 6 * squared_diff / (n_pairs**3 - n_pairs), xp.nan
            )

        # Drop trailing columns that are unusable in every triangle, such as
        # the last column of a complete triangle, which has one pair
        has_weight = xp.any(weight > 0, axis=(0, 1))
        n_columns = int(xp.flatnonzero(has_weight)[-1]) + 1 if has_weight.any() else 0
        t_k, weight = t_k[..., :n_columns], weight[..., :n_columns]

        # k values are the column indexes for which we are calculating T_k
        k = xp.array(range(2, 2 + n_columns))

        # Big T is the weighted average of the T_k values. Its variance is one
        # over the total weight, which on a complete triangle is formula G6,
        # 2 / ((I - 2)(I - 3)). A triangle without a usable column gets NaN
        # for both.
        total_weight = xp.sum(weight, axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_expectation = xp.where(
                total_weight > 0,
                xp.sum(xp.nan_to_num(weight * t_k), axis=2) / total_weight,
                xp.nan,
            )
            t_variance = xp.where(total_weight > 0, 1 / total_weight, xp.nan)

        idx = triangle.index.set_index(triangle.key_labels).index

        # array of t values for the first triangle
        self.t = pd.DataFrame(t_k[0, 0][None], columns=k, index=["T_k"])

        # array of weights for the first triangle
        self.weights = pd.DataFrame(weight[0, 0][None], columns=k, index=["I-k-1"])

        # final big T
        self.t_expectation = pd.DataFrame(
            t_expectation, columns=triangle.columns, index=idx
        )

        self.t_variance = pd.DataFrame(t_variance, columns=triangle.columns, index=idx)

        # table of Spearman's rank coefficients Tk, can be used to verify consistency
        # with paper
        self.corr = pd.concat([self.t, self.weights])

        self.corr.columns.names = ["k"]

        # construct confidence interval based on selection of p_critical
        self.confidence_interval = (
            norm.ppf(0.5 - (1 - p_critical) / 2) * xp.sqrt(self.t_variance),
            norm.ppf(0.5 + (1 - p_critical) / 2) * xp.sqrt(self.t_variance),
        )

        # if T lies outside this range, we reject the null hypothesis; a NaN T
        # compares False, so a triangle without a usable column is not flagged
        self.t_critical = (self.t_expectation < self.confidence_interval[0]) | (
            self.t_expectation > self.confidence_interval[1]
        )

        # hypothesis test result, False means fail to reject the null hypothesis
        self.reject = self.t_critical.values[0][0]


class ValuationCorrelation:
    """
    Mack (1994) test for calendar year effect. A calendar period has impact
    across developments if the probability of the number of small (or large)
    development factors, Z, in that period occurring randomly is less than
    ``p_critical``

    Parameters
    ----------
    triangle: Triangle
        Triangle on which to test whether the calendar effects violate independence
        requirements of the chainladder method.
    p_critical: float (default=0.10)
        Value between 0 and 1 giving the significance level of the test. 0.1
        implies 90% confidence.
    total: boolean
        Whether to calculate valuation correlation in total across all
        years (True), as in Appendix H of Mack (1994), or for each year
        separately (False).

    Attributes
    ----------
    z : Triangle or DataFrame
        Z values for each Valuation Period
    z_critical : Triangle or DataFrame
        Boolean value for whether correlation is too high at significance level
        ``p_critical``.
    z_expectation : Triangle or DataFrame
        The expected value of Z.
    z_variance : Triangle or DataFrame
        The variance value of Z.
    range : tuple of arrays
        Lower and upper bounds of the no-effect band, E(Z) plus or minus
        ``norm.ppf(1 - p_critical / 2)`` standard deviations of Z. Only set
        when ``total=True``.

    References
    ----------
    Mack, T. (1994). Measuring the Variability of Chain Ladder Reserve
    Estimates. Casualty Actuarial Society Forum, Spring 1994, Appendix H.
    https://www.casact.org/sites/default/files/2021-02/pubs_forum_94spforum_94spf101.pdf

    Examples
    --------

    Mack's second prerequisite for the chain-ladder method is that no
    calendar period systematically inflates or deflates link ratios (for
    example from a one-off reserve strengthening or a change in case
    reserving practice). ``ValuationCorrelation`` flags any diagonal on
    which the split of high versus low link ratios is unlikely under random
    ordering.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        raa = cl.load_sample('raa')
        vc = cl.ValuationCorrelation(raa, p_critical=0.1, total=False)
        print(vc.z_critical)

    .. testoutput::

               1982   1983   1984   1985   1986   1987   1988   1989   1990
        1981  False  False  False  False  False  False  False  False  False

    No diagonal crosses the 90% threshold, so the calendar-effect assumption
    is supported. If any cell read ``True`` you would inspect that diagonal
    before relying on Mack or chain-ladder ultimates.

    The same test can be aggregated to a whole-triangle form
    (``total=True``, as in Appendix H of Mack (1994)) instead of the
    per-diagonal form (``total=False``) shown above. In total form the ``z``
    and ``z_critical`` attributes are returned as DataFrames rather than
    Triangles:

    .. testcode::

        vc_total = cl.ValuationCorrelation(raa, p_critical=0.1, total=True)
        print(round(float(vc_total.z.iloc[0, 0]), 4))
        print(bool(vc_total.z_critical.iloc[0, 0]))

    .. testoutput::

        14.0
        False

    The whole-triangle ``z`` statistic also falls inside the no-effect band,
    confirming the per-diagonal result.
    """

    def __init__(self, triangle: Triangle, p_critical: float = 0.1, total: bool = True):

        def p_z_lower(z: int, n: int, p: float = 0.5) -> float:
            return min(1, 2 * binom.cdf(z, n, p))

        self.p_critical = p_critical

        # Check that critical value is a probability
        validate_critical(p_critical=p_critical)

        self.total = total
        triangle = triangle.set_backend("numpy")
        xp = triangle.get_array_module()
        lr = triangle.link_ratio

        # Rank link ratios for each column
        m1 = xp.apply_along_axis(
            func1d=rankdata, axis=2, arr=lr.values, nan_policy="omit"
        ) * (lr.values * 0 + 1)

        med = xp.nanmedian(a=m1, axis=2, keepdims=True)

        m1large = (xp.nan_to_num(m1) > med) + (lr.values * 0)
        m1small = (xp.nan_to_num(m1) < med) + (lr.values * 0)
        m2large = triangle.link_ratio
        m2large.values = m1large
        m2small = triangle.link_ratio
        m2small.values = m1small
        S = xp.nan_to_num(m2small.dev_to_val().sum(axis=2).set_backend("numpy").values)
        L = xp.nan_to_num(m2large.dev_to_val().sum(axis=2).set_backend("numpy").values)
        z = xp.minimum(L, S)
        n = L + S
        m = xp.floor((n - 1) / 2)
        c = comb(n - 1, m)
        EZ = (n / 2) - c * n / (2**n)
        VarZ = n * (n - 1) / 4 - c * n * (n - 1) / (2**n) + EZ - EZ**2
        if not self.total:
            T = []
            for i in range(0, xp.max(m1large.shape[2:]) + 1):
                T.append([
                    p_z_lower(i, j, 0.5)
                    for j in range(0, xp.max(m1large.shape[2:]) + 1)
                ])
            T = np.array(T)
            z_idx, n_idx = z.astype(int), n.astype(int)
            self.probs = T[z_idx, n_idx]
            # One column per link-ratio diagonal, labeled by ending valuation.
            # Slicing by valuation (rather than dropna) keeps diagonals that are
            # entirely NaN, which a triangle missing its earliest diagonals has:
            # dropna() removed them while self.probs still carried one entry per
            # diagonal, so the two disagreed and the DataFrame below raised
            # "Shape of passed values is (1, 10), indices imply (1, 9)" (#320).
            z_critical = triangle.dev_to_val().sum("origin")
            z_critical = z_critical[z_critical.valuation > triangle.valuation.min()] * 0
            z_critical.values = np.array(self.probs) < p_critical
            z_critical.odims = triangle.odims[0:1]
            self.z_critical = z_critical
            self.z = self.z_critical.copy()
            self.z.values = z
            self.z_expectation = self.z_critical.copy()
            self.z_expectation.values = EZ
            self.z_variance = self.z_critical.copy()
            self.z_variance.values = VarZ
        else:
            # Appendix H of Mack, "Measuring the Variability of Chain Ladder
            # Reserve Estimates" (CAS Forum, 1994): Z is approximately normal, so
            # the test rejects "no calendar year effect" when Z falls outside
            # E(Z) +/- norm.ppf(1 - p_critical / 2) * sqrt(Var(Z)). ci2 is that
            # half-width with a negative sign, so range is (lower, upper).
            ci2 = norm.ppf(0.5 - (1 - p_critical) / 2) * xp.sqrt(xp.sum(VarZ, axis=-1))
            self.range = (xp.sum(EZ, axis=-1) + ci2, xp.sum(EZ, axis=-1) - ci2)
            idx = triangle.index.set_index(triangle.key_labels).index
            z_total = z.sum(axis=-1)
            self.z_critical = pd.DataFrame(
                ((self.range[0] > z_total) | (z_total > self.range[1]))[..., 0],
                columns=triangle.columns,
                index=idx,
            )
            self.z = pd.DataFrame(
                z.sum(axis=-1)[..., 0], columns=triangle.columns, index=idx
            )
            self.z_expectation = pd.DataFrame(
                EZ.sum(axis=-1)[..., 0], columns=triangle.columns, index=idx
            )
            self.z_variance = pd.DataFrame(
                VarZ.sum(axis=-1)[..., 0], columns=triangle.columns, index=idx
            )


def validate_critical(p_critical: float) -> None:
    """
    Checks whether value passed to the p_critical parameter in ValuationCorrelation or
    DevelopmentCorrelation classes is a percentage, that is, between 0 and 1.

    Parameters
    ----------
    p_critical: float
        Significance level used to test the null hypothesis in Mack correlation
        tests.
    """
    if 0 <= p_critical <= 1:
        pass
    else:
        raise ValueError("p_critical must be between 0 and 1.")
