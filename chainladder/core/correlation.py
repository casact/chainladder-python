# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.special import comb

from scipy.stats import binom, norm, rankdata

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core.triangle import Triangle


class DevelopmentCorrelation:
    """
    Mack (1997) test for correlations between subsequent development
    factors. Results should be within confidence interval range
    otherwise too much correlation

    Parameters
    ----------
    triangle: Triangle
        Triangle on which to estimate correlation between subsequent development
        factors.
    p_critical: float (default=0.5)
        Value between 0 and 1 representing the confidence level for the test. A
        value of 0.5 implies a 50% confidence. The default value is based on the example
        provided in the Mack 97 paper, the selection of which is justified on the basis of the
        test being only an approximate measure of correlations and the desire to detect
        correlations already in a substantial part of the triangle.

    Attributes
    ----------
    t_critical: DataFrame
        Boolean value for whether correlation is too high based on ``p_critical``
        confidence level.
    t_expectation: DataFrame
        Values representing the Spearman rank correlation
    t_variance: float
        Variance measure of Spearman rank correlation
    confidence_interval: tuple
        Range within which ``t_expectation`` must fall for independence assumption
        to be significant.
    """

    def __init__(self, triangle, p_critical: float = 0.5):
        self.p_critical = p_critical

        # Check that critical value is a probability
        validate_critical(p_critical=p_critical)

        if triangle.array_backend != "numpy":
            triangle = triangle.set_backend("numpy")
        xp = triangle.get_array_module()

        m1 = triangle.link_ratio

        # Rank link ratios by development period, assigning a score of 1 for the lowest
        m1_val = xp.apply_along_axis(
            func1d=rankdata, axis=2, arr=m1.values, nan_policy="omit"
        ) * (m1.values * 0 + 1)

        # Remove the last element from each column, and then rank again
        m2 = triangle[triangle.valuation < triangle.valuation_date].link_ratio
        m2.values = xp.apply_along_axis(
            func1d=rankdata, axis=2, arr=m2.values, nan_policy="omit"
        ) * (m2.values * 0 + 1)

        m1 = m2.copy()

        # remove the first column from m1 since it is not used in the comparison to m2
        m1.values = m1_val[..., : m2.shape[2], 1:]

        # Apply Spearman Rank Correlation formula
        # numerator is the one in formula G4 of the Mack 97 paper
        numerator = ((m1 - m2) ** 2).sum("origin")

        # remove last column because it was not part of the comparison with m2
        numerator.values = numerator.values[..., :-1]
        numerator.ddims = numerator.ddims[:-1]

        # I is the number of development periods in the triangle
        I = len(triangle.development)

        # k values are the column indexes for which we are calculating T_k
        k = xp.array(range(2, 2 + numerator.shape[3]))

        # denominator is the one in formula G4 of the Mack 97 paper
        denominator = ((I - k) ** 3 - I + k)[None, None, None]

        # complete formula G4, results in array of each T_k value
        self.t = 1 - 6 * xp.nan_to_num(numerator.values) / denominator

        # per Mack, weight is one less than the number of pairs for each T_k
        weight = (I - k - 1)[None, None, None]

        # Calculate big T, the weighted average of the T_k values
        t_expectation = (
            xp.sum(xp.nan_to_num(weight * self.t), axis=3) / xp.sum(weight, axis=3)
        )[..., None]

        idx = triangle.index.set_index(triangle.key_labels).index

        # variance is result of formula G6
        self.t_variance = 2 / ((I - 2) * (I - 3))

        # array of t values
        self.t = pd.DataFrame(self.t[0, 0, ...], columns=k, index=["T_k"])

        # array of weights
        self.weights = pd.DataFrame(weight[0, 0, ...], columns=k, index=["I-k-1"])

        # final big T
        self.t_expectation = pd.DataFrame(
            t_expectation[..., 0, 0], columns=triangle.vdims, index=idx
        )

        # table of Spearman's rank coefficients Tk, can be used to verify consistency with paper
        self.corr = pd.concat([self.t, self.weights])

        self.corr.columns.names = ["k"]

        # construct confidence interval based on selection of p_critical
        self.confidence_interval = (
            norm.ppf(0.5 - (1 - p_critical) / 2) * xp.sqrt(self.t_variance),
            norm.ppf(0.5 + (1 - p_critical) / 2) * xp.sqrt(self.t_variance),
        )

        # if T lies outside this range, we reject the null hypothesis
        self.t_critical = (self.t_expectation < self.confidence_interval[0]) | (
            self.t_expectation > self.confidence_interval[1]
        )

        # hypothesis test result, False means fail to reject the null hypothesis
        self.reject = self.t_critical.values[0][0]


class ValuationCorrelation:
    """
    Mack (1997) test for calendar year effect.A calendar period has impact
    across developments if the probability of the number of small (or large)
    development factors, Z, in that period occurring randomly is less than
    ``p_critical``

    Parameters
    ----------
    triangle: Triangle
        Triangle on which to test whether the calendar effects violate independence
        requirements of the chainladder method.
    p_critical: float (default=0.10)
        Value between 0 and 1 representing the confidence level for the test. 0.1
        implies 90% confidence.
    total: boolean
        Whether to calculate valuation correlation in total across all
        years (True) consistent with Mack 1993 or for each year separately
        (False) consistent with Mack 1997.

    Attributes
    ----------
    z : Triangle or DataFrame
        Z values for each Valuation Period
    z_critical : Triangle or DataFrame
        Boolean value for whether correlation is too high based on ``p_critical``
        confidence level.
    z_expectation : Triangle or DataFrame
        The expected value of Z.
    z_variance : Triangle or DataFrame
        The variance value of Z.
    """

    def __init__(self, triangle: Triangle, p_critical: float = 0.1, total: bool = True):

        def pZlower(z: int, n: int, p: float = 0.5) -> float:
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
                T.append(
                    [
                        pZlower(i, j, 0.5)
                        for j in range(0, xp.max(m1large.shape[2:]) + 1)
                    ]
                )
            T = np.array(T)
            z_idx, n_idx = z.astype(int), n.astype(int)
            self.probs = T[z_idx, n_idx]
            z_critical = triangle[triangle.valuation > triangle.valuation.min()]
            # z_critical = z_critical[z_critical.development > z_critical.development.min()].dev_to_val().sum(
            #     "origin") * 0
            z_critical = z_critical.dev_to_val().dropna().sum("origin") * 0
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
            ci2 = norm.ppf(0.5 - (1 - p_critical) / 2) * xp.sqrt(xp.sum(VarZ, axis=-1))
            self.range = (xp.sum(VarZ, axis=-1) + ci2, xp.sum(VarZ, axis=-1) - ci2)
            idx = triangle.index.set_index(triangle.key_labels).index
            self.z_critical = pd.DataFrame(
                (
                    (self.range[0] > VarZ.sum(axis=-1))
                    | (VarZ.sum(axis=-1) > self.range[1])
                )[..., 0],
                columns=triangle.vdims,
                index=idx,
            )
            self.z = pd.DataFrame(
                z.sum(axis=-1)[..., 0], columns=triangle.vdims, index=idx
            )
            self.z_expectation = pd.DataFrame(
                EZ.sum(axis=-1)[..., 0], columns=triangle.vdims, index=idx
            )
            self.z_variance = pd.DataFrame(
                VarZ.sum(axis=-1)[..., 0], columns=triangle.vdims, index=idx
            )


def validate_critical(p_critical: float) -> None:
    """
    Checks whether value passed to the p_critical parameter in ValuationCorrelation or DevelopmentCorrelation
    classes is a percentage, that is, between 0 and 1.

    Parameters
    ----------
    p_critical: float
        Critical value used to test null hypothesis in Mack correlation tests.
    """
    if 0 <= p_critical <= 1:
        pass
    else:
        raise ValueError("p_critical must be between 0 and 1.")
