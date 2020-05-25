# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from scipy.stats import binom, norm, rankdata
from scipy.special import comb
from chainladder.utils.cupy import cp
import pandas as pd
import copy

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
    p_critical: float (default=0.10)
        Value between 0 and 1 representing the confidence level for the test. A
        value of 0.1 implies 90% confidence.

    Attributes
    ----------
    t_critical : DataFrame
        Boolean value for whether correlation is too high based on `p_critical`
        confidence level.
    t_expectation : DataFrame
        Values representing the Spearman rank correlation
    t_variance : float
        Variance measure of Spearman rank correlation
    range : tuple
        Range within which `t_expectation` must fall for independence assumption
        to be significant.
    """
    def __init__(self, triangle, p_critical=0.5):
        self.p_critical = p_critical
        xp = cp.get_array_module(triangle.values)
        m1 = triangle.link_ratio
        m1_val = xp.apply_along_axis(rankdata, 2, m1.values)*(m1.values*0+1)
        m2 = triangle[triangle.valuation<triangle.valuation_date].link_ratio
        m2.values = xp.apply_along_axis(rankdata, 2, m2.values)*(m2.values*0+1)
        m1 = copy.deepcopy(m2)
        m1.values = m1_val[..., :m2.shape[2], 1:]
        numerator = ((m1-m2)**2).sum('origin')
        numerator.values = numerator.values[..., :-1]
        numerator.ddims = numerator.ddims[:-1]
        I = triangle.shape[3]
        k = xp.array(range(2, 2+numerator.shape[3]))
        denominator = ((I - k)**3 - I + k)[None, None, None]
        self.t = 1-6*xp.nan_to_num(numerator.values)/denominator
        weight = (I-k-1)[None, None, None]
        t_expectation = (xp.sum(xp.nan_to_num(weight*self.t), axis=3) /
                         xp.sum(weight, axis=3))[..., None]
        idx = triangle._idx_table().index
        self.t_variance = 2/((I-2)*(I-3))
        self.t = pd.DataFrame(
            self.t[..., 0, 0], columns=triangle.vdims, index=idx)
        self.t_expectation = pd.DataFrame(
            t_expectation[..., 0, 0], columns=triangle.vdims, index=idx)
        self.range = (norm.ppf(0.5-(1-p_critical)/2)*xp.sqrt(self.t_variance),
                      norm.ppf(0.5+(1-p_critical)/2)*xp.sqrt(self.t_variance))
        self.t_critical = (self.t_expectation<self.range[0]) | \
                          (self.t_expectation>self.range[1])

class ValuationCorrelation:
    """
    Mack (1997) test for calendar year effect.A calendar period has impact
    across developments if the probability of the number of small (or large)
    development factors, Z, in that period occurring randomly is less than
    `p_critical`

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
        Boolean value for whether correlation is too high based on `p_critical`
        confidence level.
    z_expectation : Triangle or DataFrame
        The expected value of Z.
    z_variance : Triangle or DataFrame
        The variance value of Z.
    """
    def __init__(self, triangle, p_critical=.1, total=True):

        def pZlower(z,n,p=0.5):
            return min(1, 2*binom.cdf(z,n,p))

        self.p_critical = p_critical
        self.total = total
        xp = cp.get_array_module(triangle.values)
        lr = triangle.link_ratio
        m1 = xp.apply_along_axis(rankdata, 2, lr.values)*(lr.values*0+1)
        med = xp.nanmedian(m1, axis=2, keepdims=True)
        m1large = (xp.nan_to_num(m1) > med) + (lr.values*0)
        m1small = (xp.nan_to_num(m1) < med) + (lr.values*0)
        m2large = triangle.link_ratio
        m2large.values = m1large
        m2small = triangle.link_ratio
        m2small.values = m1small
        S = xp.nan_to_num(m2small.dev_to_val().sum(axis=2).values)
        L = xp.nan_to_num(m2large.dev_to_val().sum(axis=2).values)
        z = xp.minimum(L, S)
        n = L + S
        m = xp.floor((n - 1)/2)
        EZ = (n/2) - comb(n-1, m)*n/(2**n)
        VarZ = n*(n - 1) / 4 - comb(n-1, m)*n * (n-1) / (2**n) + EZ - EZ**2
        if not self.total:
            T=[]
            for i in range(0,xp.max(m1large.shape[2:])+1):
                T.append([pZlower(i,j,0.5) for j in range(0,xp.max(m1large.shape[2:])+1)])
            T=xp.array(T)
            self.probs = xp.array(T[z.astype(int),n.astype(int)])
            z_critical = triangle[triangle.valuation>triangle.valuation.min()]
            z_critical = z_critical.dev_to_val().dropna().sum('origin')*0
            z_critical.values = (xp.array(self.probs)<p_critical)
            z_critical.odims=['(All)']
            self.z_critical = z_critical
            self.z = copy.deepcopy(self.z_critical)
            self.z.values = z
            self.z_expectation = copy.deepcopy(self.z_critical)
            self.z_expectation.values = EZ
            self.z_variance = copy.deepcopy(self.z_critical)
            self.z_variance.values = VarZ
        else:
            ci2 = norm.ppf(0.5-(1-p_critical)/2)*xp.sqrt(xp.sum(VarZ, axis=-1))
            self.range = (xp.sum(VarZ, axis=-1) + ci2,
                          xp.sum(VarZ, axis=-1) - ci2)
            idx = triangle._idx_table().index
            self.z_critical = pd.DataFrame(
                ((self.range[0] > VarZ.sum(axis=-1)) | \
                (VarZ.sum(axis=-1) > self.range[1]))[..., 0],
                columns=triangle.vdims, index=idx)
            self.z =pd.DataFrame(
                z.sum(axis=-1)[..., 0], columns=triangle.vdims, index=idx)
            self.z_expectation =pd.DataFrame(
                EZ.sum(axis=-1)[..., 0], columns=triangle.vdims, index=idx)
            self.z_variance = pd.DataFrame(
                VarZ.sum(axis=-1)[..., 0], columns=triangle.vdims, index=idx)
