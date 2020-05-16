from scipy.stats import binom, rankdata
from chainladder.utils.cupy import cp
import copy

class DevelopmentCorrelation:
    """ Mack (1997) test for correlations between subsequent development
        factors. Results should be between -.67x and +.67x stdError
        otherwise too much correlation

        Returns
        -------
            development factors correlation: float
            development factors correlation variance: float
    """
    def __init__(self, triangle):
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
        T = 1-6*numerator/denominator
        weight = (I-k-1)[None, None, None]
        spearman_corr = (xp.sum(xp.nan_to_num(weight*T.values), axis=3) /
                         xp.sum(weight, axis=3))[..., None]
        spearman_corr_var = 2/((I-2)*(I-3))
        obj = copy.deepcopy(triangle)
        obj.values = spearman_corr
        obj.odims =['(All)']
        obj.ddims = ['Spearman Correlation']
        self.spearman_corr = obj
        self.spearman_corr_var = spearman_corr_var

class ValuationCorrelation:
    """
    Mack (1997) test for calendar year effect
    A calendar period has impact across developments if the probability of
    the number of small (or large) development factors in that period
    occurring randomly is less than p_critical

    Parameters
    ----------
    p_critical: float (default=0.10)
        Value between 0 and 1 representing the confidence level for the test

    Returns
    ----------
        Series of bool indicating whether that specific period shows
        statistically significant influence at `p_critical` confidence level
        on the development factors
    """
    def __init__(self, triangle, p_critical=.1):

        def pZlower(z,n,p):
            return min(1, 2*binom.cdf(z,n,p))

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
        probs = xp.array(
            [[[pZlower(z[i, c, 0, d], n[i, c, 0, d], 0.5) for d in range(S.shape[3])]
              for c in range(S.shape[1])] for i in range(S.shape[0])])[:,:,None,:]
        obj = triangle[triangle.valuation>triangle.valuation.min()].dev_to_val().dropna().sum('origin')*0
        obj.values = (xp.array(probs)<p_critical)
        obj.odims=['(All)']
        self.z_critical = obj
