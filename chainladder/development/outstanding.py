from chainladder.development.base import Development, DevelopmentBase
from chainladder import ULT_VAL
from chainladder.utils.utility_functions import concat

import numpy as np
import pandas as pd


class CaseOutstanding(DevelopmentBase):
    """ A determinisic method based on outstanding case reserves.

    The CaseOutstanding method is a deterministic approach that develops
    patterns of incremental payments as a percent of previous period case
    reserves as well as patterns for case reserves as a percent of previous
    period case reserves.  Although the patterns produces by the approach
    approximate incremental payments and case outstanding, they are converted
    into comparable multiplicative patterns for usage with the various IBNR
    methods.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    paid_to_incurred : tuple or list of tuples
        A tuple representing the paid and incurred ``columns`` of the triangles
        such as ``('paid', 'incurred')``
    paid_n_periods : integer, optional (default=-1)
        number of origin periods to be used in the paid pattern averages. For
        all origin periods, set paid_n_periods=-1
    case_n_periods : integer, optional (default=-1)
        number of origin periods to be used in the case pattern averages. For
        all origin periods, set paid_n_periods=-1

    Attributes
    ----------
    ldf_ : Triangle
        The estimated (multiplicative) loss development patterns.
    cdf_ : Triangle
        The estimated (multiplicative) cumulative development patterns.
    case_to_prior_case_ : Triangle
        The case to prior case ratios used for fitting the estimator
    case_ldf_ :
        The selected case to prior case ratios of the fitted estimator
    paid_to_prior_case_ : Triangle
        The paid to prior case ratios used for fitting the estimator
    paid_ldf_ :
        The selected paid to prior case ratios of the fitted estimator
    """
    def __init__(self, paid_to_incurred=None, paid_n_periods=-1, case_n_periods=-1):
        self.paid_to_incurred = paid_to_incurred
        self.paid_n_periods = paid_n_periods
        self.case_n_periods = case_n_periods

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle
            Set of LDFs to which the munich adjustment will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X.copy()
        paid_tri = self.X_[self.paid_to_incurred[0]]
        incurred_tri = self.X_[self.paid_to_incurred[1]]
        self.paid_w_ = Development(n_periods=self.paid_n_periods).fit(self.X_.iloc[0,0]).w_
        self.case_w_ = Development(n_periods=self.case_n_periods).fit(self.X_.iloc[0,0]).w_
        self.case_ldf_ = self.case_to_prior_case_.mean(2)
        self.paid_ldf_ = self.paid_to_prior_case_.mean(2)

        case = incurred_tri-paid_tri
        patterns = ((1 - np.nan_to_num(case.nan_triangle[..., 1:])) *
                    self.case_ldf_.values)
        for i in range(np.isnan(case.nan_triangle[-1]).sum()):
            increment = (
                (case - case[case.valuation < case.valuation_date]).iloc[..., :-1] *
                patterns)
            increment.ddims = case.ddims[1:]
            increment.valuation_date = case.valuation[case.valuation>=case.valuation_date].drop_duplicates()[1]
            case = case + increment

        patterns = ((1-np.nan_to_num(self.X_.nan_triangle[..., 1:]))*self.paid_ldf_.values)

        paid = (case.iloc[..., :-1]*patterns)
        paid.ddims = case.ddims[1:]
        paid.valuation_date = pd.Timestamp(ULT_VAL)
        paid = (paid_tri.cum_to_incr() + paid).incr_to_cum()
        inc = (case[case.valuation>self.X_.valuation_date] +
               paid[paid.valuation>self.X_.valuation_date] +
               incurred_tri)
        paid.columns = [self.paid_to_incurred[0]]
        inc.columns = [self.paid_to_incurred[1]]
        cols = self.X_.columns[self.X_.columns.isin([self.paid_to_incurred[0], self.paid_to_incurred[1]])]
        dev = concat((paid, inc), 1)[list(cols)]
        self.dev_ = dev
        dev = (dev.iloc[..., -1]/dev).iloc[..., :-1]
        dev.valuation_date = pd.Timestamp(ULT_VAL)
        dev.ddims = self.X_.link_ratio.ddims
        dev.is_pattern=True
        dev.is_cumulative=True
        self.ldf_ = dev.cum_to_incr()
        return self

    @property
    def case_to_prior_case_(self):
        paid_tri = self.X_[self.paid_to_incurred[0]]
        incurred_tri = self.X_[self.paid_to_incurred[1]]
        out = ((incurred_tri - paid_tri).iloc[..., 1:] * self.case_w_ /
               (incurred_tri - paid_tri).iloc[..., :-1].values)
        out.is_pattern = True
        out.is_cumulative=False
        return out

    @property
    def paid_to_prior_case_(self):
        paid_tri = self.X_[self.paid_to_incurred[0]]
        incurred_tri = self.X_[self.paid_to_incurred[1]]
        out = (
            paid_tri.cum_to_incr().iloc[..., 1:] * self.paid_w_ /
            (incurred_tri - paid_tri).iloc[..., :-1].values)
        out.is_pattern=True
        return out

    def transform(self, X):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = X.copy()
        triangles = ["ldf_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new
