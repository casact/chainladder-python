from chainladder.development import Development, DevelopmentBase
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
    def __init__(self, paid_to_incurred=None, paid_n_periods=-1,
                 case_n_periods=-1, groupby=None):
        self.paid_to_incurred = paid_to_incurred
        self.paid_n_periods = paid_n_periods
        self.case_n_periods = case_n_periods
        self.groupby = groupby

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
        backend = "cupy" if X.array_backend == "cupy" else "numpy"
        self.X_ = X.copy()
        self.paid_w_ = Development(n_periods=self.paid_n_periods).fit(self.X_.sum(0).sum(1)).w_
        self.case_w_ = Development(n_periods=self.case_n_periods).fit(self.X_.sum(0).sum(1)).w_
        self.case_ldf_ = self.case_to_prior_case_.mean(2)
        self.paid_ldf_ = self.paid_to_prior_case_.mean(2)
        self.ldf_ = self._set_ldf(self.X_).set_backend(backend)
        return self

    def _set_ldf(self, X):
        paid_tri = X[self.paid_to_incurred[0]]
        incurred_tri = X[self.paid_to_incurred[1]]
        case = incurred_tri-paid_tri
        original_val_date = case.valuation_date

        case_ldf_ = self.case_ldf_.copy()
        case_ldf_.valuation_date = pd.Timestamp(ULT_VAL)
        xp = case_ldf_.get_array_module()
        # Broadcast triangle shape
        case_ldf_ = case_ldf_ * case.latest_diagonal / case.latest_diagonal
        case_ldf_.odims = case.odims
        case_ldf_.is_pattern = False
        case_ldf_.values = xp.concatenate(
            (xp.ones(list(case_ldf_.shape[:-1])+[1]), case_ldf_.values),
            axis=-1)

        case_ldf_.ddims = case.ddims
        case_ldf_.valuation_date = case_ldf_.valuation.max()
        case_ldf_ = case_ldf_.dev_to_val().set_backend(self.case_ldf_.array_backend)

        # Will this work for sparse?
        forward = case_ldf_[case_ldf_.valuation>original_val_date].values
        forward[xp.isnan(forward)] = 1.0
        forward = xp.cumprod(forward, -1)
        1/case_ldf_[case_ldf_.valuation<=original_val_date]

        backward = 1/case_ldf_[case_ldf_.valuation<=original_val_date].values
        backward[xp.isnan(backward)] = 1.0
        backward = xp.cumprod(backward[..., ::-1], -1)[..., ::-1][..., 1:]
        nans = case_ldf_/case_ldf_
        case_ldf_.values = xp.concatenate((backward, (case.latest_diagonal*0+1).values,  forward), -1)
        case = (case_ldf_*nans.values*case.latest_diagonal.values).val_to_dev().iloc[..., :len(case.ddims)]
        ld = case[case.valuation==X.valuation_date].sum('development').sum('origin')
        ld = ld / ld
        patterns = ((1-np.nan_to_num(X.nan_triangle[..., 1:]))*(self.paid_ldf_*ld).values)
        paid = (case.iloc[..., :-1]*patterns)
        paid.ddims = case.ddims[1:]
        paid.valuation_date = pd.Timestamp(ULT_VAL)
        #Create a full triangle of incurrds to support a multiplicative LDF
        paid = (paid_tri.cum_to_incr() + paid).incr_to_cum()
        inc = (case[case.valuation>X.valuation_date] +
               paid[paid.valuation>X.valuation_date] +
               incurred_tri)
        # Combined paid and incurred into a single object
        paid.columns = [self.paid_to_incurred[0]]
        inc.columns = [self.paid_to_incurred[1]]
        cols = X.columns[X.columns.isin([self.paid_to_incurred[0], self.paid_to_incurred[1]])]
        dev = concat((paid, inc), 1)[list(cols)]
        # Convert the paid/incurred to multiplicative LDF
        dev = (dev.iloc[..., -1]/dev).iloc[..., :-1]
        dev.valuation_date = pd.Timestamp(ULT_VAL)
        dev.ddims = X.link_ratio.ddims
        dev.is_pattern=True
        dev.is_cumulative=True
        self.case = case
        self.paid=paid
        return dev.cum_to_incr()

    @property
    def case_to_prior_case_(self):
        paid_tri = self.X_[self.paid_to_incurred[0]]
        incurred_tri = self.X_[self.paid_to_incurred[1]]
        if self.groupby is not None:
            paid_tri = paid_tri.groupby(self.groupby).sum()
            incurred_tri = incurred_tri.groupby(self.groupby).sum()
        out = ((incurred_tri - paid_tri).iloc[..., 1:] * self.case_w_ /
               (incurred_tri - paid_tri).iloc[..., :-1].values)
        out.is_pattern = True
        out.is_cumulative=False
        return out

    @property
    def paid_to_prior_case_(self):
        paid_tri = self.X_[self.paid_to_incurred[0]]
        incurred_tri = self.X_[self.paid_to_incurred[1]]
        if self.groupby is not None:
            paid_tri = paid_tri.groupby(self.groupby).sum()
            incurred_tri = incurred_tri.groupby(self.groupby).sum()
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
        X_new.ldf_ = self._set_ldf(X_new)
        X_new._set_slicers()
        X_new.paid_ldf_ = self.paid_ldf_
        X_new.case_ldf_ = self.case_ldf_
        return X_new
