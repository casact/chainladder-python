# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development import Development, DevelopmentBase
from chainladder import options
from chainladder.utils.utility_functions import concat
from chainladder.utils.utility_functions import num_to_nan

import numpy as np
import pandas as pd


class CaseOutstanding(DevelopmentBase):
    """ Deterministic development from prior-lag case reserves.

    Estimates incremental paid amounts and case-reserve runoff as fractions of
    the prior lag's carried case reserve. Like
    :class:`~chainladder.MunichAdjustment` and
    :class:`~chainladder.BerquistSherman`, this is useful when case reserves
    should inform paid ultimates. A triangle with both paid and incurred columns
    is required.

    The incremental ``paid_ldf_`` patterns are not multiplicative link ratios;
    the estimator also builds origin-specific implied multiplicative ``ldf_``
    so standard IBNR methods can be applied.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    paid_to_incurred: tuple or list of tuples
        A tuple representing the paid and incurred ``columns`` of the triangles
        such as ``('paid', 'incurred')``
    paid_n_periods: integer, optional (default=-1)
        number of origin periods to be used in the paid pattern averages. For
        all origin periods, set paid_n_periods=-1
    case_n_periods: integer, optional (default=-1)
        number of origin periods to be used in the case pattern averages. For
        all origin periods, set case_n_periods=-1

    Attributes
    ----------
    ldf_: Triangle
        Implied multiplicative loss development patterns (by paid/incurred
        column); each origin period has its own pattern.
    cdf_: Triangle
        The estimated (multiplicative) cumulative development patterns.
    case_to_prior_case_: Triangle
        Case-to-prior-case incremental ratios by origin (for review).
    case_ldf_: Triangle
        Selected case-to-prior-case ratios averaged across origins.
    paid_to_prior_case_: Triangle
        Paid-to-prior-case incremental ratios by origin (for review).
    paid_ldf_: Triangle
        Selected paid-to-prior-case ratios averaged across origins.

    Examples
    --------
    On ``usauto``, incremental paid in 12–24 is about 84% of case outstanding
    at lag 12 (first entry in ``paid_ldf_`` at development 24–36):

    .. testsetup::

        import chainladder as cl

    .. testcode::

        import numpy as np

        tri = cl.load_sample("usauto")
        model = cl.CaseOutstanding(
            paid_to_incurred=("paid", "incurred")
        ).fit(tri)
        print(np.round(model.paid_ldf_.values[0, 0, 0, :4], 4))

    .. testoutput::

        [0.8428 0.71   0.7084 0.6968]

    Implied multiplicative ``ldf_`` differ by accident year; the 1998 origin
    paid pattern is shown below (compare to volume-weighted chainladder).

    .. testcode::

        import numpy as np

        tri = cl.load_sample("usauto")
        model = cl.CaseOutstanding(
            paid_to_incurred=("paid", "incurred")
        ).fit(tri)
        print(np.round(model.ldf_["paid"].values[0, 0, 0, :4], 4))

    .. testoutput::

        [1.7925 1.2056 1.0956 1.0457]

    Review origin-level ``paid_to_prior_case_`` and ``case_to_prior_case_``
    when tuning ``paid_n_periods`` and ``case_n_periods``; fitted selections
    appear in ``paid_ldf_`` and ``case_ldf_``.

    .. testcode::

        import numpy as np

        tri = cl.load_sample("usauto")
        model = cl.CaseOutstanding(
            paid_to_incurred=("paid", "incurred")
        ).fit(tri)
        print(np.round(model.case_to_prior_case_.values[0, 0, 0, :4], 4))
        print(np.round(model.case_ldf_.values[0, 0, 0, :4], 4))

    .. testoutput::

        [0.5378 0.5541 0.5253 0.4981]
        [0.534  0.5638 0.5296 0.49  ]

    """

    def __init__(
        self, paid_to_incurred=None, paid_n_periods=-1, case_n_periods=-1, groupby=None
    ):
        self.paid_to_incurred = paid_to_incurred
        self.paid_n_periods = paid_n_periods
        self.case_n_periods = case_n_periods
        self.groupby = groupby

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle
            Triangle with paid and incurred columns for ``paid_to_incurred``.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        backend = "cupy" if X.array_backend == "cupy" else "numpy"
        self.X_ = X.copy()

        self.paid_w_ = (
            Development(n_periods=self.paid_n_periods).fit(self.X_.sum(0).sum(1)).w_
        )
        self.case_w_ = (
            Development(n_periods=self.case_n_periods).fit(self.X_.sum(0).sum(1)).w_
        )

        case_to_prior_case_naned = self.case_to_prior_case_
        case_to_prior_case_naned.values = num_to_nan(self.case_to_prior_case_.values)

        self.case_ldf_ = self.case_to_prior_case_.mean(2)  # this has the wrong value
        self.case_ldf_.values = case_to_prior_case_naned.mean(axis=2).values

        paid_to_prior_case_naned = self.paid_to_prior_case_
        paid_to_prior_case_naned.values = num_to_nan(self.paid_to_prior_case_.values)

        self.paid_ldf_ = self.paid_to_prior_case_.mean(2)  # this has the wrong value
        self.paid_ldf_.values = paid_to_prior_case_naned.mean(axis=2).values

        self.ldf_ = self._set_ldf(self.X_).set_backend(backend)

        return self

    def _set_ldf(self, X):
        paid_tri = X[self.paid_to_incurred[0]]
        incurred_tri = X[self.paid_to_incurred[1]]
        case_tri = incurred_tri - paid_tri

        original_val_date = case_tri.valuation_date

        case_ldf_ = self.case_ldf_.copy()
        case_ldf_.valuation_date = pd.Timestamp(options.ULT_VAL)
        xp = case_ldf_.get_array_module()
        # Broadcast triangle shape
        case_ldf_ = case_ldf_ * case_tri.latest_diagonal / case_tri.latest_diagonal
        case_ldf_.odims = case_tri.odims
        case_ldf_.is_pattern = False
        case_ldf_.values = xp.concatenate(
            (xp.ones(list(case_ldf_.shape[:-1]) + [1]), case_ldf_.values), axis=-1
        )

        case_ldf_.ddims = case_tri.ddims
        case_ldf_.valuation_date = case_ldf_.valuation.max()
        case_ldf_ = case_ldf_.dev_to_val().set_backend(self.case_ldf_.array_backend)

        # Will this work for sparse?
        forward = case_ldf_[case_ldf_.valuation > original_val_date].values
        forward[xp.isnan(forward)] = 1.0
        forward = xp.cumprod(forward, -1)
        1 / case_ldf_[case_ldf_.valuation <= original_val_date]

        backward = 1 / case_ldf_[case_ldf_.valuation <= original_val_date].values
        backward[xp.isnan(backward)] = 1.0
        backward = xp.cumprod(backward[..., ::-1], -1)[..., ::-1][..., 1:]
        nans = case_ldf_ / case_ldf_
        case_ldf_.values = xp.concatenate(
            (backward, (case_tri.latest_diagonal * 0 + 1).values, forward), -1
        )
        case_tri = (
            (case_ldf_ * nans.values * case_tri.latest_diagonal.values)
            .val_to_dev()
            .iloc[..., : len(case_tri.ddims)]
        )
        ld = (
            case_tri[case_tri.valuation == X.valuation_date]
            .sum("development")
            .sum("origin")
        )
        ld = ld / ld
        patterns = (1 - np.nan_to_num(X.nan_triangle[..., 1:])) * (
            self.paid_ldf_ * ld
        ).values
        paid = case_tri.iloc[..., :-1] * patterns
        paid.ddims = case_tri.ddims[1:]
        paid.valuation_date = pd.Timestamp(options.ULT_VAL)
        # Create a full triangle of incurrds to support a multiplicative LDF
        paid = (paid_tri.cum_to_incr() + paid).incr_to_cum()
        inc = (
            case_tri[case_tri.valuation > X.valuation_date]
            + paid[paid.valuation > X.valuation_date]
            + incurred_tri
        )
        # Combined paid and incurred into a single object
        paid.columns = [self.paid_to_incurred[0]]
        inc.columns = [self.paid_to_incurred[1]]
        cols = X.columns[
            X.columns.isin([self.paid_to_incurred[0], self.paid_to_incurred[1]])
        ]

        dev = concat((paid, inc), 1)[list(cols)]
        # Convert the paid/incurred to multiplicative LDF
        dev = (dev.iloc[..., -1] / dev).iloc[..., :-1]
        dev.valuation_date = pd.Timestamp(options.ULT_VAL)
        dev.ddims = X.link_ratio.ddims
        dev.is_pattern = True
        dev.is_cumulative = True

        self.case = case_tri
        self.paid = paid
        return dev.cum_to_incr()

    @property
    def case_to_prior_case_(self):
        paid_tri = self.X_[self.paid_to_incurred[0]]
        incurred_tri = self.X_[self.paid_to_incurred[1]]

        if self.groupby is not None:
            paid_tri = paid_tri.groupby(self.groupby).sum()
            incurred_tri = incurred_tri.groupby(self.groupby).sum()

        out = (
            (incurred_tri - paid_tri).iloc[..., 1:]
            * self.case_w_
            / (incurred_tri - paid_tri).iloc[..., :-1].values
        )
        out.is_pattern = True
        out.is_cumulative = False

        return out

    @property
    def paid_to_prior_case_(self):
        paid_tri = self.X_[self.paid_to_incurred[0]]
        incurred_tri = self.X_[self.paid_to_incurred[1]]

        if self.groupby is not None:
            paid_tri = paid_tri.groupby(self.groupby).sum()
            incurred_tri = incurred_tri.groupby(self.groupby).sum()

        out = (
            paid_tri.cum_to_incr().iloc[..., 1:]
            * self.paid_w_
            / (incurred_tri - paid_tri).iloc[..., :-1].values
        )
        out.is_pattern = True

        return out

    def transform(self, X):
        """If X and self are of different shapes, align self to X, else
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
        X_new.ldf_ = self._set_ldf(X_new).set_backend(self.ldf_.array_backend)
        X_new._set_slicers()
        X_new.paid_ldf_ = self.paid_ldf_
        X_new.case_ldf_ = self.case_ldf_
        return X_new
