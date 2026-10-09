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
    """
    Deterministic development from prior-lag case reserves.

    Estimates incremental paid amounts and case-reserve runoff as fractions of
    the prior lag's carried case reserve. Like
    :class:`~chainladder.MunichAdjustment` and
    :class:`~chainladder.BerquistSherman`, this is useful when case reserves
    should inform paid ultimates. A triangle with both paid and incurred columns
    is required.

    The incremental ``paid_ldf_`` patterns are not multiplicative link ratios;
    the estimator also builds origin-specific implied multiplicative ``ldf_``
    so standard IBNR methods can be applied.

    For Approach 2 (Friedland 2010, Chapter 12, Exhibit III) using benchmark
    reported and paid development patterns, see
    :class:`~chainladder.ImpliedCaseDevelopment`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    paid_to_incurred: tuple or list of tuples, optional
        A tuple representing the paid and incurred ``columns`` of the triangles
        such as ``('paid', 'incurred')``.
    paid_n_periods: integer, optional (default=-1)
        number of origin periods to be used in the paid pattern averages. For
        all origin periods, set paid_n_periods=-1
    case_n_periods: integer, optional (default=-1)
        number of origin periods to be used in the case pattern averages. For
        all origin periods, set case_n_periods=-1
    groupby: optional
        Grouping option.

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
        self,
        paid_to_incurred=None,
        paid_n_periods=-1,
        case_n_periods=-1,
        groupby=None,
        **kwargs,
    ):
        if (
            "reported_pattern" in kwargs
            or "paid_pattern" in kwargs
            or kwargs.get("approach") == 2
        ):
            raise ValueError(
                "Approach 2 has been decoupled into cl.ImpliedCaseDevelopment (or cl.ImpliedCase). "
                "Use cl.ImpliedCaseDevelopment(reported_pattern=..., paid_pattern=...) instead."
            )
        self.paid_to_incurred = paid_to_incurred
        self.paid_n_periods = paid_n_periods
        self.case_n_periods = case_n_periods
        self.groupby = groupby

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the model with X.

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
        """
        If X and self are of different shapes, align self to X, else
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


class ImpliedCaseDevelopment(DevelopmentBase):
    """
    Implied Case Development Estimator (Friedland Approach 2).

    An Estimator that derives implied case development factors (LDFs/CDFs)
    from benchmark reported and paid development patterns (Friedland 2010,
    Chapter 12, Exhibit III).

    This method is commonly used by self-insurers or for excess/reinsurance
    layers where credible historical triangles are unavailable, but carried
    case reserves and industry/benchmark development patterns are known.

    Parameters
    ----------
    reported_pattern : dict, Series, DataFrame, Triangle, or Development
        Benchmark reported loss development pattern. Can be cumulative (CDF)
        or incremental (LDF), as specified by ``style``.
    paid_pattern : dict, Series, DataFrame, Triangle, or Development
        Benchmark paid loss development pattern. Can be cumulative (CDF)
        or incremental (LDF), as specified by ``style``.
    style : {'cdf', 'ldf'}, optional (default='cdf')
        Whether ``reported_pattern`` and ``paid_pattern`` are cumulative development
        factors ('cdf') or incremental link ratios ('ldf').
    groupby : list, str, or callable, optional
        Grouping option for the triangle index.

    Attributes
    ----------
    ldf_ : Triangle
        Estimated loss development patterns for case outstanding.
    case_ldf_ : Triangle
        Alias for ``ldf_``.
    case_cdf_ : Triangle
        Implied cumulative case development factors to ultimate.
    case_ : Triangle
        The case triangle used during fitting.

    Examples
    --------
    .. testsetup::

        import chainladder as cl

    .. testcode::

        import pandas as pd
        import numpy as np
        import chainladder as cl

        case_data = pd.DataFrame({
            "origin": [1998, 1999, 2000, 2001, 2002, 2003],
            "valuation": 2008,
            "case": [500000, 650000, 800000, 850000, 975000, 1000000],
        })
        case_tri = cl.Triangle(
            case_data, origin="origin", valuation="valuation", columns="case", cumulative=True
        ).val_to_dev()

        rep_cdfs = {132: 1.015, 120: 1.020, 108: 1.030, 96: 1.051, 84: 1.077, 72: 1.131}
        paid_cdfs = {132: 1.046, 120: 1.067, 108: 1.109, 96: 1.187, 84: 1.306, 72: 1.489}

        model = cl.ImpliedCaseDevelopment(
            reported_pattern=rep_cdfs,
            paid_pattern=paid_cdfs,
            style="cdf",
        ).fit(case_tri)

        print(np.round(model.case_cdf_["case"].values[0, 0, 0, 5:], 3))

    .. testoutput::

        [1.545 1.439 1.445 1.421 1.454 1.506]

    """

    def __init__(
        self,
        reported_pattern=None,
        paid_pattern=None,
        style="cdf",
        groupby=None,
    ):
        self.reported_pattern = reported_pattern
        self.paid_pattern = paid_pattern
        self.style = style
        self.groupby = groupby

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the model with X.

        Parameters
        ----------
        X : Triangle
            Triangle to fit. Can be a 1-channel case outstanding triangle,
            or a 2-channel triangle with paid and incurred columns (from which
            case outstanding is computed as incurred - paid).
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        from chainladder.development.constant import DevelopmentConstant

        backend = "cupy" if X.array_backend == "cupy" else "numpy"
        self.X_ = X.copy()

        if self.reported_pattern is None or self.paid_pattern is None:
            raise ValueError(
                "ImpliedCaseDevelopment requires both reported_pattern and paid_pattern."
            )

        # 1. Determine case triangle
        if len(X.columns) > 1 and "incurred" in X.columns and "paid" in X.columns:
            case_tri = (X["incurred"] - X["paid"]).copy()
            case_tri.columns = ["case"]
        elif len(X.columns) > 1 and "Incurred" in X.columns and "Paid" in X.columns:
            case_tri = (X["Incurred"] - X["Paid"]).copy()
            case_tri.columns = ["case"]
        else:
            case_tri = X

        obj = self._set_fit_groups(case_tri).val_to_dev().copy()

        # 2. Extract series of CDFs
        rep_s = self._extract_cdf_series(self.reported_pattern, self.style)
        paid_s = self._extract_cdf_series(self.paid_pattern, self.style)

        # 3. Determine common development ages
        common_devs = sorted(set(rep_s.index).intersection(set(paid_s.index)))
        if not common_devs:
            raise ValueError(
                "reported_pattern and paid_pattern share no common development ages."
            )

        # 4. Implied Case CDF formula (Friedland Exhibit III):
        # Case_CDF = [((CDF_rep - 1.0) * CDF_paid) / (CDF_paid - CDF_rep)] + 1.0
        case_cdf_dict = {}
        for d in obj.ddims:
            if d in rep_s and d in paid_s:
                r = float(rep_s[d])
                p = float(paid_s[d])
                den = p - r
                if np.isclose(den, 0.0) or den < 0:
                    case_cdf_dict[d] = 1.0
                else:
                    case_cdf_dict[d] = ((r - 1.0) * p) / den + 1.0
            else:
                case_cdf_dict[d] = 1.0

        dev_const = DevelopmentConstant(patterns=case_cdf_dict, style="cdf").fit(obj)

        self.ldf_ = dev_const.ldf_.set_backend(backend)
        self.case_ldf_ = dev_const.ldf_.set_backend(backend)
        self.case_cdf_ = dev_const.cdf_.set_backend(backend)
        self.case_ = case_tri
        return self

    def transform(self, X):
        """
        Transform the triangle by attaching implied case development factors.

        If X is a multi-channel triangle with paid and incurred columns, the
        case triangle (incurred - paid) is extracted before attaching ldf_.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed.

        Returns
        -------
        X_new : Triangle
            Transformed case triangle with ldf_, case_ldf_, and case_cdf_ attached.
        """
        if len(X.columns) > 1 and "incurred" in X.columns and "paid" in X.columns:
            case_tri = (X["incurred"] - X["paid"]).copy()
            case_tri.columns = ["case"]
        elif len(X.columns) > 1 and "Incurred" in X.columns and "Paid" in X.columns:
            case_tri = (X["Incurred"] - X["Paid"]).copy()
            case_tri.columns = ["case"]
        else:
            case_tri = X

        X_new = case_tri.copy()
        X_new.group_index = self._set_transform_groups(X_new)
        X_new.ldf_ = self.ldf_
        X_new.case_ldf_ = self.case_ldf_
        X_new.case_cdf_ = self.case_cdf_
        X_new._set_slicers()
        return X_new

    @staticmethod
    def _extract_cdf_series(pattern, style="cdf"):
        """Extract a pd.Series of CDF factors indexed by integer development ages."""
        if hasattr(pattern, "cdf_"):
            pattern = pattern.cdf_
            style = "cdf"
        elif hasattr(pattern, "ldf_") and not hasattr(pattern, "cdf_"):
            pattern = pattern.ldf_
            style = "ldf"

        if isinstance(pattern, dict):
            s = pd.Series(pattern, dtype=float)
        elif isinstance(pattern, pd.Series):
            s = pattern.astype(float).copy()
        elif isinstance(pattern, pd.DataFrame):
            if pattern.shape[0] == 1:
                s = pattern.iloc[0].astype(float)
            elif pattern.shape[1] == 1:
                s = pattern.iloc[:, 0].astype(float)
            else:
                raise ValueError("DataFrame pattern must have 1 row or 1 column.")
        elif hasattr(pattern, "values") and hasattr(pattern, "ddims"):
            s = pd.Series(pattern.values[0, 0, 0, :], index=pattern.ddims, dtype=float)
        else:
            raise TypeError(f"Unsupported pattern type: {type(pattern)}")

        try:
            s.index = [int(str(x).split("-")[0]) for x in s.index]
        except (ValueError, TypeError):
            pass
        s = s.sort_index()

        if style == "ldf":
            s = s.iloc[::-1].cumprod().iloc[::-1]
        return s


ImpliedCase = ImpliedCaseDevelopment
