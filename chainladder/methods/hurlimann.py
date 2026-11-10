"""
The Hürlimann credible loss ratio reserving model
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from chainladder.methods import MethodBase

if TYPE_CHECKING:
    from chainladder.core.triangle import Triangle
    from chainladder.core.typing import TriangleProtocol


class HurlimannCLR(MethodBase):
    """
    The Hürlimann credible loss ratio reserving model

    Implement the credible loss ratio claims reserves of Hürlimann
    :cite:`hurlimann2009`. The reserve for each origin period is a credibility
    mixture of an individual reserve, which grosses up the latest claims
    by a loss ratio payout factor, and a collective reserve, which applies
    that payout factor to a burning cost ultimate (the loss ratio method of
    Mack). The Benktander and Neuhaus methods and the optimal credible reserve
    are special cases that differ only in the credibility weight.

    Unlike the chain-ladder based methods, the payout pattern is estimated
    from incremental claims and exposure (supplied as ``sample_weight``) rather
    than from link ratios.

    Parameters
    ----------
    method: {"optimal", "benktander", "neuhaus", "collective", "individual"}, optional (default="optimal")
        The credibility weight ``Z`` placed on the individual reserve:

        - ``"optimal"``: :math:`Z = p / (p + \\sqrt{p})`, the weight that
          minimizes the mean squared error (Hürlimann, Theorem 6.1 with
          :math:`f = 1`).
        - ``"benktander"``: :math:`Z = p`.
        - ``"neuhaus"``: :math:`Z = p \\cdot ELR`, the cumulative loss ratio
          pattern to date. Unlike the others this can exceed 1 when the
          burning cost loss ratio is above 1.
        - ``"collective"``: :math:`Z = 0`.
        - ``"individual"``: :math:`Z = 1`.

    Attributes
    ----------
    m_: Triangle
        The loss ratio pattern :math:`\\hat m_k`, the incremental claims
        divided by the exposure of the origin periods observed at each
        development period (eq. 2.1).
    elr_: Triangle
        The burning cost loss ratio :math:`\\sum_k \\hat m_k`.
    p_: Triangle
        The loss ratio payout factor of each origin period, the proportion of
        its ultimate expected to have emerged to date (eq. 2.3).
    Z_: Triangle
        The credibility weight ``Z`` on the individual reserve.
    r_individual_: Triangle
        The individual reserve :math:`q C / p` (eq. 2.6).
    r_collective_: Triangle
        The collective reserve :math:`q \\cdot ELR \\cdot V` (eq. 2.7).
    mse_: Triangle
        The mean squared error of the reserve, in units of the common factor
        :math:`E[a_i^2(U_i)]` (eq. 4.11 with :math:`f = 1` and
        :math:`t = \\sqrt{p}`). It is only meaningful as a ratio between
        methods, such as to the ``"optimal"`` method's, which is minimal.
    ldf_: Triangle
        The age to age factors implied by the loss ratio pattern,
        :math:`\\sum_{j \\le k+1} \\hat m_j / \\sum_{j \\le k} \\hat m_j`. They do
        not depend on ``method``.
    cdf_: Triangle
        The cumulative development factors, the inverse of the payout pattern
        :math:`\\sum_{j \\le k} \\hat m_j / ELR`. The ``"individual"`` method is
        the chainladder method with these factors, so its ``ultimate_`` equals the
        latest claims times ``cdf_`` at each origin's age.
    ultimate_: Triangle
        The ultimate losses per the method
    ibnr_: Triangle
        The credible reserve, which is ``ultimate_`` minus the latest claims.
        What it represents depends on the claims in ``X``: for a reported or
        incurred triangle it is the IBNR, and for a paid triangle it is the
        unpaid claims reserve, which includes case reserves.

    See Also
    --------
    Benktander : Iterative Bornhuetter-Ferguson method.
    CapeCod : Estimates apriori loss ratios from exposure.

    Notes
    -----
    The paper assumes no development beyond the last development period of
    the triangle, so there is no tail. This is a strong assumption for paid
    claims. Negative incremental claims, such as released case reserves, are
    included in the loss ratio pattern as they are, so they can push the payout
    factors above one.

    References
    ----------
    :cite:`hurlimann2009`

    Examples
    --------
    Like :class:`Benktander` and :class:`CapeCod`, the model needs an exposure
    for each origin period, passed as ``sample_weight``. The IBNR is a
    credibility mixture of the individual and collective reserves, here with
    the optimal weight.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        xyz = cl.load_sample("xyz")
        model = cl.HurlimannCLR().fit(
            xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal
        )
        print(model.ibnr_)

    .. testoutput::

                      2261
        1998           NaN
        1999     92.109615
        2000    749.172531
        2001   1993.678395
        2002   4965.526581
        2003  10852.366005
        2004  27727.196945
        2005  50276.767126
        2006  54489.165192
        2007  47769.672234
        2008  42569.157521

    ``p_`` is the proportion of each origin's ultimate expected to have emerged
    to date, estimated from the loss ratio pattern rather than from link
    ratios.

    .. testcode::

        print(model.p_.iloc[..., -3:, :])

    .. testoutput::

                  2008
        2006  0.315457
        2007  0.154676
        2008  0.042753

    The optimal weight on the individual reserve, ``Z_``, is a function of
    ``p_`` alone and never exceeds one half.

    .. testcode::

        print(model.Z_.iloc[..., -3:, :])

    .. testoutput::

                  2008
        2006  0.359654
        2007  0.282274
        2008  0.171341

    The ``method`` parameter chooses the credibility weight. The
    ``"collective"`` and ``"individual"`` methods are the two extremes. The other
    methods result in a reserve estimate between these two extremes per origin period.

    The total reserve however, may not fall between that of the collective and
    individual reserves, since the individual reserve is below the collective reserve
    for some origin periods and above it for others.

    .. testcode::

        X = xyz["Paid"]
        sample_weight = xyz["Premium"].latest_diagonal
        for method in ["collective", "individual", "neuhaus", "benktander", "optimal"]:
            ibnr = cl.HurlimannCLR(method=method).fit(X, sample_weight=sample_weight).ibnr_
            print(f"{method:<11}{ibnr.sum():>10,.0f}")

    .. testoutput::

        collective    235,411
        individual    280,336
        neuhaus       233,365
        benktander    232,780
        optimal       241,485
    """

    # Fitted attributes, declared for type checkers and IDEs only. These bare
    # annotations create no attributes, so none exist until ``fit`` is called.
    X_: Triangle
    sample_weight_: Triangle
    m_: Triangle
    elr_: Triangle
    p_: Triangle
    Z_: Triangle
    r_individual_: Triangle
    r_collective_: Triangle
    mse_: Triangle
    ultimate_: Triangle
    ibnr_: Triangle
    ldf_: Triangle  # pyright: ignore[reportIncompatibleMethodOverride]
    cdf_: Triangle  # pyright: ignore[reportIncompatibleMethodOverride]

    _METHODS = (
        "optimal",
        "benktander",
        "neuhaus",
        "collective",
        "individual",
    )

    def __init__(
        self,
        method: Literal[
            "optimal",
            "benktander",
            "neuhaus",
            "collective",
            "individual",
        ] = "optimal",
    ) -> None:
        self.method = method

    def fit(
        self,
        X: Triangle,  # noqa - sklearn convention
        y: None = None,
        sample_weight: Triangle | None = None,
    ) -> HurlimannCLR:
        """
        Apply the Hürlimann credible loss ratio technique to triangle **X**

        Parameters
        ----------
        X: Triangle
            Loss data to which the model will be applied.
        y: None
            Ignored
        sample_weight: Triangle
            Required exposure, such as premium, of each origin period.

        Returns
        -------
        self: object
            Returns the instance itself.

        Examples
        --------
        Fit returns the estimator itself, with ``ultimate_`` populated. The
        repr shows non-default parameters.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            xyz = cl.load_sample("xyz")
            model = cl.HurlimannCLR(method="benktander").fit(
                xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal
            )
            print(model)
            print(model.ultimate_.iloc[..., -3:, :])

        .. testoutput::

            HurlimannCLR(method='benktander')
                          2261
            2006  77651.322500
            2007  56599.277036
            2008  40739.685328
        """
        sample_weight = self._require_sample_weight(sample_weight)
        if self.method not in self._METHODS:
            raise ValueError(
                f"method must be one of {self._METHODS}, got {self.method!r}."
            )
        # Skip MethodBase.fit: it fits a Development, which this method does not use.
        X, X_np, inc_tri, inc, V = self._prepare(X, sample_weight)  # noqa N806
        backend = X.array_backend
        self.X_ = X
        self.sample_weight_ = sample_weight.set_backend(backend)

        seen = ~np.isnan(inc)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.nansum(inc, axis=2, keepdims=True) / np.sum(
                np.where(seen, V, 0.0), axis=2, keepdims=True
            )  # eq. 2.1
        r = self._credible_reserve(inc, V, m)

        # MethodBase.ldf_ reads X_.ldf_, which is also what cdf_ is built from.
        setattr(self.X_, "ldf_", self._ldf_like(X_np, r["ldf"]).set_backend(backend))  # noqa B010
        self.m_ = self._like(inc_tri.sum("origin"), m).set_backend(backend)
        self.elr_ = self._like(self.m_.iloc[..., :1], r["elr"]).set_backend(backend)
        self._set_results(self, X_np.latest_diagonal, r, backend)
        return self

    def predict(
        self,
        X: Triangle,  # noqa N803 - sklearn convention
        sample_weight: Triangle | None = None,
    ) -> Triangle:
        """
        Predicts the credible loss ratio ultimate on a new triangle **X**

        The loss ratio pattern and burning cost loss ratio are the fitted
        ``m_`` and ``elr_``, not re-estimated from **X**. They are applied to the
        claims and the exposure of **X** to obtain its payout factors, reserves
        and ultimates, so ``method`` has the same meaning as in ``fit``.

        Parameters
        ----------
        X: Triangle
            Loss data to which the model will be applied. It may not have more
            development periods than the triangle that was fitted.
        sample_weight: Triangle
            Required exposure, such as premium, of each origin period of **X**.

        Returns
        -------
        X_new: Triangle
            Loss data with ``ultimate_``, ``ldf_`` and the payout factors
            ``p_``, the credibility weights ``Z_``, the reserves
            ``r_individual_`` and ``r_collective_`` and ``mse_`` applied.

        Examples
        --------
        Fit on all the origin periods, then apply the model to the three most
        recent. The pattern is the fitted one, so the ultimates are those of
        the fit.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            xyz = cl.load_sample("xyz")
            X, V = xyz["Paid"], xyz["Premium"].latest_diagonal
            model = cl.HurlimannCLR().fit(X, sample_weight=V)
            recent = X.origin >= X.origin[-3]
            predicted = model.predict(X[recent], sample_weight=V[V.origin >= V.origin[-3]])
            print(predicted.ultimate_)

        .. testoutput::

                          2261
            2006  77308.165192
            2007  59634.672234
            2008  45978.157521
        """
        sample_weight = self._require_sample_weight(sample_weight)
        X_new, X_np, _, inc, V = self._prepare(X, sample_weight)  # noqa N806
        if X_new.shape[3] > self.m_.shape[3]:
            raise ValueError("X has ages that exceed those available in model.")
        ages, fitted_ages = np.asarray(X_new.ddims), np.asarray(self.m_.ddims)
        if not np.array_equal(ages, fitted_ages[: len(ages)]):
            raise ValueError(
                "The development ages of X do not match those of the model, "
                f"{ages.tolist()} versus {fitted_ages.tolist()}."
            )

        # The fitted arrays, taken in the order of the segments and columns of X.
        rows = self._positions(
            pd.DataFrame(X_new.index), pd.DataFrame(self.m_.index), "The index of X"
        )
        cols = self._positions(
            self._labels(X_new.columns),
            self._labels(self.m_.columns),
            "The columns of X",
        )
        grid = np.ix_(rows, cols)
        m = np.asarray(self.m_.set_backend("numpy").values)[grid]
        elr = np.asarray(self.elr_.set_backend("numpy").values)[grid]
        ldf = np.asarray(self.ldf_.set_backend("numpy").values)[grid]
        r = self._credible_reserve(inc, V, m, elr=elr)

        setattr(  # noqa B010
            X_new,
            "ldf_",
            self._ldf_like(X_np, ldf, np.asarray(self.ldf_.ddims)).set_backend(
                X_new.array_backend
            ),
        )
        self._set_results(X_new, X_np.latest_diagonal, r, X_new.array_backend)
        return X_new

    @staticmethod
    def _labels(labels: Any) -> pd.DataFrame:
        """Wrap the labels of an axis, such as the columns, as a one-column frame."""
        return pd.DataFrame({"label": list(labels)})

    @staticmethod
    def _positions(target: pd.DataFrame, source: pd.DataFrame, what: str) -> np.ndarray:
        """
        The position in ``source`` of each label in ``target``.

        A ``source`` with one label is broadcast: it applies to every label of
        ``target``.

        Parameters
        ----------
        target: pd.DataFrame
            The labels to look up, one row per label.
        source: pd.DataFrame
            The labels to look in, with the same columns.
        what: str
            What ``target`` is, for the error message.

        Returns
        -------
        np.ndarray
            The position in ``source`` of each row of ``target``.
        """
        if len(source) == 1:
            return np.zeros(len(target), dtype=int)
        if list(target.columns) != list(source.columns):
            raise ValueError(
                f"{what} is labelled by {list(target.columns)}, not "
                f"{list(source.columns)} as in the model."
            )
        lookup = {tuple(k): i for i, k in enumerate(source.itertuples(index=False))}
        missing = [
            tuple(k) for k in target.itertuples(index=False) if tuple(k) not in lookup
        ]
        if missing:
            raise ValueError(f"{what} has labels not in the model: {missing}.")
        return np.array([lookup[tuple(k)] for k in target.itertuples(index=False)])

    def _require_sample_weight(self, sample_weight: Triangle | None) -> Triangle:
        """The exposure, which ``fit`` and ``predict`` need, or an error if missing."""
        if sample_weight is None:
            raise ValueError(
                f"sample_weight is required. For {self.__class__.__name__}, "
                "this is the exposure of each origin period."
            )
        return sample_weight

    def _prepare(
        self,
        X: Triangle,  # noqa N803
        sample_weight: Triangle,
    ) -> tuple[Triangle, Triangle, Triangle, np.ndarray, np.ndarray]:
        """
        Validate the inputs of ``fit`` and ``predict`` and put them in numpy.

        The arithmetic is done in numpy whatever the backend of **X**, and the
        results are converted back at the end.

        Returns
        -------
        tuple
            ``X`` as a cumulative triangle by development period, that in numpy,
            the incremental triangle in numpy, its values shaped (index, column,
            origin, development), and the exposure of each origin shaped
            (index, column, origin, 1).
        """
        X = X.incr_to_cum().val_to_dev() if not X.is_cumulative else X.val_to_dev()  # noqa N806
        self.validate_weight(X, sample_weight)
        if sample_weight.shape[2] != X.shape[2] or not X.origin.equals(
            sample_weight.origin
        ):
            raise ValueError("X and sample_weight must have the same origin periods.")

        X_np = X.set_backend("numpy")  # noqa N806
        inc_tri = X_np.cum_to_incr()
        inc = np.asarray(inc_tri.values)
        V = np.asarray(sample_weight.set_backend("numpy").latest_diagonal.values)  # noqa N806
        # The exposure of each segment of X, matched by label rather than position.
        V = V[self._positions(X.index, sample_weight.index, "The index of X"), ...]  # noqa N806
        V = np.broadcast_to(V, inc.shape[:3] + (1,))  # noqa N806
        return X, X_np, inc_tri, inc, V

    def _set_results(
        self,
        target: Any,
        latest: Triangle,
        r: dict[str, Any],
        backend: str,
    ) -> None:
        """
        Set the reserve attributes on ``target``, the estimator or a predicted triangle.

        Parameters
        ----------
        target: object
            Receives ``p_``, ``Z_``, ``r_individual_``, ``r_collective_``, ``mse_``
            and ``ultimate_``.
        latest: Triangle
            The latest diagonal in numpy, whose labels the attributes copy.
        r: dict
            The result of ``_credible_reserve``.
        backend: str
            The array backend to convert the attributes to.
        """
        for name, key in (
            ("p_", "p"),
            ("Z_", "Z"),
            ("r_individual_", "r_ind"),
            ("r_collective_", "r_coll"),
            ("mse_", "mse"),
        ):
            setattr(target, name, self._like(latest, r[key]).set_backend(backend))
        ultimate = self._like(latest, r["C"] + r["ibnr"]).set_backend(backend)
        target.ultimate_ = self._set_ult_attr(ultimate)

    def _credible_reserve(
        self,
        inc: np.ndarray,
        V: np.ndarray,  # noqa N803
        m: np.ndarray,
        elr: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        The credible reserve of each origin period for a given loss ratio pattern.

        Parameters
        ----------
        inc: np.ndarray
            Incremental claims, shaped (index, column, origin, development), with
            nan for the development periods not yet observed.
        V: np.ndarray
            The exposure of each origin period, shaped (index, column, origin, 1).
        m: np.ndarray
            The loss ratio pattern, shaped (index, column, 1, development), where
            development may exceed that of ``inc``. Periods with nan, observed for
            no origin, contribute nothing.
        elr: np.ndarray, optional
            The burning cost loss ratio. By default, the sum of ``m``.

        Returns
        -------
        dict
            The arrays ``elr``, ``p``, ``Z``, ``C``, ``r_ind``, ``r_coll``,
            ``ibnr``, ``mse`` and ``ldf``.
        """
        seen = ~np.isnan(inc)
        ages = np.arange(inc.shape[3])
        last = np.max(
            np.where(seen, ages, -1),
            axis=3,
            keepdims=True,
        )  # latest development index
        # The burning cost loss ratio.
        elr_ = np.sum(np.nan_to_num(m), axis=3, keepdims=True) if elr is None else elr
        with np.errstate(divide="ignore", invalid="ignore"):
            # Development periods observed for no origin (e.g. the last column after
            # dropping the oldest origin) have m = 0/0 = NaN. They carry no data, so
            # they add nothing to the pattern rather than poisoning it.
            m_filled = np.nan_to_num(m)
            cum_m = np.cumsum(m_filled, axis=3)
            to_date = np.take_along_axis(
                np.broadcast_to(cum_m, inc.shape[:3] + cum_m.shape[3:]),
                np.maximum(last, 0),
                axis=3,
            )
            p = to_date / elr_  # eq. 2.3
            q = 1 - p
            C = np.take_along_axis(  # noqa N806
                np.cumsum(np.nan_to_num(inc), axis=3), np.maximum(last, 0), axis=3
            )
            r_ind = q / p * C  # eq. 2.6
            r_coll = q * elr_ * V  # eq. 2.7
            Z = {  # noqa N806
                "individual": np.ones_like(p),
                "collective": np.zeros_like(p),
                "benktander": p,
                "neuhaus": to_date,
                # The limit as p -> 0 is 0, where 0/0 would be nan.
                "optimal": np.where(p > 0, p / (p + np.sqrt(np.where(p > 0, p, 1))), 0),
            }[self.method]
            ibnr = np.where(Z == 0, 0.0, Z * r_ind) + (1 - Z) * r_coll  # eq. 3.1
            # eq. 4.11 with t = sqrt(p), written to stay finite when q = 0. As p -> 0
            # the collective term diverges, so a zero weight leaves an infinite mse.
            mse = (
                np.where(Z == 0, 0.0, q**2 * Z**2 / p)
                + q
                + q**2 * (1 - Z) ** 2 / np.sqrt(p)
            )
            # The payout pattern cum_m / elr is the inverse of the cdf, so the age to
            # age factors are the ratios of successive cumulative loss ratios.
            ldf = cum_m[..., 1:] / cum_m[..., :-1]
        return {
            "elr": elr_,
            "p": p,
            "Z": Z,
            "C": C,
            "r_ind": r_ind,
            "r_coll": r_coll,
            "ibnr": ibnr,
            "mse": mse,
            "ldf": ldf,
        }

    @staticmethod
    def _ldf_like(
        X: Triangle,  # noqa N803
        values: np.ndarray,
        ddims: np.ndarray | None = None,
    ) -> Triangle:
        """Wrap age to age factors in a pattern Triangle, as ``Development`` does."""
        from chainladder import options

        obj = X[X.origin == X.origin.min()]
        obj.values = values
        # Not X.link_ratio.ddims, which drops trailing development periods that
        # no origin has observed, while the pattern keeps them.
        obj.ddims = np.asarray(X.ddims)[:-1] if ddims is None else ddims
        obj.valuation_date = pd.to_datetime(options.ULT_VAL)
        obj.is_pattern = True
        obj.is_cumulative = False
        obj.virtual_columns.columns = {}
        obj._set_slicers()  # noqa
        return obj

    @staticmethod
    def _like(
        template: TriangleProtocol,
        values: np.ndarray,
    ) -> Triangle:
        """
        Return a copy of a triangle with its values replaced.

        ``fit`` does its arithmetic on bare numpy arrays. This helper wraps those
        arrays back into Triangles by reusing the labels, axes and metadata of an
        existing triangle, so the fitted attributes look like the input.

        Parameters
        ----------
        template: TriangleProtocol
            The triangle whose index, columns, origin and development axes are
            reused. It is not modified.
        values: np.ndarray
            The new values. They must have the same shape as ``template.values``.

        Returns
        -------
        Triangle
            A copy of ``template`` holding ``values``.

        Examples
        --------
        The result has the shape and labels of the template but the new values,
        and the template itself is left unchanged.

        .. testsetup::

            import chainladder as cl
            from chainladder.methods import HurlimannCLR

        .. testcode::

            latest = cl.load_sample("raa").latest_diagonal
            doubled = HurlimannCLR._like(latest, latest.values * 2)
            print(doubled.iloc[..., -3:, :])
            print(latest.iloc[..., -3:, :])

        .. testoutput::

                     1990
            1988  26224.0
            1989  10790.0
            1990   4126.0
                     1990
            1988  13112.0
            1989   5395.0
            1990   2063.0
        """
        obj = template.copy()
        obj.values = values
        return obj
