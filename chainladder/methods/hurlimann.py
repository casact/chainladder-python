# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from chainladder.methods import MethodBase


class HurlimannCLR(MethodBase):
    """
    The Hürlimann credible loss ratio reserving model

    Implements the credible loss ratio claims reserves of Hürlimann
    :cite:`hurlimann2009`. The reserve for each origin period is a credibility
    mixture of an individual reserve, which grosses up the latest paid claims
    by a loss ratio payout factor, and a collective reserve, which applies
    that payout factor to a burning cost ultimate (the loss ratio method of
    Mack). The Benktander and Neuhaus methods and the optimal credible reserve
    are special cases that differ only in the credibility weight.

    Unlike the chain-ladder based methods, the payout pattern is estimated
    from incremental claims and exposure rather than from link ratios, so it
    depends on ``sample_weight``.

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
        its ultimate expected to be paid to date (eq. 2.3).
    credibility_: Triangle
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
    ultimate_: Triangle
        The ultimate losses per the method
    ibnr_: Triangle
        The IBNR per the method

    See Also
    --------
    Benktander : Iterative Bornhuetter-Ferguson method.
    CapeCod : Estimates apriori loss ratios from exposure.

    Notes
    -----
    The paper assumes no development beyond the last development period of
    the triangle, so there is no tail.

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

    ``p_`` is the proportion of each origin's ultimate expected to be paid to
    date, estimated from the loss ratio pattern rather than from link ratios.
    The optimal weight on the individual reserve, ``credibility_``, is a
    function of ``p_`` alone and never exceeds one half.

    .. testcode::

        print(model.p_.iloc[..., -3:, :])
        print(model.credibility_.iloc[..., -3:, :])

    .. testoutput::

                  2008
        2006  0.315457
        2007  0.154676
        2008  0.042753
                  2008
        2006  0.359654
        2007  0.282274
        2008  0.171341

    The ``method`` parameter chooses the credibility weight. The
    ``"collective"`` and ``"individual"`` methods are the two extremes. For
    each origin period the others fall between the two, though a total need
    not, since the individual reserve is below the collective one for some
    origins and above it for others.

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

    _METHODS = ("optimal", "benktander", "neuhaus", "collective", "individual")

    def __init__(self, method="optimal"):
        self.method = method

    def fit(self, X, y=None, sample_weight=None):
        """
        Applies the Hürlimann credible loss ratio technique to triangle **X**

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
        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        if self.method not in self._METHODS:
            raise ValueError(
                f"method must be one of {self._METHODS}, got {self.method!r}."
            )
        # Skip MethodBase.fit: it fits a Development, which this method does not use.
        X = X.incr_to_cum().val_to_dev() if not X.is_cumulative else X.val_to_dev()
        self.X_ = X
        self.validate_weight(X, sample_weight)
        self.sample_weight_ = sample_weight.set_backend(X.array_backend)
        if sample_weight.shape[2] != X.shape[2]:
            raise ValueError("X and sample_weight must have the same origin periods.")

        # Computed in numpy whatever the backend of X, and converted back at the end.
        backend = X.array_backend
        X_np = X.set_backend("numpy")
        inc_tri = X_np.cum_to_incr()
        inc = inc_tri.values  # (index, column, origin, development)
        # Exposure of each origin, broadcast to the shape of X.
        V = self.sample_weight_.set_backend("numpy").latest_diagonal.values
        V = np.broadcast_to(V, inc.shape[:3] + (1,))

        seen = ~np.isnan(inc)
        ages = np.arange(inc.shape[3])
        last = np.max(
            np.where(seen, ages, -1), axis=3, keepdims=True
        )  # latest development index
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.nansum(inc, axis=2, keepdims=True) / np.sum(
                np.where(seen, V, 0.0), axis=2, keepdims=True
            )  # eq. 2.1
            elr = np.sum(m, axis=3, keepdims=True)  # burning cost loss ratio
            cum_m = np.cumsum(m, axis=3)
            to_date = np.take_along_axis(
                np.broadcast_to(cum_m, inc.shape), np.maximum(last, 0), axis=3
            )
            p = to_date / elr  # eq. 2.3
            q = 1 - p
            C = np.take_along_axis(
                np.cumsum(np.nan_to_num(inc), axis=3), np.maximum(last, 0), axis=3
            )
            r_ind = q / p * C  # eq. 2.6
            r_coll = q * elr * V  # eq. 2.7
            z = {
                "individual": np.ones_like(p),
                "collective": np.zeros_like(p),
                "benktander": p,
                "neuhaus": to_date,
                "optimal": p / (p + np.sqrt(p)),
            }[self.method]
            ibnr = z * r_ind + (1 - z) * r_coll  # eq. 3.1
            # eq. 4.11 with t = sqrt(p), written to stay finite when q = 0.
            mse = q**2 * z**2 / p + q + q**2 * (1 - z) ** 2 / np.sqrt(p)

        latest = X_np.latest_diagonal
        self.m_ = self._like(inc_tri.sum("origin"), m)
        self.elr_ = self._like(self.m_.iloc[..., :1], elr)
        self.p_ = self._like(latest, p)
        self.credibility_ = self._like(latest, z)
        self.r_individual_ = self._like(latest, r_ind)
        self.r_collective_ = self._like(latest, r_coll)
        self.mse_ = self._like(latest, mse)
        self.ultimate_ = self._set_ult_attr(self._like(latest, C + ibnr))
        for name in (
            "m_",
            "elr_",
            "p_",
            "credibility_",
            "r_individual_",
            "r_collective_",
            "mse_",
            "ultimate_",
        ):
            setattr(self, name, getattr(self, name).set_backend(backend))
        return self

    def predict(self, X, sample_weight=None):
        raise NotImplementedError("HurlimannCLR does not support predict.")

    @staticmethod
    def _like(template, values):
        """A copy of ``template`` holding ``values``, which has the same shape."""
        obj = template.copy()
        obj.values = values
        return obj
