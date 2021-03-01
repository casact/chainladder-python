# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.tails import TailBase
from chainladder.development import Development


class TailConstant(TailBase):
    """Allows for the entry of a constant tail factor to LDFs.

    Parameters
    ----------
    tail : float
        The constant to apply to all LDFs within a triangle object.
    decay : float (default=0.50)
        An exponential decay constant that allows for decay over future
        development periods.  A decay rate of 0.5 sets the development portion
        of each successive LDF to 50% of the previous LDF.
    attachment_age: int (default=None)
        The age at which to attach the fitted curve.  If None, then the latest
        age is used. Measures of variability from original ``ldf_`` are retained
        when being used in conjunction with the MackChainladder method.

    Attributes
    ----------
    ldf_ :
        ldf with tail applied.
    cdf_ :
        cdf with tail applied.
    tail_ : DataFrame
        Point estimate of tail at latest maturity available in the Triangle.
    sigma_ :
        sigma with tail factor applied.
    std_err_ :
        std_err with tail factor applied

    Notes
    -----
    The tail constant does not support the entry of variability parameters
    necessary for stochastic approaches, so any usage of TailConstant will be
    inherently deterministic.

    See also
    --------
    TailCurve

    """

    def __init__(self, tail=1.0, decay=0.5, attachment_age=None):
        self.tail = tail
        self.decay = decay
        self.attachment_age = attachment_age

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the tail will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, sample_weight)
        xp = self.ldf_.get_array_module()
        tail = self.tail
        if self.attachment_age:
            attach_idx = xp.min(xp.where(X.ddims >= self.attachment_age))
        else:
            attach_idx = len(X.ddims) - 1
        self = self._apply_decay(X, tail, attach_idx)
        obj = Development().fit_transform(X) if "ldf_" not in X else X
        self._get_tail_stats(obj)
        return self
