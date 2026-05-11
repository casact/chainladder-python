# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.tails import TailBase
from chainladder.development import Development


class TailConstant(TailBase):
    """Allows for the entry of a constant tail factor to LDFs.

    Parameters
    ----------
    tail: float
        The constant to apply to all LDFs within a triangle object.
    decay: float (default=0.50)
        An exponential decay constant that allows for decay over future
        development periods.  A decay rate of 0.5 sets the development portion
        of each successive LDF to 50% of the previous LDF.
    attachment_age: int (default=None)
        The age at which to attach the fitted curve.  If None, then the latest
        age is used. Measures of variability from original ``ldf_`` are retained
        when being used in conjunction with the MackChainladder method.
    projection_period: int
        The number of months beyond the latest available development age the
        `ldf_` and `cdf_` vectors should extend.

    Attributes
    ----------
    ldf_:
        ldf with tail applied.
    cdf_:
        cdf with tail applied.
    tail_: DataFrame
        Point estimate of tail at latest maturity available in the Triangle.

    Notes
    -----
    The tail constant does not support the entry of variability parameters
    necessary for stochastic approaches, so any usage of TailConstant will be
    inherently deterministic.

    See also
    --------
    TailCurve

    Examples
    --------

    Applying a 5% tail factor to the RAA sample triangle. The ``tail_``
    attribute returns the point estimate of the tail at the latest maturity,
    while ``ldf_`` and ``cdf_`` are extended to include the tail.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        raa = cl.load_sample('raa')
        dev = cl.Development().fit_transform(raa)
        tail = cl.TailConstant(tail=1.05).fit(dev)
        print(tail.tail_)

    .. testoutput::

               120-Ult
        (All)     1.05

    .. testcode::

        print(tail.cdf_)

    .. testoutput::

                 12-Ult    24-Ult    36-Ult    48-Ult    60-Ult    72-Ult   84-Ult    96-Ult   108-Ult  120-Ult  132-Ult
        (All)  9.366246  3.122749  1.923441  1.513462  1.291708  1.160163  1.11347  1.077625  1.059677     1.05  1.02538

    Using ``decay`` to control how the tail factor is distributed across
    future development periods. A higher ``decay`` rate retains more of the
    tail factor in the immediately following development period rather than
    spreading it out evenly. Compare the ``120-132`` and ``132-144`` LDFs
    below against the default ``decay=0.5``.

    .. testcode::

        tail = cl.TailConstant(tail=1.10, decay=0.75).fit(dev)
        print(tail.ldf_)

    .. testoutput::

                  12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132   132-144
        (All)  2.999359  1.623523  1.270888  1.171675  1.113385  1.041935  1.033264  1.016936  1.009217  1.023512  1.074731

    Using ``attachment_age`` to attach the tail at an earlier development age
    than the latest available. Below, the tail is attached at age 72 instead
    of the default 120, so ages 84 onward are filled with the decayed tail
    factors rather than the original LDFs.

    .. testcode::

        tail = cl.TailConstant(tail=1.05, attachment_age=72).fit(dev)
        print(tail.cdf_)

    .. testoutput::

                 12-Ult    24-Ult  36-Ult    48-Ult    60-Ult  72-Ult   84-Ult    96-Ult  108-Ult   120-Ult   132-Ult
        (All)  8.476874  2.826229  1.7408  1.369751  1.169054    1.05  1.02538  1.013216  1.00717  1.004156  1.002652

    Using ``projection_period`` to extend the ``ldf_`` and ``cdf_`` vectors
    further beyond the latest available development age. Below, the projection
    is extended by 36 months instead of the default 12, adding two extra
    development columns to ``cdf_``.

    .. testcode::

        tail = cl.TailConstant(tail=1.05, projection_period=36).fit(dev)
        print(tail.cdf_)

    .. testoutput::

                 12-Ult    24-Ult    36-Ult    48-Ult    60-Ult    72-Ult   84-Ult    96-Ult   108-Ult  120-Ult  132-Ult   144-Ult  156-Ult
        (All)  9.366246  3.122749  1.923441  1.513462  1.291708  1.160163  1.11347  1.077625  1.059677     1.05  1.02538  1.013216  1.00717
    """

    def __init__(self, tail=1.0, decay=0.5, attachment_age=None, projection_period=12):
        self.tail = tail
        self.decay = decay
        self.attachment_age = attachment_age
        self.projection_period = projection_period

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
