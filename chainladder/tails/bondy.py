# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from chainladder.tails import TailBase
from chainladder.development import DevelopmentBase, Development


class TailBondy(TailBase):
    """Estimator for the Generalized Bondy tail factor.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    earliest_age: int
        The earliest age from which the Bondy exponent is to be calculated.
        Defaults to latest age in the Triangle. Any available development
        age can be used.
    attachment_age: int (default=None)
        The age at which to attach the fitted curve.  If None, then the latest
        age is used. Measures of variability from original ``ldf_`` are retained
        when being used in conjunction with the MackChainladder method.

    Attributes
    ----------
    ldf_: Triangle
        ldf with tail applied.
    cdf_: Triangle
        cdf with tail applied.
    tail_: DataFrame
        Point estimate of tail at latest maturity available in the Triangle.
    b_: DataFrame
        The Bondy exponent
    earliest_ldf_: DataFrame
        The LDF associated with the ``earliest_age`` pick.
    projection_period: int
        The number of months beyond the latest available development age the
        `ldf_` and `cdf_` vectors should extend.

    See also
    --------
    TailCurve

    Examples
    --------
    Suppose an actuary needs a tail provision for a reinsurance triangle
    that stops at age 120. Without any hyperparameters, ``TailBondy``
    reproduces the classic Bondy method: the exponent ``b_`` keeps its
    starting value of 0.5, which makes the tail a simple repeat of the last
    observed development factor.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        dev = cl.Development().fit_transform(cl.load_sample("raa"))
        bondy = cl.TailBondy().fit(dev)
        print(round(float(dev.ldf_.values[0, 0, 0, -1]), 4))
        print(round(float(bondy.b_.iloc[0, 0]), 4))
        print(round(float(bondy.tail_.iloc[0, 0]), 4))

    .. testoutput::

        1.0092
        0.5
        1.0092

    Setting ``earliest_age`` turns this into the Generalized Bondy method:
    every development factor from that age forward enters the regression and
    the Bondy exponent is estimated from the data. The exponent measures how
    quickly development decays. Each log development factor is assumed to be
    ``b_`` times the prior one, so an estimate below 0.5 means the pattern
    decays faster than the classic Bondy assumption implies (a lighter
    tail), while an estimate above 0.5 signals more persistent development
    (a heavier tail).

    .. testcode::

        gen = cl.TailBondy(earliest_age=12).fit(dev)
        print(round(float(gen.b_.iloc[0, 0]), 4))
        print(round(float(gen.tail_.iloc[0, 0]), 4))

    .. testoutput::

        0.4845
        1.0031

    Although 0.4845 looks close to 0.5, the exponent compounds through the
    tail formula (the tail is the last fitted factor raised to
    ``b_ / (1 - b_)``), so the tail provision falls from 0.92% to 0.31% of
    losses, roughly a threefold difference in the booked tail. Comparing the
    full patterns confirms the two fits only diverge beyond the edge of the
    triangle:

    .. testcode::

        print(bondy.ldf_.values[0, 0, 0, :].round(4))
        print(gen.ldf_.values[0, 0, 0, :].round(4))

    .. testoutput::

        [2.9994 1.6235 1.2709 1.1717 1.1134 1.0419 1.0333 1.0169 1.0092 1.0046
         1.0046]
        [2.9994 1.6235 1.2709 1.1717 1.1134 1.0419 1.0333 1.0169 1.0092 1.0016
         1.0015]

    The choice of ``earliest_age`` is a judgment call: include enough ages
    for a stable estimate of ``b_``, but exclude immature ages whose
    volatile link ratios would distort the decay rate the exponent is meant
    to capture.

    """

    def __init__(self, earliest_age=None, attachment_age=None, projection_period=12):
        self.earliest_age = earliest_age
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
        if self.attachment_age and self.attachment_age < self.earliest_age:
            raise ValueError("attachment_age must not be before earliest_age.")
        backend = X.array_backend
        if X.array_backend == "cupy":
            X = X.set_backend("numpy", deep=True)
        else:
            X = X.set_backend("numpy")
        xp = X.get_array_module()
        super().fit(X, y, sample_weight)

        if self.earliest_age is None:
            earliest_age = X.ddims[-2]
        else:
            earliest_age = X.ddims[
                int(
                    self.earliest_age / ({"Y": 12, "S": 6, "Q": 3, "M": 1}[X.development_grain])
                )
                - 1
            ]
        attachment_age = self.attachment_age if self.attachment_age else X.ddims[-2]
        obj = Development().fit_transform(X) if "ldf_" not in X else X
        b_optimized = []
        initial = xp.where(obj.ddims == earliest_age)[0][0] if earliest_age else 0
        for num in range(len(obj.vdims)):
            b0 = (xp.ones(obj.shape[0]) * 0.5)[:, None]
            data = xp.log(obj.ldf_.values[:, num, 0, initial:])
            b0 = xp.concatenate((b0, data[..., 0:1]), axis=1)
            b_optimized.append(
                least_squares(
                    TailBondy._solver, x0=b0.flatten(), kwargs={"data": data, "xp": xp}
                ).x
            )
        self.b_ = xp.concatenate(
            [item.reshape(-1, 2)[:, 0:1] for item in b_optimized], axis=1
        )[..., None, None]
        self.earliest_ldf_ = xp.exp(
            xp.concatenate(
                [item.reshape(-1, 2)[:, 1:2] for item in b_optimized], axis=1
            )[..., None, None]
        )
        if sum(X.ddims > earliest_age) > 1:
            tail = xp.exp(self.earliest_ldf_ * self.b_ ** (len(obj.ldf_.ddims) - 1))
        else:
            tail = self.ldf_.values[..., 0, initial]
        tail = tail ** (self.b_ / (1 - self.b_))
        f0 = self.ldf_.values[..., 0:1, initial : initial + 1]
        fitted = f0 ** (
            self.b_ ** (np.arange(sum(X.ddims >= earliest_age))[None, None, None, :])
        )
        fitted = xp.concatenate(
            (fitted, fitted[..., -1:] ** (self.b_ / (1 - self.b_))), axis=-1
        )
        fitted = xp.repeat(fitted, self.ldf_.shape[2], axis=2)
        rows = X.index.set_index(X.key_labels).index
        self.b_ = pd.DataFrame(self.b_[..., 0, 0], index=rows, columns=X.vdims)
        self.earliest_ldf_ = pd.DataFrame(
            self.earliest_ldf_[..., 0, 0], index=rows, columns=X.vdims
        )
        self.ldf_.values = xp.concatenate(
            (
                self.ldf_.values[..., : sum(X.ddims <= attachment_age)],
                fitted[..., -sum(X.ddims >= attachment_age) :],
            ),
            axis=-1,
        )
        self._get_tail_stats(obj)
        if backend == "cupy":
            self = self.set_backend(backend, inplace=True, deep=True)
        return self

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : Triangle
            Triangle must contain the ``ldf_`` development attribute.

        Returns
        -------
        X_new : Triangle
            New Triangle with tail factor applied to its development
            attributes.
        """
        X_new = super().transform(X)
        X_new.b_ = self.b_
        X_new.earliest_ldf_ = self.earliest_ldf_
        return X_new

    @staticmethod
    def _solver(b, data, xp):
        b = b.reshape(-1, 2)
        arange = xp.repeat(xp.arange(data.shape[-1])[None, :], data.shape[0], 0)
        out = data - (b[:, 1:2]) * b[:, 0:1] ** (arange)
        return out.flatten()
