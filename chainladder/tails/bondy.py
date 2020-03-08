# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from scipy.optimize import least_squares
from chainladder.utils.cupy import cp
from chainladder.tails import TailBase
from chainladder.development import DevelopmentBase, Development

class TailBondy(TailBase):
    """Estimator for the Generalized Bondy tail factor.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    earliest_age : int
        The earliest age from which the Bondy exponent is to be calculated.
        Defaults to earliest available in the Triangle.
    decay : float (default=0.50)
        An exponential decay constant that allows for decay over future
        development periods.  A decay rate of 0.5 sets the development portion
        of each successive LDF to 50% of the previous LDF.

    Attributes
    ----------
    b_ :
        The Bondy exponent
    ldf_ :
        ldf with tail applied.
    cdf_ :
        cdf with tail applied.
    sigma_ :
        sigma with tail factor applied.
    std_err_ :
        std_err with tail factor applied

    See also
    --------
    TailCurve

    """
    def __init__(self, earliest_age=None, decay=0.5):
        self.earliest_age = earliest_age
        self.decay = decay

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
        xp = cp.get_array_module(X.values)
        obj = Development().fit_transform(X) if 'ldf_' not in X else X
        b_optimized = []
        initial = xp.where(obj.ddims==self.earliest_age)[0][0] if self.earliest_age else 0
        for num in range(len(obj.vdims)):
            b0 = xp.ones(obj.shape[0])*.5
            data = xp.log(obj.ldf_.values[:, num, 0,initial:])
            b_optimized.append(least_squares(
                TailBondy.solver, x0=b0, kwargs={'data': data}).x[:, None])
        self.b_ = xp.concatenate(b_optimized, 1)
        tail = xp.exp(xp.log(obj.ldf_.values[..., 0:1, initial:initial+1]) * \
                      self.b_**(len(obj.ldf_.ddims)-1))
        tail = (tail**(self.b_/(1-self.b_)))*tail
        self = self._apply_decay(obj, tail)
        sigma, std_err = self._get_tail_stats(obj)
        self.sigma_.values[..., -1] = sigma[..., -1]
        self.std_err_.values[..., -1] = std_err[..., -1]
        return self

    def transform(self, X):
        """Transform X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

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
        X.std_err_ = self.std_err_
        X.cdf_ = self.cdf_
        X.ldf_ = self.ldf_
        X.sigma_ = self.sigma_
        X.b_ = self.b_
        return X

    @staticmethod
    def solver(b, data):
        xp = cp.get_array_module(data)
        arange = xp.repeat(xp.arange(data.shape[-1])[None, :], data.shape[0], 0)
        out = data - (data[:, 0])[:, None]*b**(arange)
        return out.flatten()
