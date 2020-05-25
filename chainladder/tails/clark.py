# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.tails import TailBase
from chainladder.development import DevelopmentBase
from chainladder.development.clark import ClarkLDF
import numpy as np
from chainladder.utils.cupy import cp


class TailClark(TailBase):
    """Allows for extraploation of LDFs to form a tail factor.

    Parameters
    ----------
    growth : {'loglogistic', 'weibull'}
        The growth function to be used in curve fitting development patterns.
        Options are 'loglogistic' and 'weibull'

    Attributes
    ----------
    ldf_ :
        ldf with tail applied.
    cdf_ :
        cdf with tail applied.

    """
    def __init__(self, growth='loglogistic'):
        self.growth = growth

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the tail will be applied.
        y : Ignored
        sample_weight : Triangle-like
            Exposure vector used to invoke the Cape Cod method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, sample_weight)
        model = ClarkLDF(growth=self.growth).fit(X, sample_weight=sample_weight)
        xp = cp.get_array_module(X.values)
        age_offset = {'Y':6., 'Q':1.5, 'M':0.5}[X.development_grain]
        tail = 1/model.G_(xp.array(
            [item*self._ave_period[1]+X.ddims[-1]-age_offset
             for item in range(self._ave_period[0]+1)]))
        tail = xp.concatenate((tail.values[..., :-1]/tail.values[..., -1],
                               tail.values[..., -1:]), -1)
        self.ldf_.values = xp.concatenate(
            (X.ldf_.values,  xp.repeat(tail, X.shape[2], 2)), -1)
        self.cdf_ = DevelopmentBase._get_cdf(self)
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

        X.cdf_ = self.cdf_
        X.ldf_ = self.ldf_
        return X
