# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development.base import DevelopmentBase
import copy
import numpy as np
from chainladder.utils.cupy import cp


class DevelopmentConstant(DevelopmentBase):
    """ A Estimator that allows for including of extneral patterns into a
    Development style model.  Currently, this only supports single triangles.
    When this estimator is fit against a triangle, only the grain of the
    existing triangle is retained.


    Parameters
    ----------
    patterns : dict,  (default={})
        A dictionary key/value representation of age(in months)/value
    style : string, optional (default='ldf')
        type of averaging to use for ldf average calculation.  Options include
        'volume', 'simple', and 'regression'


    Attributes
    ----------
    ldf_ : Triangle
        The estimated loss development patterns
    cdf_ : Triangle
        The estimated cumulative development patterns

    """
    def __init__(self, patterns={}, style='ldf'):
        self.patterns = patterns
        self.style = style

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the munich adjustment will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        obj = copy.copy(X)
        xp = cp.get_array_module(obj.values)
        obj.values = xp.ones(X.shape)[..., :-1]
        ldf = xp.array([float(self.patterns[item]) for item in obj.ddims[:-1]])
        if self.style == 'cdf':
            ldf = xp.concatenate((ldf[:-1]/ldf[1:], xp.array([ldf[-1]])))
        ldf = ldf[xp.newaxis, xp.newaxis, xp.newaxis, ...]
        obj.values = obj.values * ldf
        obj.ddims = X.link_ratio.ddims
        obj.valuation = obj._valuation_triangle(obj.ddims)
        obj.nan_override = True
        obj._set_slicers()

        self.ldf_ = obj
        self.cdf_ = self._get_cdf(self)
        self.sigma_ = self.ldf_*0+1
        self.std_err_ = self.ldf_*0+1
        return self

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
        X_new = copy.copy(X)
        triangles = ['cdf_', 'ldf_', 'sigma_', 'std_err_']
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new
