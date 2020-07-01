# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development.base import DevelopmentBase
import copy
import numpy as np
import pandas as pd
from chainladder.utils.cupy import cp


class DevelopmentConstant(DevelopmentBase):
    """ A Estimator that allows for including of external patterns into a
    Development style model.  When this estimator is fit against a triangle,
    only the grain of the existing triangle is retained.

    Parameters
    ----------
    patterns : dict or callable
        A dictionary key:value representation of age(in months):value. If callable
        is supplied, callable must return a dict for each element of the callable axis
    style : string, optional (default='ldf')
        Type of pattern given to the Estimator.  Options include 'cdf' or 'ldf'.
    callable_axis : 0 or 1
        If a callable is supplied, the axis (index or column) along which to apply
        the callable.  If patterns is not a callable, then this parameter is ignored.

    Attributes
    ----------
    ldf_ : Triangle
        The estimated loss development patterns
    cdf_ : Triangle
        The estimated cumulative development patterns

    """
    def __init__(self, patterns=None, style='ldf', callable_axis=0):
        self.patterns = patterns
        self.style = style
        self.callable_axis = callable_axis

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
        obj.values = xp.ones(X.shape)[..., 0:1, :-1]
        if callable(self.patterns):
            if len(obj.key_labels) == 1 and self.callable_axis==0:
                ldf = obj.index.iloc[:, 0].apply(self.patterns)
            else:
                ldf = obj.index.apply(self.patterns, axis=self.callable_axis).iloc[:, 0]
            ldf = pd.concat(ldf.apply(pd.DataFrame, index=[0]).values, axis=0).fillna(1)[obj.ddims[:-1]].values
            if self.callable_axis==0:
                ldf = ldf[:, None, None, :]
            else:
                ldf = ldf[None, :, None, :]
        else:
            ldf = xp.array([float(self.patterns[item]) for item in obj.ddims[:-1]])
            ldf = ldf[None, None, None, :]
        if self.style == 'cdf':
            ldf = xp.concatenate((ldf[..., :-1]/ldf[..., 1:], ldf[..., -1:]), -1)
        obj.values = obj.values * ldf
        obj.ddims = X.link_ratio.ddims
        obj.odims = obj.odims[0:1]
        obj.valuation = obj._valuation_triangle(obj.ddims)
        obj.nan_override = True
        obj._set_slicers()

        self.ldf_ = obj
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
        triangles = ['ldf_', 'sigma_', 'std_err_']
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new
