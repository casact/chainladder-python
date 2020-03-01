# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from chainladder.utils.cupy import cp
import copy
from chainladder.methods import MethodBase


class Chainladder(MethodBase):
    """
    The basic determinsitic chainladder method.

    Parameters
    ----------
    None

    Attributes
    ----------
    X_
        returns **X** used to fit the triangle
    ultimate_
        The ultimate losses per the method
    ibnr_
        The IBNR per the method
    full_expectation_
        The ultimates back-filled to each development period in **X** replacing
        the known data
    full_triangle_
        The ultimates back-filled to each development period in **X** retaining
        the known data

    Examples
    --------
    Comparing the ultimates for a company using company LDFs vs industry LDFs

    >>> clrd = cl.load_dataset('clrd')['CumPaidLoss']
    >>> clrd = clrd[clrd['LOB'] == 'wkcomp']
    >>> industry = clrd.sum()
    >>> allstate_industry_cl = cl.Chainladder().fit(industry).predict(clrd).ultimate_.loc['Allstate Ins Co Grp']
    >>> allstate_company_cl = cl.Chainladder().fit(clrd.loc['Allstate Ins Co Grp']).ultimate_
    >>> (allstate_industry_cl - allstate_company_cl).rename(development='Industry to Company LDF Diff')
          Industry to Company LDF Diff
    1988                      0.000000
    1989                   -202.830662
    1990                  -4400.765402
    1991                  -5781.419855
    1992                  -6286.251679
    1993                  -4792.896607
    1994                  -6652.981043
    1995                  -8327.246649
    1996                  -7169.608047
    1997                   -273.269090
    """

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Data to which the model will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, sample_weight)
        obj = copy.deepcopy(self)
        self.ultimate_ = self._get_ultimate_(X, sample_weight, obj)
        self.full_triangle_ = self._get_full_triangle_()
        return self

    def predict(self, X, sample_weight=None):
        obj = super().predict(X, sample_weight)
        obj.ultimate_ = obj._get_ultimate_(X, sample_weight, obj)
        obj.full_triangle_ = obj._get_full_triangle_()
        return obj

    def _get_ultimate_(self, X, sample_weight, obj):
        development = -1
        xp = cp.get_array_module(self.X_.values)
        nans = self.X_._nan_triangle()
        obj = copy.copy(self.X_)
        obj.values = xp.repeat(self.X_.latest_diagonal.values,
                               self.cdf_.shape[development], development)
        cdf = self.cdf_.values[..., :nans.shape[development]]
        obj_tri = obj.values[..., :nans.shape[development]]
        if xp.unique(cdf, axis=2).shape[2] == 1 and \
           len(obj.odims) != cdf.shape[2]:
            cdf = xp.repeat(xp.unique(cdf, axis=2), len(obj.odims), axis=2)
        obj.values = (cdf*obj_tri)*nans
        obj = obj.latest_diagonal
        obj.ddims = np.array([None])
        obj.valuation = obj._valuation_triangle()
        obj._set_slicers()
        return obj
