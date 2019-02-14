"""
:ref:`chainladder.methods<methods>`.Chainladder
===============================================

:ref:`Chainladder<chainladder>` is the most basic and fundamental method.
The IBNR model is entirely described by the LDFs
"""

import numpy as np
import pandas as pd
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
        self.full_triangle_ = self._get_full_triangle_()
        return self

    @property
    def ultimate_(self):
        obj = copy.deepcopy(self.X_)
        obj.triangle = np.repeat(self.X_.latest_diagonal.triangle,
                                 self.cdf_.shape[3], 3)
        obj.triangle = (self.cdf_.triangle*obj.triangle)*self.X_.nan_triangle()
        obj = obj.latest_diagonal
        obj.ddims = ['Ultimate']
        obj.valuation = pd.DatetimeIndex([pd.to_datetime('2262-04-11')]*obj.shape[-2])
        return obj
