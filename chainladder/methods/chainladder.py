# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
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
        self.ultimate_ = self._get_ultimate(self.X_)
        self.process_variance_ = self._include_process_variance()
        return self

    def _get_ultimate(self, X, sample_weight=None):
        """ Private method that uses CDFs to obtain an ultimate vector """
        xp = cp.get_array_module(X.values)
        ultimate = copy.deepcopy(X)
        cdf = self._align_cdf(ultimate)
        ultimate = X.latest_diagonal * cdf
        return self._set_ult_attr(ultimate)
