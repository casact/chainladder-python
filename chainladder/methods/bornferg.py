# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.methods import Benktander


class BornhuetterFerguson(Benktander):
    """The deterministic Bornhuetter Ferguson IBNR model

    Parameters
    ----------
    apriori : float, optional (default=1.0)
        Multiplier for the sample_weight used in the Bornhuetter Ferguson
        method. If sample_weight is already an apriori measure of ultimate,
        then use 1.0

    Attributes
    ----------
    ultimate_ : Triangle
        The ultimate losses per the method
    ibnr_ : Triangle
        The IBNR per the method

    Examples
    --------
    Smoothing chainladder ultimates by using them as apriori figures in the
    Bornhuetter Ferguson method.

    >>> raa = cl.load_dataset('RAA')
    >>> cl_ult = cl.Chainladder().fit(raa).ultimate_ # Chainladder Ultimate
    >>> apriori = cl_ult*0+(cl_ult.sum()/10)[0] # Mean Chainladder Ultimate
    >>> cl.BornhuetterFerguson(apriori=1).fit(raa, sample_weight=apriori).ultimate_
              Ultimate
    1981  18834.000000
    1982  16898.632172
    1983  24012.333266
    1984  28281.843524
    1985  28203.700714
    1986  19840.005163
    1987  18840.362337
    1988  22789.948877
    1989  19541.155136
    1990  20986.022826

    References
    ----------
    .. [1] Bornhuetter, R. and Ferguson, R. (1972) The Actuary and IBNR. In
           Proceedings of the Casualty Actuarial Society, Vol. LIX, 181 - 195
    """
    def __init__(self, apriori=1.0):
        self.apriori = apriori

    def fit(self, X, y=None, sample_weight=None):
        """Applies the Bornhuetter-Ferguson technique to triangle **X**

        Parameters
        ----------
        X : Triangle
            Loss data to which the model will be applied.
        y : None
            Ignored
        sample_weight : Triangle
            Required exposure to be used in the calculation.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_iters = 1
        super().fit(X, y, sample_weight)
        return self
