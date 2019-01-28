"""
:ref:`chainladder.methods<methods>`.Benktander
===============================================

:ref:`Benktander<benktander>` is a generalization of both the traditional
chainladder (CL) and the Bornhutter-Ferguson method.  It is also known as the
interated BF method.
The generalized formula is:

:math:`\\sum_{k=0}^{n-1}(1-\\frac{1}{CDF}) + Apriori\\times (1-\\frac{1}{CDF})^{n}`

`n=1` yields the traditional BF method, and when `n` is sufficiently large, the
method converges to the traditional CL method.

Mack noted the Benktander method is found to have almost always a smaller mean
squared error than the other two methods and to be almost as precise as an exact
Bayesian procedure.
"""

import numpy as np
import copy
from chainladder.methods import MethodBase


class Benktander(MethodBase):
    """ The Benktander (or iterated Bornhuetter-Ferguson) IBNR model

    Parameters
    ----------
    apriori : float, optional (default=1.0)
        Multiplier for the sample_weight used in the Benktander method
        method. If sample_weight is already an apriori measure of ultimate,
        then use 1.0
    n_iters : int, optional (default=1)
        Multiplier for the sample_weight used in the Bornhuetter Ferguson
        method. If sample_weight is already an apriori measure of ultimate,
        then use 1.0

    Attributes
    ----------
    ultimate_ : Triangle
        The ultimate losses per the method
    ibnr_ : Triangle
        The IBNR per the method

    References
    ----------
    .. [2] Benktander, G. (1976) An Approach to Credibility in Calculating IBNR for Casualty Excess Reinsurance. In The Actuarial Review, April 1976, p.7
    """
    def __init__(self, apriori=1.0, n_iters=1):
        self.apriori = apriori
        self.n_iters = n_iters

    def fit(self, X, y=None, sample_weight=None):
        """Applies the Benktander technique to triangle **X**

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

        super().fit(X, y, sample_weight)
        self.sample_weight_ = sample_weight
        latest = self.X_.latest_diagonal.triangle
        apriori = sample_weight.triangle * self.apriori
        obj = copy.deepcopy(self.X_)
        obj.triangle = self.X_.cdf_.triangle * (obj.triangle*0+1)
        cdf = obj.latest_diagonal.triangle
        cdf = np.expand_dims(1-1/cdf, 0)
        exponents = np.arange(self.n_iters+1)
        exponents = np.reshape(exponents, tuple([len(exponents)]+[1]*4))
        cdf = cdf**exponents
        obj.triangle = np.sum(cdf[:-1, ...], 0)*latest+cdf[-1, ...]*apriori
        obj.ddims = ['Ultimate']
        self.ultimate_ = obj
        return self
