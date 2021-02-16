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
    apriori_sigma : float, optional (default=0.0)
        Standard deviation of the apriori.  When used in conjunction with the
        bootstrap model, the model samples aprioris from a lognormal distribution
        using this argument as a standard deviation.
    random_state : int, RandomState instance or None, optional (default=None)
        Seed for sampling from the apriori distribution.  This is ignored when
        using as a deterministic method.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    ultimate_ : Triangle
        The ultimate losses per the method
    ibnr_ : Triangle
        The IBNR per the method
    """

    def __init__(self, apriori=1.0, apriori_sigma=0.0, random_state=None):
        self.apriori = apriori
        self.apriori_sigma = apriori_sigma
        self.random_state = random_state

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

    def predict(self, X, sample_weight=None):
        """Predicts the Bornhuetter-Ferguson ultimate on a new triangle **X**

        Parameters
        ----------
        X : Triangle
            Loss data to which the model will be applied.
        sample_weight : Triangle
            Required exposure to be used in the calculation.

        Returns
        -------
        X_new: Triangle
            Loss data with Bornhuetter-Ferguson ultimate applied
        """
        return super().predict(X, sample_weight)
