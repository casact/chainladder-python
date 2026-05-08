# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.methods import Benktander


class ExpectedLoss(Benktander):
    """The deterministic Expected Loss IBNR model, it ignores all data in the 
    triangle, and only uses the sample_weight modified by the apriori to 
    calculate the ultimate losses.

    Parameters
    ----------
    apriori: float, optional (default=1.0)
        Multiplier for the sample_weight used in the Expected Loss
        method. If sample_weight is already an apriori measure of ultimate,
        then use 1.0
    apriori_sigma: float, optional (default=0.0)
        Standard deviation of the apriori.  When used in conjunction with the
        bootstrap model, the model samples aprioris from a lognormal distribution
        using this argument as a standard deviation.
    random_state: int, RandomState instance or None, optional (default=None)
        Seed for sampling from the apriori distribution.  This is ignored when
        using as a deterministic method.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    ultimate_: Triangle
        The ultimate losses per the method
    ibnr_: Triangle
        The IBNR per the method

    Examples
    --------

    .. testsetup::

        import chainladder as cl

    .. testcode::

        xyz = cl.load_sample("xyz")
        
        ibnr = (
            cl.ExpectedLoss()
            .fit(X=xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal)
            .ibnr_
        )
        print(ibnr)

    .. testoutput::
        
                 2261
        1998   4178.0
        1999   6683.0
        2000   8218.0
        2001  11481.0
        2002  16746.0
        2003  29855.0
        2004  46511.0
        2005  98125.0
        2006  84759.0
        2007  50573.0
        2008  44388.0

    We can specify the apriori as a percentage of the premium.

    .. testcode::

        xyz = cl.load_sample("xyz")

        ibnr = (
            cl.ExpectedLoss(apriori=0.9)
            .fit(X=xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal)
            .ibnr_
        )
        print(ibnr)

    .. testoutput::
        
                 2261
        1998   2178.0
        1999   3533.0
        2000   3718.0
        2001   6481.0
        2002  10627.7
        2003  22937.5
        2004  36578.8
        2005  84309.9
        2006  74001.2
        2007  44329.2
        2008  39608.3
    """

    def __init__(self, apriori=1.0, apriori_sigma=0.0, random_state=None):
        self.apriori = apriori
        self.apriori_sigma = apriori_sigma
        self.random_state = random_state

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
        self.n_iters = 0
        super().fit(X, y, sample_weight)
        return self

    def predict(self, X, sample_weight=None):
        """Predicts the Benktander ultimate on a new triangle **X**

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

