# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.methods import Benktander


class BornhuetterFerguson(Benktander):
    """The deterministic Bornhuetter Ferguson IBNR model

    Parameters
    ----------
    apriori: float, optional (default=1.0)
        Multiplier for the `sample_weight` used in the Bornhuetter Ferguson
        method. If `sample_weight` is already an apriori measure of ultimate,
        then use 1.0. 
        The recommended pratice is to seperate the model parameter assumption 
        and data apart.
        For example, if the apriori s 80% of premium, it is recommended to set 
        the aprior as 0.8 and leave the premium data in `sample_weight` argument 
        unmodified.
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
    Bornhuetter-Ferguson requires an apriori expected ultimate per origin,
    supplied through ``sample_weight``.

    A common idiom for building a flat per-origin apriori is to take any
    same-shape Triangle, zero it out, and add the desired value. Here is an example.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        raa = cl.load_sample("raa")
        premium = raa.latest_diagonal * 0 + 40_000  # zero out and add 40,000 to each origin
        ibnr = cl.BornhuetterFerguson(apriori=0.7).fit(X=raa, sample_weight=premium).ibnr_
        print(ibnr)

    .. testoutput::

                      2261
        1981           NaN
        1982    255.707763
        1983    717.772687
        1984   1596.061515
        1985   2658.738155
        1986   5239.441491
        1987   8574.335344
        1988  12714.889984
        1989  18585.219714
        1990  24861.068855

    One might be tempted to set never set the aprior and modify the sample_weight directly, and they will result in the same answer, but this is not the recommended practice. It not only add confusion, but it alos mixes the model parameter assumption and data together.

    .. testcode::

        raa = cl.load_sample("raa")
        premium = raa.latest_diagonal * 0 + 40_000 * 0.7  # premium is modified by 70%
        ibnr = cl.BornhuetterFerguson().fit(X=raa, sample_weight=premium).ibnr_
        print(ibnr)

    .. testoutput::

                      2261
        1981           NaN
        1982    255.707763
        1983    717.772687
        1984   1596.061515
        1985   2658.738155
        1986   5239.441491
        1987   8574.335344
        1988  12714.889984
        1989  18585.219714
        1990  24861.068855
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

        Examples
        --------
        Fit returns the estimator itself, with ``ultimate_`` populated.

        .. testsetup::
        
            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            apriori = cl.Chainladder().fit(tr).ultimate_ * 0 + 14000
            model = cl.BornhuetterFerguson(apriori=1.0).fit(tr, sample_weight=apriori)
            print(model)

        .. testoutput::

            BornhuetterFerguson()
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

        Examples
        --------
        Fit on a prior-period view of the data, then apply the model to the
        current Triangle and a refreshed apriori.

        .. testsetup::
        
            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            tr_prior = tr[tr.valuation < tr.valuation_date]
            apriori_prior = cl.Chainladder().fit(tr_prior).ultimate_ * 0 + 14000
            apriori = cl.Chainladder().fit(tr).ultimate_ * 0 + 14000
            model = cl.BornhuetterFerguson(apriori=1.0).fit(
                tr_prior, sample_weight=apriori_prior
            )
            ultimate = model.predict(tr, sample_weight=apriori).ultimate_
            print(ultimate)

        .. testoutput::

                          2261
            2007  12690.000000
            2008  12746.000000
            2009  13658.425101
            2010  12883.599658
            2011  13610.582796
            2012  15360.020613
            2013  15893.717063
        """
        return super().predict(X, sample_weight)
