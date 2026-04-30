# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.methods import Benktander


class BornhuetterFerguson(Benktander):
    """The deterministic Bornhuetter Ferguson IBNR model

    Parameters
    ----------
    apriori: float, optional (default=1.0)
        Multiplier for the sample_weight used in the Bornhuetter Ferguson
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
    Bornhuetter-Ferguson requires an apriori expected ultimate per origin,
    supplied through ``sample_weight``. ``sample_weight`` must be a
    chainladder Triangle aligned with ``X``, not a scalar; passing
    ``sample_weight=14000`` would raise ``AttributeError`` because the model
    accesses ``.shape``.

    A common idiom for building a flat per-origin apriori is to take any
    same-shape Triangle, zero it out, and add the desired value. Below uses
    the chainladder ultimate as the shape donor.

    .. testsetup:

        import chainladder as cl

    .. testcode:

        tr = cl.load_sample('ukmotor')
        cl_ult = cl.Chainladder().fit(tr).ultimate_
        apriori = cl_ult * 0 + float(cl_ult.sum()) / 7
        print(apriori)

    .. testoutput:

                      2261
        2007  14903.967562
        2008  14903.967562
        2009  14903.967562
        2010  14903.967562
        2011  14903.967562
        2012  14903.967562
        2013  14903.967562

    Fit with that apriori. The BF ultimates pull the immature origins toward
    the apriori while leaving mature origins close to chainladder.

    .. testcode:

        model = cl.BornhuetterFerguson(apriori=1.0).fit(tr, sample_weight=apriori)
        print(model.ultimate_)

    .. testoutput:

                      2261
        2007  12690.000000
        2008  13145.318280
        2009  14095.125641
        2010  13412.748068
        2011  14150.549749
        2012  15999.244850
        2013  16658.824705
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

        >>> tr = cl.load_sample('ukmotor')
        >>> apriori = cl.Chainladder().fit(tr).ultimate_ * 0 + 14000
        >>> cl.BornhuetterFerguson(apriori=1.0).fit(tr, sample_weight=apriori)
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

        >>> tr = cl.load_sample('ukmotor')
        >>> tr_prior = tr[tr.valuation < tr.valuation_date]
        >>> apriori_prior = cl.Chainladder().fit(tr_prior).ultimate_ * 0 + 14000
        >>> apriori = cl.Chainladder().fit(tr).ultimate_ * 0 + 14000
        >>> model = cl.BornhuetterFerguson(apriori=1.0).fit(
        ...     tr_prior, sample_weight=apriori_prior
        ... )
        >>> model.predict(tr, sample_weight=apriori).ultimate_
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
