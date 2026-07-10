# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.methods import MethodBase
import numpy as np


class Benktander(MethodBase):
    """The Benktander (or iterated Bornhuetter-Ferguson) IBNR model

    Parameters
    ----------
    apriori: float, optional (default=1.0)
        Multiplier for the sample_weight used in the Benktander method
        method. If sample_weight is already an apriori measure of ultimate,
        then use 1.0
    n_iters: int, optional (default=1)
        Number of iterations to use in the Benktander model.
        When n_iters=1, the result is equivalent to the BornhuetterFerguson method.
        When n_iters>>1, the result converges to the traditional Chainladder model.
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
    Benktander is the iterated Bornhuetter-Ferguson model. Like BF, it
    requires a per-origin apriori expected ultimate supplied through
    ``sample_weight``. The ``n_iters`` parameter interpolates between BF
    (``n_iters=1``) and chainladder (``n_iters`` large): each additional
    iteration shifts the ultimate further toward the chainladder estimate.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        xyz = cl.load_sample("xyz")
        ibnr = cl.Benktander().fit(X=xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal).ibnr_
        print(ibnr)

    .. testoutput::

                      2261
        1998           NaN
        1999    115.472127
        2000    914.033812
        2001   2432.394513
        2002   6037.026677
        2003  13928.934651
        2004  33925.451475
        2005  69724.761575
        2006  73410.593920
        2007  52977.560411
        2008  45873.769490

    When `n_iters=1`, the model is exactly the same as the BornhuetterFerguson model.

    .. testcode::

        xyz = cl.load_sample("xyz")
        bk_ibnr = (
            cl.Benktander(n_iters=1)
            .fit(X=xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal)
            .ibnr_
        )
        bf_ibnr = (
            cl.BornhuetterFerguson()
            .fit(X=xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal)
            .ibnr_
        )
        print(bk_ibnr - bf_ibnr)

    .. testoutput::

              2261
        1998   NaN
        1999   NaN
        2000   NaN
        2001   NaN
        2002   NaN
        2003   NaN
        2004   NaN
        2005   NaN
        2006   NaN
        2007   NaN
        2008   NaN

    When `n_iters>>1`, the model converges to the traditional Chainladder model.

    .. testcode::

        xyz = cl.load_sample("xyz")
        bk_ibnr = cl.Benktander(n_iters=1000).fit(X=xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal).ibnr_
        cl_ibnr = cl.Chainladder().fit(xyz["Paid"]).ibnr_
        print(bk_ibnr - cl_ibnr)

    .. testoutput::

                      2261
        1998           NaN
        1999           NaN
        2000           NaN
        2001  1.455192e-11
        2002 -7.275958e-12
        2003  7.275958e-12
        2004  1.455192e-11
        2005 -1.455192e-11
        2006  2.910383e-11
        2007 -5.820766e-11
        2008 -7.275958e-11
    """

    def __init__(self, apriori=1.0, n_iters=1, apriori_sigma=0, random_state=None):
        self.apriori = apriori
        self.n_iters = n_iters
        self.apriori_sigma = apriori_sigma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """Applies the Benktander technique to triangle **X**

        Parameters
        ----------
        X: Triangle
            Loss data to which the model will be applied.
        y: None
            Ignored
        sample_weight: Triangle
            Required exposure to be used in the calculation.

        Returns
        -------
        self: object
            Returns the instance itself.

        Examples
        --------
        Fit returns the estimator itself, with ``ultimate_`` populated. The
        repr shows non-default parameters.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            xyz = cl.load_sample("xyz")
            ultimate = (
                cl.Benktander(apriori=1, n_iters=2)
                .fit(X=xyz["Paid"], sample_weight=xyz["Premium"].latest_diagonal)
                .ultimate_
            )
            print(ultimate)

        .. testoutput::

                          2261
            1998  15822.000000
            1999  24908.397003
            2000  37547.676656
            2001  40511.198946
            2002  49417.354765
            2003  50042.095135
            2004  82437.601111
            2005  95417.171135
            2006  88485.508416
            2007  66882.788227
            2008  50708.755370

        """
        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        super().fit(X, y, sample_weight)
        self.expectation_ = self._get_benktander_aprioris(X, sample_weight)
        self.ultimate_ = self._get_ultimate(self.X_, self.expectation_)
        self.process_variance_ = self._include_process_variance()
        return self

    def predict(self, X, sample_weight=None):
        """Predicts the Benktander ultimate on a new triangle **X**

        Parameters
        ----------
        X: Triangle
            Loss data to which the model will be applied.
        sample_weight: Triangle
            Required exposure to be used in the calculation.

        Returns
        -------
        X_new: Triangle
            Loss data with Benktander ultimate applied

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
            model = cl.Benktander(apriori=1.0, n_iters=2).fit(
                tr_prior, sample_weight=apriori_prior
            )
            ultimate = model.predict(tr, sample_weight=apriori).ultimate_
            print(ultimate)

        .. testoutput::

                          2261
            2007  12690.000000
            2008  12746.000000
            2009  13642.189922
            2010  12740.812082
            2011  13516.188545
            2012  15914.716737
            2013  17193.715555
        """
        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        X_new = super().predict(X, sample_weight)
        X_new.expectation_ = self._get_benktander_aprioris(X, sample_weight)
        X_new.ultimate_ = self._get_ultimate(X_new, X_new.expectation_)
        X_new.n_iters = self.n_iters
        X_new.apriori = self.apriori
        return X_new

    def _get_benktander_aprioris(self, X, sample_weight):
        """Private method to establish Benktander Apriori"""
        xp = X.get_array_module()
        if self.apriori_sigma != 0:
            random_state = xp.random.RandomState(self.random_state)
            apriori = random_state.normal(self.apriori, self.apriori_sigma, X.shape[0])
            apriori = apriori.reshape(X.shape[0], -1)[..., None, None]
            apriori = sample_weight * apriori
            apriori.kdims = X.kdims
            apriori.key_labels = X.key_labels
        else:
            apriori = sample_weight * self.apriori
        apriori.columns = sample_weight.columns
        return apriori

    def _get_ultimate(self, X, expectation):
        from chainladder.utils.utility_functions import num_to_nan

        ld = X.incr_to_cum().latest_diagonal
        ultimate = X.incr_to_cum().copy()
        cdf = self._align_cdf(ultimate.val_to_dev(), expectation)
        backend = cdf.array_backend
        xp = cdf.get_array_module()
        cdf = cdf.sort_index()
        ld = ld.sort_index()
        expectation = expectation.sort_index()
        ultimate = ultimate.sort_index()
        cdf = (1 - 1 / num_to_nan(cdf.values))[None]
        exponents = xp.arange(self.n_iters + 1)
        exponents = xp.reshape(exponents, tuple([len(exponents)] + [1] * 4))
        cdf = cdf ** (((cdf + 1e-16) / (cdf + 1e-16) * exponents))
        cdf = xp.nan_to_num(cdf)
        a = xp.sum(cdf[:-1, ...], 0) * xp.nan_to_num(ld.set_backend(backend).values)
        b = cdf[-1, ...] * xp.nan_to_num(expectation.set_backend(backend).values)
        ultimate.values = num_to_nan(a + b)
        ultimate.array_backend = backend
        ultimate.ddims = self.cdf_.ddims[: ultimate.shape[-1]]
        return self._set_ult_attr(ultimate)
