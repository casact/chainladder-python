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

        tr = cl.load_sample('ukmotor')
        apriori = cl.Chainladder().fit(tr).ultimate_ * 0 + 14000

    With ``n_iters=1`` Benktander reproduces Bornhuetter-Ferguson exactly.

    .. testcode::

        print(
            cl.Benktander(apriori=1.0, n_iters=1).fit(
                tr, sample_weight=apriori
            ).ultimate_
        )

    .. testoutput::

                      2261
        2007  12690.000000
        2008  13121.098503
        2009  14028.278620
        2010  13272.048822
        2011  13911.968891
        2012  15614.145287
        2013  16029.501746

    Increasing ``n_iters`` pulls the immature origins toward the chainladder
    estimate. The 2013 origin shows this most: ``16029`` at ``n_iters=1``,
    rising to ``19110`` at ``n_iters=4`` and approaching the chainladder
    ultimate of ``20680``.

    .. testcode::

        print(
            cl.Benktander(apriori=1.0, n_iters=4).fit(
                tr, sample_weight=apriori
            ).ultimate_
        )

    .. testoutput::

                      2261
        2007  12690.000000
        2008  13096.902490
        2009  14030.535854
        2010  13138.365841
        2011  13880.984774
        2012  16719.527550
        2013  19110.806503
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

            tr = cl.load_sample('ukmotor')
            apriori = cl.Chainladder().fit(tr).ultimate_ * 0 + 14000
            print(cl.Benktander(apriori=1.0, n_iters=2).fit(tr, sample_weight=apriori))

        .. testoutput::

            Benktander(n_iters=2)
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

            print(model.predict(tr, sample_weight=apriori).ultimate_)

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
