# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
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
        Number of iterations to use in the Benktander model.
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

    def __init__(self, apriori=1.0, n_iters=1, apriori_sigma=0, random_state=None):
        self.apriori = apriori
        self.n_iters = n_iters
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

        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        super().fit(X, y, sample_weight)
        if hasattr(X, "_get_process_variance"):
            if X.shape[0] != sample_weight.shape[0]:
                sample_weight = sample_weight.broadcast_axis("index", X.index)
        self.ultimate_ = self._get_ultimate(self.X_, self.sample_weight_)
        self.process_variance_ = self._include_process_variance()
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
            Loss data with Benktander ultimate applied
        """
        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        return super().predict(X, sample_weight)

    def _get_ultimate(self, X, sample_weight):
        xp = X.get_array_module()
        from chainladder.utils.utility_functions import num_to_nan

        ultimate = X.copy()
        # Apriori
        if self.apriori_sigma != 0:
            random_state = xp.random.RandomState(self.random_state)
            apriori = random_state.normal(self.apriori, self.apriori_sigma, X.shape[0])
            apriori = apriori.reshape(X.shape[0], -1)[..., None, None]
            apriori = sample_weight.values * apriori
        else:
            apriori = sample_weight.values * self.apriori
        # Benktander formula -> Triangle
        cdf = self._align_cdf(ultimate, sample_weight)
        cdf = (1 - 1 / num_to_nan(cdf))[None]
        exponents = xp.arange(self.n_iters + 1)
        exponents = xp.reshape(exponents, tuple([len(exponents)] + [1] * 4))
        cdf = cdf ** (((cdf + 1e-16) / (cdf + 1e-16) * exponents))
        cdf = xp.nan_to_num(cdf)
        ultimate.values = xp.sum(cdf[:-1, ...], 0) * xp.nan_to_num(
            X.latest_diagonal.values
        ) + cdf[-1, ...] * xp.nan_to_num(apriori)
        return self._set_ult_attr(ultimate)
