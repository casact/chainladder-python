# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.methods import MethodBase
from chainladder.development import DevelopmentConstant
import numpy as np


class CaseOutstanding(MethodBase):
    """The CaseOutstanding IBNR model

    Parameters
    ----------
    paid_pattern: DevelopmentConstant()
        The payment pattern to be used to derive the development pattern for
        oustanding case reserves.
    reported_pattern: DevelopmentConstant()
        The reported pattern to be used to derive the development pattern for
        oustanding case reserves.

    Attributes
    ----------
    ultimate_: Triangle
        The ultimate losses per the method
    ibnr_: Triangle
        The IBNR per the method
    """

    def __init__(self, paid_pattern=None, reported_pattern=None):
        print("in CaseOutstanding init")
        self.paid_pattern = None
        self.reported_pattern = None

    #
    # def fit(self, X, y=None, sample_weight=None):
    #     """Applies the Benktander technique to triangle **X**
    #
    #     Parameters
    #     ----------
    #     X: Triangle
    #         Loss data to which the model will be applied.
    #     y: None
    #         Ignored
    #     sample_weight: Triangle
    #         Required exposure to be used in the calculation.
    #
    #     Returns
    #     -------
    #     self: object
    #         Returns the instance itself.
    #     """
    #     if sample_weight is None:
    #         raise ValueError("sample_weight is required.")
    #     super().fit(X, y, sample_weight)
    #     self.expectation_ = self._get_benktander_aprioris(X, sample_weight)
    #     self.ultimate_ = self._get_ultimate(self.X_, self.expectation_)
    #     self.process_variance_ = self._include_process_variance()
    #     return self
    #
    # def predict(self, X, sample_weight=None):
    #     """Predicts the Benktander ultimate on a new triangle **X**
    #
    #     Parameters
    #     ----------
    #     X: Triangle
    #         Loss data to which the model will be applied.
    #     sample_weight: Triangle
    #         Required exposure to be used in the calculation.
    #
    #     Returns
    #     -------
    #     X_new: Triangle
    #         Loss data with Benktander ultimate applied
    #     """
    #     X_new = super().predict(X, sample_weight)
    #     X_new.expectation_ = self._get_benktander_aprioris(X, sample_weight)
    #     X_new.ultimate_ = self._get_ultimate(X_new, X_new.expectation_)
    #     X_new.n_iters = self.n_iters
    #     X_new.apriori = self.apriori
    #     return X_new
    #
    # def _get_benktander_aprioris(self, X, sample_weight):
    #     """Private method to establish Benktander Apriori"""
    #     xp = X.get_array_module()
    #     if self.apriori_sigma != 0:
    #         random_state = xp.random.RandomState(self.random_state)
    #         apriori = random_state.normal(self.apriori, self.apriori_sigma, X.shape[0])
    #         apriori = apriori.reshape(X.shape[0], -1)[..., None, None]
    #         apriori = sample_weight * apriori
    #         apriori.kdims = X.kdims
    #         apriori.key_labels = X.key_labels
    #     else:
    #         apriori = sample_weight * self.apriori
    #     apriori.columns = sample_weight.columns
    #     return apriori
    #
    # def _get_ultimate(self, X, expectation):
    #     from chainladder.utils.utility_functions import num_to_nan
    #
    #     if X.is_cumulative == False:
    #         ld = X.sum("development")
    #         ultimate = ld.val_to_dev()
    #     else:
    #         ld = X.latest_diagonal
    #         ultimate = X.copy()
    #     cdf = self._align_cdf(ultimate.val_to_dev(), expectation)
    #     backend = cdf.array_backend
    #     xp = cdf.get_array_module()
    #     cdf = cdf.sort_index()
    #     ld = ld.sort_index()
    #     expectation = expectation.sort_index()
    #     ultimate = ultimate.sort_index()
    #     cdf = (1 - 1 / num_to_nan(cdf.values))[None]
    #     exponents = xp.arange(self.n_iters + 1)
    #     exponents = xp.reshape(exponents, tuple([len(exponents)] + [1] * 4))
    #     cdf = cdf ** (((cdf + 1e-16) / (cdf + 1e-16) * exponents))
    #     cdf = xp.nan_to_num(cdf)
    #     a = xp.sum(cdf[:-1, ...], 0) * xp.nan_to_num(ld.set_backend(backend).values)
    #     b = cdf[-1, ...] * xp.nan_to_num(expectation.set_backend(backend).values)
    #     ultimate.values = num_to_nan(a + b)
    #     ultimate.array_backend = backend
    #     ultimate.ddims = self.cdf_.ddims[: ultimate.shape[-1]]
    #     return self._set_ult_attr(ultimate)
