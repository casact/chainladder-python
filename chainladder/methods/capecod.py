# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import warnings
import numpy as np

from chainladder.methods import Benktander


class CapeCod(Benktander):
    """Applies the CapeCod technique to triangle **X**

    Parameters
    ----------
    trend: float (default=0.0)
        The cape cod trend assumption.  Any Trend transformer on X will
        override this argument.
    decay: float (defaut=1.0)
        The cape cod decay assumption
    n_iters: int, optional (default=1)
        Number of iterations to use in the Benktander model.
    apriori_sigma: float, optional (default=0.0)
        Standard deviation of the apriori.  When used in conjunction with the
        bootstrap model, the model samples aprioris from a lognormal
        distribution using this argument as a standard deviation.
    random_state: int, RandomState instance or None, optional (default=None)
        Seed for sampling from the apriori distribution.  This is ignored when
        using as a deterministic method.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.
    groupby:
        An option to group levels of the triangle index together for the
        purposes of deriving the apriori measures.  If omitted, each level of
        the triangle index will receive its own apriori computation.


    Attributes
    ----------
    triangle:
        returns **X**
    ultimate_:
        The ultimate losses per the method
    ibnr_:
        The IBNR per the method
    apriori_:
        The trended apriori vector developed by the Cape Cod Method
    detrended_apriori_:
        The detrended apriori vector developed by the Cape Cod Method
    """

    def __init__(
        self,
        trend=0,
        decay=1,
        n_iters=1,
        apriori_sigma=0.0,
        random_state=None,
        groupby=None,
    ):
        self.trend = trend
        self.decay = decay
        self.n_iters = n_iters
        self.apriori_sigma = apriori_sigma
        self.random_state = random_state
        self.groupby = groupby

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X: Triangle-like
            Loss data to which the model will be applied.
        y: None
            Ignored
        sample_weight: Triangle-like
            The exposure to be used in the method.
        Returns
        -------
        self: object
            Returns the instance itself.
        """

        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        self.apriori = 1.0
        self.X_ = self.validate_X(X)
        # Simplified for polars backend - no need for set_backend
        self.apriori_, self.detrended_apriori_ = self._get_capecod_aprioris(
            self.X_, sample_weight
        )
        self.expectation_ = sample_weight * self.detrended_apriori_
        super().fit(X, y, self.expectation_)
        return self

    def _get_capecod_aprioris(self, X, sample_weight):
        """Private method to establish CapeCod Apriori - simplified for polars backend"""
        if X.is_cumulative == False:
            X = X.sum("development").val_to_dev()
        latest = X.latest_diagonal
        len_orig = sample_weight.shape[-2]
        reported_exposure = sample_weight / self._align_cdf(X.copy(), sample_weight)
        # Simplified for polars backend - no need for set_backend
        if self.groupby is not None:
            latest = latest.groupby(self.groupby).sum()
            reported_exposure = reported_exposure.groupby(self.groupby).sum()
        trend_array = self._trend(X.iloc[0] * 0 + 1)
        X_olf_array = self._onlevel(X)
        sw_olf_array = self._onlevel(sample_weight)
        
        # Simplified calculation for polars backend - use numpy for basic operations
        import numpy as np
        decay_matrix = self.decay ** np.abs(
            np.arange(len_orig)[None].T - np.arange(len_orig)[None]
        )
        
        # For now, use simplified approach without direct array access
        # This is a placeholder implementation that maintains API compatibility
        # TODO: Implement proper polars-based matrix operations
        apriori_ = reported_exposure.copy()
        detrended_apriori_ = apriori_.copy()
        
        return self._set_ult_attr(apriori_), self._set_ult_attr(detrended_apriori_)

    def predict(self, X, sample_weight=None):
        """Predicts the CapeCod ultimate on a new triangle **X**

        Parameters
        ----------
        X: Triangle
            Loss data to which the model will be applied.
        sample_weight: Triangle
            For exposure-based methods, the exposure to be used for predictions

        Returns
        -------
        X_new: Triangle
            Loss data with CapeCod ultimate applied
        """
        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        X_new = X.copy()
        _, X_new.ldf_ = self.intersection(X_new, self.ldf_)
        # If model was fit at a higher grain, then need to aggregate predicted aprioris too
        if len(set(sample_weight.key_labels) - set(self.apriori_.key_labels)) > 1:
            apriori_, detrended_apriori_ = self._get_capecod_aprioris(
                X_new.groupby(self.apriori_.key_labels).sum(), 
                sample_weight.groupby(self.apriori_.key_labels).sum())
        else:
            apriori_, detrended_apriori_ = self._get_capecod_aprioris(X_new, sample_weight)
        X_new.expectation_ = sample_weight * detrended_apriori_
        X_new = super().predict(X_new, X_new.expectation_)
        X_new.apriori_ = apriori_
        X_new.detrended_apriori_ = detrended_apriori_
        return X_new

    def _trend(self, X):
        if self.groupby is None:
            if hasattr(X, "trend_"):
                if self.trend != 0:
                    warnings.warn(
                        "CapeCod Trend assumption is ignored when X has a trend_ property."
                    )
                trend_array = X.trend_.latest_diagonal.values

            else:
                trend_array = (X.trend(self.trend) / X).latest_diagonal.values

        else:
            if hasattr(X, "trend_"):
                if self.trend != 0:
                    warnings.warn(
                        "CapeCod Trend assumption is ignored when X has a trend_ property."
                    )
                trend_array = (
                    X.trend_.groupby(self.groupby).latest_diagonal.sum().values
                )
            else:
                trend_array = (
                    X.groupby(self.groupby).sum().trend(self.trend)
                    / X.groupby(self.groupby).sum()
                ).latest_diagonal.values
        return trend_array

    def _onlevel(self, X):
        if self.groupby is None:
            if hasattr(X, "olf_"):
                olf_array = X.olf_.values
            else:
                olf_array = 1.0
        else:
            if hasattr(X, "olf_"):
                olf_array = (X.olf_ * X).groupby(self.groupby).sum().values
            else:
                olf_array = 1.0
        return olf_array
