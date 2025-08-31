# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development
from chainladder.legacy.io import EstimatorIO
from chainladder.legacy.common import Common


class MethodBase(BaseEstimator, EstimatorIO, Common):

    _estimator_type = "chainladder"

    def validate_X(self, X):
        obj = X.copy()
        if not hasattr(obj, "ldf_"):
            obj = Development().fit_transform(obj)
        # Simplified tail check for polars-based Triangle    
        # if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
        #     obj = TailConstant().fit_transform(obj)
        return obj.val_to_dev()

    def _align_cdf(self, X, sample_weight=None):
        """ Simplified _align_cdf for polars backend """
        # For basic functionality, return the CDF directly
        # This is a simplified implementation that avoids complex arithmetic
        if hasattr(X, 'cdf_') and X.cdf_ is not None:
            cdf = X.cdf_.iloc[..., : X.shape[-1]] if hasattr(X.cdf_, 'iloc') else X.cdf_
            return cdf.latest_diagonal if hasattr(cdf, 'latest_diagonal') else cdf
        else:
            # Return a simple default CDF if not available
            return X.latest_diagonal * 0 + 1.0

    def _set_ult_attr(self, ultimate):
        """ Ultimate scaffolding - simplified for polars backend """
        from chainladder import options
        
        # Simplified for polars backend - skip array module operations
        # Handle NaN values - this may not be necessary for polars backend
        # ultimate.ddims = pd.DatetimeIndex([options.ULT_VAL])  # Skip for now
        # ultimate.virtual_columns.columns = {}  # Skip for now
        # ultimate.is_cumulative = True  # This should already be set
        # Skip legacy methods: _set_slicers, _drop_subtriangles
        return ultimate

    @property
    def ldf_(self):
        return self.X_.ldf_

    @property
    def latest_diagonal(self):
        if self.X_.is_cumulative:
            return self.X_.latest_diagonal
        else:
            return self.X_.sum('development')

    def fit(self, X, y=None, sample_weight=None):
        """Applies the chainladder technique to triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : Ignored
        sample_weight : Triangle
            For exposure-based methods, the exposure to be used for fitting

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = self.validate_X(X)
        self.validate_weight(X, sample_weight)
        if sample_weight:
            # Simplified for polars backend - no need for set_backend
            self.sample_weight_ = sample_weight
        else:
            self.sample_weight_ = sample_weight
        return self

    def predict(self, X, sample_weight=None):
        """Predicts the chainladder ultimate on a new triangle **X**

        Parameters
        ----------
        X : Triangle
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        sample_weight : Triangle
            For exposure-based methods, the exposure to be used for predictions

        Returns
        -------
        X_new: Triangle

        """
        X_new = X.val_to_dev()
        if sum(X_new.ddims > self.ldf_.ddims.max()) > 0:
            raise ValueError("X has ages that exceed those available in model.")
        X_new = X_new + (self.X_.val_to_dev().iloc[0,0].sum(2) * 0)
        self.validate_weight(X_new, sample_weight)
        if sample_weight:
            # Simplified for polars backend - no need for set_backend
            pass
        X_new.ldf_ = self.ldf_
        X_new, X_new.ldf_ = self.intersection(X_new, X_new.ldf_)
        return X_new
        
    def intersection(self, a, b):
        """ Given two Triangles with mismatched indices, this method aligns
            their indices """
        if len(a) == 1 and len(b) == 1:
            return a, b
        intersection = list(set(a.key_labels).intersection(set(b.key_labels)))
        if intersection == []:
            return a, b
        a_idx = a.index[intersection]
        b_idx = b.index[intersection]
        idx_intersection = list(
            set(a_idx.set_index(intersection).index.intersection(
                b_idx.set_index(intersection).index)))
        if (len(a) == 1 or len(b) == 1) and idx_intersection == []:
            return a, b
        b = b.iloc[b_idx[b_idx[intersection].set_index(intersection).index.isin(idx_intersection)].index]
        a = a.iloc[a_idx[a_idx[intersection].set_index(intersection).index.isin(idx_intersection)].index]
        return a, b

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.predict(X, sample_weight)

    def _include_process_variance(self):
        if hasattr(self.X_, "_get_process_variance"):
            full = self.full_triangle_
            obj = self.X_._get_process_variance(full)
            self.ultimate_.values = obj.values[..., -1:]
            process_var = obj - full
        else:
            process_var = None
        return process_var

    def validate_weight(self, X, sample_weight):
        if (
            sample_weight
            and X.shape[:-1] != sample_weight.shape[:-1]
            and sample_weight.shape[2] != 1
            and sample_weight.shape[0] > 1
        ):
            warnings.warn(
                "X and sample_weight are not aligned. Broadcasting may occur.\n")
