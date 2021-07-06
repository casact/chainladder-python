# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import warnings
from sklearn.base import BaseEstimator
from chainladder.tails import TailConstant
from chainladder.development import Development
from chainladder.core.io import EstimatorIO
from chainladder.core.common import Common


class MethodBase(BaseEstimator, EstimatorIO, Common):

    _estimator_type = "chainladder"

    def validate_X(self, X):
        obj = X.copy()
        if "ldf_" not in obj:
            obj = Development().fit_transform(obj)
        if len(obj.ddims) - len(obj.ldf_.ddims) == 1:
            obj = TailConstant().fit_transform(obj)
        return obj

    def _align_cdf(self, X, sample_weight=None):
        """ Vertically align CDF to origin period latest diagonal. """
        valuation = X.valuation_date
        cdf = self.cdf_.iloc[..., : X.shape[-1]]
        a = X.iloc[0, 0] * 0
        a = a + a.nan_triangle
        if X.array_backend == "sparse":
            a = a - a[a.valuation < a.valuation_date]
        if sample_weight:
            X = X * a + sample_weight * a
        else:
            X = X * a
        cdf = X / X * cdf
        cdf.valuation_date = valuation
        return cdf.latest_diagonal

    def _set_ult_attr(self, ultimate):
        """ Ultimate scaffolding """
        from chainladder import ULT_VAL

        xp = ultimate.get_array_module()
        if ultimate.array_backend != "sparse":
            ultimate.values[~xp.isfinite(ultimate.values)] = xp.nan
        ultimate.ddims = pd.DatetimeIndex([ULT_VAL])
        ultimate.virtual_columns.columns = {}
        ultimate.is_cumulative = True
        ultimate._set_slicers()
        ultimate.valuation_date = ultimate.valuation.max()
        ultimate._drop_subtriangles()
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
            self.sample_weight_ = sample_weight.set_backend(self.X_.array_backend)
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
        obj = X.copy()
        xp = obj.get_array_module()
        obj.ldf_ = self.ldf_
        self.validate_weight(X, sample_weight)
        if sample_weight:
            sample_weight = sample_weight.set_backend(obj.array_backend)
        obj.ultimate_ = self._get_ultimate(obj, sample_weight)
        return obj

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
                "X and sample_weight are not aligned.  Broadcasting may occur.\n"
            )
