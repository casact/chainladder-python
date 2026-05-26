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
        return obj.val_to_dev()

    def _align_cdf(self, X, sample_weight=None):
        """ Vertically align CDF to origin period latest diagonal. """
        valuation = X.valuation_date
        cdf = X.cdf_.iloc[..., : X.shape[-1]]
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
        from chainladder import options

        xp = ultimate.get_array_module()
        if ultimate.array_backend != "sparse":
            ultimate.values[~xp.isfinite(ultimate.values)] = xp.nan
        ultimate.ddims = pd.DatetimeIndex([options.ULT_VAL])
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

    @property
    def summary(self):
        return self._get_summary(self.X_,self.ultimate_)

    def _get_summary(self,X,ult):
        #columns for melt
        ids = X.key_labels + ['origin','development','valuation']
        #create dataframe for amount
        amount = X.incr_to_cum().latest_diagonal.to_frame(implicit_axis=True,keepdims=True).reset_index().melt(id_vars=ids,var_name = 'column',value_name='actual')
        amount_index = X.key_labels + ['origin','column']
        amount = amount[amount_index + ['development','actual']]
        amount.set_index(amount_index,inplace=True)
        #create dataframe for ultimate
        ultimate = ult.to_frame(implicit_axis=True,keepdims=True).reset_index().melt(id_vars=ids,var_name = 'column',value_name='ultimate')
        #columns for melt for dev factors
        dev_ids = X.ldf_.key_labels + ['origin','development','valuation']
        #create dataframe for ldf
        ldf = X.ldf_.to_frame(implicit_axis=True,keepdims=True).reset_index().melt(id_vars=dev_ids,var_name = 'column',value_name='ldf')
        dev_factor_index = X.ldf_.key_labels + ['development','column']
        ldf = ldf[dev_factor_index + ['ldf']]
        ldf.set_index(dev_factor_index,inplace=True)
        #create dataframe for cdf
        cdf = X.cdf_.to_frame(implicit_axis=True,keepdims=True).reset_index().melt(id_vars=dev_ids,var_name = 'column',value_name='cdf')
        cdf = cdf[dev_factor_index + ['cdf']]
        cdf.set_index(dev_factor_index,inplace=True)
        #assemble full summary. start from ultimate, as some methods (e.g. BF) return an ultimate without any actual amount
        output = ultimate.drop(columns=['development','valuation'])
        output = output.join(amount,on=amount_index,how='left')
        output = output.join(ldf,on=dev_factor_index,how='left')
        output = output.join(cdf,on=dev_factor_index,how='left')
        return output[X.key_labels + ['column','origin','development','actual','ldf','cdf','ultimate']]

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
        X_new = X.val_to_dev()
        if sum(X_new.ddims > self.ldf_.ddims.max()) > 0:
            raise ValueError("X has ages that exceed those available in model.")
        X_new = X_new + (self.X_.val_to_dev().iloc[0,0].sum(2) * 0)
        self.validate_weight(X_new, sample_weight)
        if sample_weight:
            sample_weight = sample_weight.set_backend(X_new.array_backend)
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
