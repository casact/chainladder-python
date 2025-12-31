# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from chainladder.development.base import DevelopmentBase
from chainladder import options


class DevelopmentML(DevelopmentBase):
    """ A Estimator that interfaces with machine learning (ML) tools that implement
    the scikit-learn API.

    The `DevelopmentML` estimator is used to generate ``ldf_`` patterns from
    the data.

    .. versionadded:: 0.8.1


    Parameters
    ----------
    estimator_ml: skearn Estimator
        Any sklearn compatible regression estimator, including Pipelines and
    y_ml: list or str or sklearn_transformer
        The response column(s) for the machine learning algorithm. It must be
        present within the Triangle.
    autoregressive: tuple, (autoregressive_col_name, lag, source_col_name)
        The subset of response column(s) to use as lagged features for the
        Time Series aspects of the model. Predictions from one development period
        get used as featues in the next development period. Lags should be negative
        integers.
    weight_step: str
        Step name within estimator_ml that is weighted
    drop: tuple or list of tuples
        Drops specific origin/development combination(s)
    drop_valuation: str or list of str (default = None)
        Drops specific valuation periods. str must be date convertible.
    fit_incrementals:
        Whether the response variable should be converted to an incremental basis for fitting.

    Attributes
    ----------
    estimator_ml: Estimator
        An sklearn-style estimator to predict development patterns
    ldf_: Triangle
        The estimated loss development patterns.
    cdf_: Triangle
        The estimated cumulative development patterns.
    """

    def __init__(self, estimator_ml=None, y_ml=None, autoregressive=False,
                 weighted_step=None,drop=None,drop_valuation=None,fit_incrementals=True):
        self.estimator_ml=estimator_ml
        self.y_ml=y_ml
        self.weighted_step = weighted_step
        self.autoregressive = autoregressive
        self.drop = drop
        self.drop_valuation = drop_valuation
        self.fit_incrementals = fit_incrementals

    def _get_y_names(self):
        """ private function to get the response column name"""
        if not self.y_ml:
            y_names = self._columns
        if hasattr(self.y_ml, '_columns'):
            y_names = self.y_ml._columns
        elif isinstance(self.y_ml, ColumnTransformer):
            y_names = self.y_ml.transformers[0][-1]
        if type(self.y_ml) is list:
            y_names = self.y_ml
        elif type(self.y_ml) is str:
            y_names = [self.y_ml]
        return y_names


    @property
    def y_ml_(self):
        defaults = self._get_y_names()
        transformer = self.y_ml
        if not transformer:
            return ColumnTransformer(
                transformers=[('passthrough', 'passthrough', defaults)])
        elif type(transformer) is list:
            return ColumnTransformer(
                transformers=[('passthrough', 'passthrough', transformer)])
        elif type(transformer) is str:
            return ColumnTransformer(
                transformers=[('passthrough', 'passthrough', [transformer])])
        else:
            return transformer

    def _get_triangle_ml(self, df, preds=None):
        """ Create fitted Triangle """
        from chainladder.core import Triangle
        if preds is None:
            preds = self.estimator_ml.predict(df)
        X_r = [df]
        y_r = [preds]
        dgrain = {'Y':12, 'S': 6, 'Q':3, 'M': 1}[self.development_grain_]
        ograin = {'Y':1, 'S': 2, 'Q':4, 'M': 12}[self.origin_grain_]
        latest_filter = (df['origin']+1)*ograin+(df['development']-dgrain)/dgrain
        latest_filter = latest_filter == latest_filter.max()
        preds=pd.DataFrame(preds.copy())[latest_filter].values
        out = df.loc[latest_filter].copy()
        dev_lags = df['development'].drop_duplicates().sort_values()
        for d in dev_lags[1:]:
            out['development'] = out['development'] + dgrain
            out['valuation'] = out['valuation'] + dgrain / 12
            if len(preds.shape) == 1:
                preds = preds[:, None]
            if self.autoregressive:
                for num, col in enumerate(self.autoregressive):
                    out[col[0]]=preds[:, num]
            out = out[out['development']<=dev_lags.max()]
            if len(out) == 0:
                continue
            X_r.append(out.copy())
            preds = self.estimator_ml.predict(out)
            y_r.append(preds.copy())
        X_r = pd.concat(X_r, axis=0).reset_index(drop=True)
        if True:
            X_r = X_r.drop(self._get_y_names(), axis=1)
        out = pd.concat((X_r,
                         pd.DataFrame(np.concatenate(y_r, 0), columns=self._get_y_names())), axis=1)
        out['origin'] = out['origin'].map({v: k for k, v in self.origin_encoder_.items()})
        out['valuation'] = out['valuation'].map({v: k for k, v in self.valuation_encoder_.items()})
        return Triangle(
            out, origin='origin', development='valuation',
            index=self._key_labels, columns=self._get_y_names(),
            cumulative=not self.fit_incrementals).dropna(), out

    def _prep_X_ml(self, X):
        """ Preps Triangle data ahead of the pipeline """
        if self.fit_incrementals:
            X_ = X.cum_to_incr()
        else:
            X_ = X.copy()
        if self.autoregressive:
            for i in self.autoregressive:
                lag = X[i[2]].shift(i[1])
                X_[i[0]] = lag[lag.valuation<=X.valuation_date]
        df_base = X.incr_to_cum().to_frame(
            keepdims=True, implicit_axis=True, origin_as_datetime=True
            ).reset_index().iloc[:, :-1]
        df = df_base.merge(X_.to_frame(
                keepdims=True, implicit_axis=True, origin_as_datetime=True
            ).reset_index(), how='left',
            on=list(df_base.columns)).fillna(0)
        df['origin'] = df['origin'].map(self.origin_encoder_)
        df['valuation'] = df['valuation'].map(self.valuation_encoder_)
        return df

    def _prep_w_ml(self,X,sample_weight=None):
        weight_base = (~np.isnan(X.values)).astype(float)
        weight = weight_base.copy()               
        if self.drop is not None:
            weight = weight * self._drop_func(X)
        if self.drop_valuation is not None:
            weight = weight * self._drop_valuation_func(X)
        if sample_weight is not None:
            weight = weight * sample_weight.values 
        return weight.flatten()[weight_base.flatten()>0]

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the estimator will be applied.
        y : None
            Ignored, use y_ml to set a reponse variable for the ML algorithm
        sample_weight : Triangle-like
            Weights to use in the regression

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self._columns = list(X.columns)
        self._key_labels = X.key_labels
        self.origin_grain_ = X.origin_grain
        self.development_grain_ = X.development_grain
        self.origin_encoder_ = dict(zip(
            X.origin.to_timestamp(how='s'),
            (pd.Series(X.origin).rank()-1)/{'Y':1, 'S': 2, 'Q':4, 'M': 12}[X.origin_grain]))
        val = X.valuation.sort_values().unique()
        self.valuation_encoder_ = dict(zip(
            val,
            (pd.Series(val).rank()-1)/{'Y':1, 'S': 2, 'Q':4, 'M': 12}[X.development_grain]))
        df = self._prep_X_ml(X)
        self.df_ = df
        weight = self._prep_w_ml(X,sample_weight)
        self.weight_ = weight
        if self.weighted_step == None:
            sample_weights = {}
        elif isinstance(self.weighted_step, list):
            sample_weights = {x + '__sample_weight':weight for x in self.weighted_step}
        else:
            sample_weights = {self.weighted_step + '__sample_weight':weight}
        # Fit model
        self.estimator_ml.fit(df, self.y_ml_.fit_transform(df).squeeze(),**sample_weights)
        #return selffit_incrementals 
        self.triangle_ml_, self.predicted_data_ = self._get_triangle_ml(df)
        return self

    @property
    def ldf_(self):
        ldf = self.triangle_ml_.incr_to_cum().link_ratio
        ldf.valuation_date = pd.to_datetime(options.ULT_VAL)
        return ldf

    def transform(self, X):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = X.copy()
        X_ml = self._prep_X_ml(X)
        y_ml=self.estimator_ml.predict(X_ml)
        triangle_ml, predicted_data = self._get_triangle_ml(X_ml, y_ml)
        backend = "cupy" if X.array_backend == "cupy" else "numpy"
        X_new.ldf_ = triangle_ml.incr_to_cum().link_ratio.set_backend(backend)
        X_new.ldf_.valuation_date = pd.to_datetime(options.ULT_VAL)
        X_new._set_slicers()
        X_new.predicted_data_ = predicted_data
        return X_new