
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from chainladder.development.base import DevelopmentBase


class DevelopmentML(DevelopmentBase):
    """ A Estimator that interfaces with machine learning (ML) tools that implement
    the scikit-learn API.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    estimator_ml : skearn Estimator
        Any sklearn compatible regression estimator, including Pipelines and
    y_ml : list or str or sklearn_transformer
        The response column(s) for the machine learning algorithm. It must be
        present within the Triangle.
    y_features :
        The subset of response column(s) to use as lagged features for the
        Time Series aspects of the model. Predictions from one development period
        get used as featues in the next development period.
    fit_incrementals :
        Whether the response variable should be converted to an incremental basis
        for fitting.


    Attributes
    ----------
    ldf_ : Triangle
        The estimated loss development patterns.
    cdf_ : Triangle
        The estimated cumulative development patterns.
    """
    def __init__(self, estimator_ml=None,
                 y_ml=None, y_features=False, fit_incrementals=True):
        self.estimator_ml=estimator_ml
        self.y_ml=y_ml
        self.y_features=y_features
        self.fit_incrementals=fit_incrementals

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

    def _get_triangle_ml(self, df):
        """ Create fitted Triangle """
        from chainladder.core import Triangle
        preds = self.estimator_ml.predict(df)
        X_r = [df]
        y_r = [preds]
        latest_filter = df['origin']+(df['development']-12)/12
        latest_filter = latest_filter == latest_filter.max()
        preds=pd.DataFrame(preds.copy())[latest_filter].values
        out = df.loc[latest_filter].copy()
        dev_lags = df['development'].drop_duplicates().sort_values()
        for d in dev_lags[1:]:
            out['development'] = out['development'] + 12
            if len(preds.shape) == 1:
                preds = preds[:, None]
            if self.y_features:
                for num, col in enumerate(self.y_features):
                    out[col[0]]=preds[:, num]
            out = out[out['development']<=dev_lags.max()]
            X_r.append(out.copy())
            preds = self.estimator_ml.predict(out)
            y_r.append(preds.copy())
        X_r = pd.concat(X_r, 0).reset_index(drop=True)
        if True:
            X_r = X_r.drop(self._get_y_names(), 1)
        out = pd.concat((X_r,
                         pd.DataFrame(np.concatenate(y_r, 0), columns=self._get_y_names())),1)
        out['valuation']=out['origin']+(out['development']-12)/12
        return Triangle(
            out, origin='origin', development='valuation', index=self._key_labels, columns=self._get_y_names()).dropna()


    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the munich adjustment will be applied.
        y : None
            Ignored, use y_ml to set a reponse variable for the ML algorithm
        sample_weight : None
            Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if self.fit_incrementals:
            X_ = X.cum_to_incr()
        else:
            X_ = X.copy()
        self._columns = list(X.columns)
        self._key_labels = X.key_labels

        # response as a feature
        if self.y_features:
            for i in self.y_features:
                lag = X[i[2]].shift(i[1])
                X_[i[0]] = lag[lag.valuation<=X.valuation_date]

        df = X_.to_frame(keepdims=True).reset_index().fillna(0)
        df['origin'] = df['origin'].dt.year # sklearn doesnt like datetimes
        self.df_ = df # Unncecessary, used for debugging

        # Fit model
        self.estimator_ml.fit(df, self.y_ml_.fit_transform(df).squeeze())
        self.triangle_ml_ = self._get_triangle_ml(df)
        return self

    @property
    def ldf_(self):
        from chainladder import ULT_VAL
        ldf = self.triangle_ml_.incr_to_cum().link_ratio
        ldf.valuation_date = pd.to_datetime(ULT_VAL)
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
        triangles = [
            "ldf_",
        ]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new.sigma_ = X_new.std_err_ = X_new.ldf_ * 0 + 1
        X_new._set_slicers()
        return X_new
