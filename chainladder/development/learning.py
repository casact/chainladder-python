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

    Examples
    --------
    ``DevelopmentML`` bridges scikit-learn and the chainladder workflow:
    it converts a Triangle to a tabular DataFrame, fits any sklearn-compatible
    estimator or Pipeline against it, then converts the predictions back into
    ``ldf_`` patterns usable with tail selection, ``Chainladder``, and other
    methods.

    ``fit_incrementals`` controls what the estimator is trained to predict.
    When an actuary believes the period-to-period change in losses is more
    predictable from development age than the cumulative total is, training
    on incrementals (``fit_incrementals=True``) is more appropriate. When
    the cumulative level is the more natural target, use
    ``fit_incrementals=False``. Both options produce an ``ldf_``, but the
    fitted pattern differs because the training target changes. (Whether the
    incremental values represent dollar amounts depends on the estimator:
    a ``LinearRegression`` with no log transform trains on raw dollar
    increments, while a log-space transform would fit log-scale values.)

    .. testsetup::

        import chainladder as cl

    .. testcode::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        from chainladder.utils.utility_functions import PatsyFormula

        tri = cl.load_sample("genins")
        pipe = Pipeline(
            steps=[
                ("design_matrix", PatsyFormula("C(development)")),
                ("model", LinearRegression(fit_intercept=False)),
            ]
        )
        m_incr = cl.DevelopmentML(
            pipe, y_ml=[tri.columns[0]], fit_incrementals=True
        ).fit(tri)
        m_cum = cl.DevelopmentML(
            pipe, y_ml=[tri.columns[0]], fit_incrementals=False
        ).fit(tri)
        print(m_incr.ldf_.to_frame(origin_as_datetime=False).iloc[0].values[:4].round(4))
        print(m_cum.ldf_.to_frame(origin_as_datetime=False).iloc[0].values[:4].round(4))

    .. testoutput::

        [3.508  1.7435 1.4379 1.1655]
        [3.515  1.735  1.3993 1.152 ]

    By default the regression treats each cell equally regardless of its loss
    size. An actuary who prefers development patterns to reflect the
    experience of larger accident years more heavily (similar to traditional
    volume-weighted averages) can pass ``sample_weight`` to ``fit``. However,
    passing ``sample_weight`` alone is not enough: without ``weighted_step``,
    the weights are computed internally but never forwarded to the sklearn
    estimator. ``weighted_step`` takes the name of a step in the Pipeline as
    a string and routes the weights to that step using sklearn's
    ``step_name__sample_weight`` fit-params convention. In the example below,
    ``weighted_step='model'`` matches the name given to the
    ``LinearRegression`` step, so it receives ``sample_weight=tri * tri``
    (squared cumulative losses) during fitting, giving larger-loss accident
    years proportionally more influence on the development pattern.

    .. testcode::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        from chainladder.utils.utility_functions import PatsyFormula

        tri = cl.load_sample("genins")
        pipe = Pipeline(
            steps=[
                ("design_matrix", PatsyFormula("C(development)")),
                ("model", LinearRegression(fit_intercept=False)),
            ]
        )
        m0 = cl.DevelopmentML(
            pipe, y_ml=[tri.columns[0]], fit_incrementals=False
        ).fit(tri)
        m1 = cl.DevelopmentML(
            pipe,
            y_ml=[tri.columns[0]],
            fit_incrementals=False,
            weighted_step="model",
        ).fit(tri, sample_weight=tri * tri)
        print(m0.ldf_.to_frame(origin_as_datetime=False).iloc[0].values[:4].round(4))
        print(m1.ldf_.to_frame(origin_as_datetime=False).iloc[0].values[:4].round(4))

    .. testoutput::

        [3.515  1.735  1.3993 1.152 ]
        [3.4459 1.7749 1.4053 1.1377]

    Loss development has a natural time-series structure: the cumulative loss
    at one age is directly related to the amount at the prior age. The
    ``autoregressive`` parameter lets the model re-feed its own predictions
    as lagged inputs when filling in the lower triangle. Each tuple in the
    list specifies ``(column_name, lag, source_column)``: the Triangle column
    ``source_column`` is shifted by ``lag`` periods and added to the tabular
    DataFrame under ``column_name``. During prediction, each period's output
    becomes the next period's lagged input automatically. The
    ``column_name`` must also appear in the Pipeline formula so the estimator
    can use it as a feature.

    .. testcode::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        from chainladder.utils.utility_functions import PatsyFormula

        tri = cl.load_sample("genins")
        col = tri.columns[0]
        pipe = Pipeline(
            steps=[
                ("design_matrix", PatsyFormula("C(development) + lag_loss")),
                ("model", LinearRegression(fit_intercept=False)),
            ]
        )
        m = cl.DevelopmentML(
            pipe,
            y_ml=[col],
            autoregressive=[("lag_loss", -1, col)],
            fit_incrementals=False,
        ).fit(tri)
        print(m.ldf_.to_frame(origin_as_datetime=False).iloc[0].values[:4].round(4))

    .. testoutput::

        [3.4815 1.6246 1.3017 1.0065]

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
