# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline as PipelineSL
from chainladder.core.io import EstimatorIO
import copy
import pandas as pd
import json


class GridSearch(BaseEstimator):
    """Exhaustive search over specified parameter values for an estimator.
    Important members are fit, predict.
    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the chainladder estimator interface.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    scoring : callable or dict of callable(s)
        Should be of the form {'name': callable}.  The callable(s) should
        return a single value.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.

    Attributes
    ----------
    results_ : DataFrame
        A DataFrame with each param_grid key as a column and the ``scoring``
        score as the last column
    """

    def __init__(self, estimator, param_grid, scoring, verbose=0, error_score="raise"):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.verbose = verbose
        self.error_score = error_score

    def fit(self, X, y=None, **fit_params):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the tail will be applied.
        y : Ignored
        fit_params : (optional) dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if type(self.scoring) is not dict:
            scoring = dict(score=self.scoring)
        else:
            scoring = self.scoring
        grid = list(ParameterGrid(self.param_grid))
        results_ = []
        for num, item in enumerate(grid):
            est = copy.deepcopy(self.estimator).set_params(**item)
            model = est.fit(X, y, **fit_params)
            for score in scoring.keys():
                item[score] = scoring[score](model)
            results_.append(item)
        self.results_ = pd.DataFrame(results_)
        return self


class Pipeline(PipelineSL, EstimatorIO):
    """This is a near direct of copy the scikit-learn Pipeline class.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    The transformers in the pipeline can be cached using ``memory`` argument.
    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    to None.
    Read more in the :ref:`User Guide <pipeline_docs>`.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters."""

    def fit(self, X, y=None, sample_weight=None, **fit_params):
        if sample_weight:
            fit_params = {} if not fit_params else fit_params
            for step in self.steps:
                fit_params[step[0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)

    def predict(self, X, sample_weight=None, **predict_params):
        if sample_weight:
            predict_params = {} if not predict_params else predict_params
            predict_params["sample_weight"] = sample_weight
        return super().predict(X, **predict_params)

    def fit_predict(self, X, y=None, sample_weight=None, **fit_params):
        self.fit(X, y, sample_weight, **fit_params)
        return self.predict(X, sample_weight, **fit_params)

    def to_json(self):
        return json.dumps(
            [
                {
                    "name": item[0],
                    "params": item[1].get_params(),
                    "__class__": item[1].__class__.__name__,
                }
                for item in self.steps
            ]
        )
