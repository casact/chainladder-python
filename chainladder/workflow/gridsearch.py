# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline as PipelineSL
from sklearn.base import clone
from chainladder.core.io import EstimatorIO
from joblib import Parallel, delayed
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
    estimator: estimator object.
        This is assumed to implement the chainladder estimator interface.
    param_grid: dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    scoring: callable or dict of callable(s)
        Should be of the form {'name': callable}.  The callable(s) should
        return a single value.
    verbose: integer
        Controls the verbosity: the higher, the more messages.
    error_score: 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is 'raise' but from
        version 0.22 it will change to np.nan.
    n_jobs: int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup for n_targets > 1 and sufficient large problems.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    results_: DataFrame
        A DataFrame with each param_grid key as a column and the ``scoring``
        score as the last column

    Examples
    --------
    Suppose an actuary reserving the industry medical malpractice line wants
    to see how the choice between simple and volume-weighted averaging
    affects the variability of the fitted development factors. ``GridSearch``
    fits one pipeline per candidate in ``param_grid``, and the ``scoring``
    callables can record any fitted attribute, such as the full vector of
    ``sigma_`` by development age.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        clrd = cl.load_sample("clrd")
        medmal = clrd.groupby("LOB").sum().loc["medmal"]["CumPaidLoss"]
        pipe = cl.Pipeline(
            [("dev", cl.Development()), ("cl", cl.Chainladder())]
        )
        param_grid = {"dev__average": ["simple", "volume"]}

        def sigma_by_age(model):
            sigma = model.named_steps.dev.sigma_
            return sigma.values[0, 0, 0, :].round(3).tolist()

        grid = cl.GridSearch(
            pipe, param_grid, scoring={"sigma": sigma_by_age}, n_jobs=1
        ).fit(medmal)
        for _, row in grid.results_.iterrows():
            print(row["dev__average"], row["sigma"])

    .. testoutput::

        simple [1.163, 0.102, 0.057, 0.038, 0.026, 0.016, 0.007, 0.01, 0.003]
        volume [116.206, 26.551, 18.805, 15.471, 11.936, 7.286, 3.458, 4.49, 1.978]

    Both candidates agree that development factor variability is concentrated
    in the 12-24 age and tapers as the line matures. The two rows are on very
    different scales because each averaging choice fits a different weighted
    regression, so ``sigma_`` magnitudes are only comparable between
    candidates that share an averaging method.

    """

    def __init__(self, estimator, param_grid, scoring, verbose=0,
                 error_score="raise", n_jobs=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        """Fit the model with X.

        Parameters
        ----------
        X: Triangle-like
            Set of LDFs to which the tail will be applied.
        y: Ignored
        fit_params: (optional) dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        if type(self.scoring) is not dict:
            scoring = dict(score=self.scoring)
        else:
            scoring = self.scoring
        grid = list(ParameterGrid(self.param_grid))

        def _fit_single_estimator(estimator, fit_params, X, y, scoring, item):
            est = clone(estimator).set_params(**item)
            model = est.fit(X, y, **fit_params)
            for score in scoring.keys():
                item[score] = scoring[score](model)
            return item
            
        results_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_single_estimator)(
            self.estimator, fit_params, X, y, scoring, item)
            for item in grid)
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
    steps: list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    memory: None, str or object with the joblib.Memory interface, optional
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
    named_steps: bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    Examples
    --------
    A ``Pipeline`` wraps transformers and a final estimator into one compact
    estimator, so an entire reserving analysis is specified by hyperparameters
    alone. The user guide builds a pipeline that develops every company in the
    CAS loss reserve database with pooled line-of-business patterns through
    the ``groupby`` hyperparameter of ``Development``. Comparing it against
    the same pipeline without ``groupby`` shows why that pooling matters:
    standalone patterns are fit to each company's own thin data.

    .. testsetup::

        import chainladder as cl
        import pandas as pd

    .. testcode::

        clrd = cl.load_sample("clrd")["CumPaidLoss"]
        industry = cl.Pipeline(
            [("dev", cl.Development(groupby="LOB")), ("model", cl.Chainladder())]
        )
        standalone = cl.Pipeline(
            [("dev", cl.Development()), ("model", cl.Chainladder())]
        )
        ibnr_industry = industry.fit(clrd).named_steps.model.ibnr_
        ibnr_standalone = standalone.fit(clrd).named_steps.model.ibnr_
        summary = pd.DataFrame(
            {
                "industry": ibnr_industry.groupby("LOB").sum().sum("origin").to_frame(),
                "standalone": ibnr_standalone.groupby("LOB").sum().sum("origin").to_frame(),
            }
        ).astype(int).rename_axis(None)
        print(summary)

    .. testoutput::

                  industry  standalone
        comauto    1743192     1683207
        medmal     1330330     1455883
        othliab    1640597   -14285800
        ppauto    17138458    17327738
        prodliab    531648      577127
        wkcomp     2777812     2498151

    The two pipelines differ in a single hyperparameter, yet the standalone
    fit lets a handful of small companies with erratic development drive the
    other liability line to a negative aggregate IBNR, while the pooled
    patterns keep every line at a reasonable estimate.

    """

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
