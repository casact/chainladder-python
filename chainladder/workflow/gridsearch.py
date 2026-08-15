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
        The number of jobs to use for the computation. See :term:`Glossary <n_jobs>`
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
    affects the fitted development pattern. ``GridSearch`` fits one pipeline
    per candidate in ``param_grid``, and the ``scoring`` callables can record
    any fitted attribute, such as the full vector of ``ldf_`` by development
    age.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        clrd = cl.load_sample("clrd")
        medmal = clrd.groupby("LOB").sum().loc["medmal"]["CumPaidLoss"]
        pipe = cl.Pipeline(
            [("dev", cl.Development()), ("cl", cl.Chainladder())]
        )
        param_grid = {"dev__average": ["simple", "volume"]}

        def ldf_by_age(model):
            ldf = model.named_steps.dev.ldf_
            return ldf.values[0, 0, 0, :].round(3).tolist()

        grid = cl.GridSearch(
            pipe, param_grid, scoring={"ldf": ldf_by_age}, n_jobs=1
        ).fit(medmal)
        for _, row in grid.results_.iterrows():
            print(row["dev__average"], row["ldf"])

    .. testoutput::

        simple [6.076, 1.976, 1.384, 1.2, 1.102, 1.068, 1.039, 1.029, 1.018]
        volume [5.856, 1.963, 1.376, 1.199, 1.099, 1.067, 1.039, 1.028, 1.018]

    Because both rows are loss development factors, they are directly
    comparable age by age. Simple averaging gives every origin year an equal
    vote, while volume weighting lets the origin years with the most losses
    dominate, and for this triangle that distinction matters most at the
    immature 12-24 age (6.076 versus 5.856). The two candidates converge as
    the line matures, so the averaging choice mainly moves the reserve
    carried for the most recent origin years.

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
    Read more in the :ref:`User Guide <workflow:pipeline>`.

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
    estimator, so an entire reserving workflow is specified in one place. A
    classic workflow chains three steps: select development patterns from
    the triangle, extrapolate a tail beyond the observed ages, and hand the
    completed patterns to an IBNR method. Fitting the pipeline runs every
    step in order, and ``named_steps`` exposes each fitted step by name.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        genins = cl.load_sample("genins")
        pipe = cl.Pipeline(
            [
                ("dev", cl.Development(average="volume")),
                ("tail", cl.TailCurve(curve="exponential")),
                ("model", cl.Chainladder()),
            ]
        )
        pipe.fit(genins)
        ibnr = (
            pipe.named_steps.model.ibnr_.to_frame(origin_as_datetime=False)
            .iloc[:, 0]
            .astype(int)
            .rename("IBNR")
        )
        print(ibnr)

    .. testoutput::

        2001     115089
        2002     254924
        2003     628182
        2004     865921
        2005    1128201
        2006    1570234
        2007    2344628
        2008    4120446
        2009    4445414
        2010    4772416
        Freq: Y-DEC, Name: IBNR, dtype: int64

    Each step feeds the next: the volume-weighted patterns from ``dev`` are
    extended past the edge of the triangle by the exponential tail in
    ``tail``, and the chainladder ``model`` projects every origin year to
    ultimate with those completed patterns. The full origin-by-origin IBNR
    comes from the final step, while intermediate results such as the fitted
    tail factor remain available through ``pipe.named_steps.tail``.

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
