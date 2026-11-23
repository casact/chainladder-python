# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline as PipelineSL
from sklearn.base import clone
from chainladder.core.io import EstimatorIO
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import json


class LeaveOneOutGrid(dict):
    """
    A ``param_grid`` that leaves out one link ratio per candidate.

    ``GridSearch.leave_one_out_grid`` builds this grid. ``GridSearch`` and
    ``ParameterGrid`` read it as a plain ``dict``, and
    ``GridSearch.compare_results`` reads its labels to name each row.

    Parameters
    ----------
    key: str
        The parameter name, such as ``"dev__drop"``.
    labels: DataFrame
        One row per candidate, indexed by label, with the ``origin``,
        ``age`` and ``link_ratio`` of the dropped ratio and the grid
        ``value`` itself. The candidate that drops nothing has the label
        ``"base"``.

    Attributes
    ----------
    key: str
        The parameter name.
    labels: DataFrame
        The label table passed in.
    """

    def __init__(self, key, labels):
        super().__init__({key: list(labels["value"])})
        self.key = key
        self.labels = labels


class _Scorer:
    """
    A scoring callable that also tells ``compare_results`` what to report.

    Parameters
    ----------
    func: callable
        Takes a fitted estimator and returns its score.
    metrics: callable
        Takes the scorer name, the compared rows, the base row and the label
        table, and returns a DataFrame of derived columns together with the
        Series of changes that orders the rows.
    """

    def __init__(self, func, metrics):
        self.func = func
        self.metrics = metrics

    def __call__(self, model):
        return self.func(model)

    def __repr__(self):
        return f"GridSearch.{self.func.__name__.lstrip('_')}"


def _final_step(model):
    return model.steps[-1][1] if isinstance(model, PipelineSL) else model


def _ibnr(model):
    return float(_final_step(model).ibnr_.sum())


def _ldf(model):
    ldf = _final_step(model).ldf_
    ages = [int(str(interval).split("-")[0]) for interval in ldf.development]
    return pd.Series(ldf.values[0, 0, 0], index=ages)


def _ibnr_metrics(name, rows, base, labels):
    change = rows[name].astype(float) - float(base[name])
    frame = pd.DataFrame({
        f"{name}_change": change,
        f"{name}_share": change / float(base[name]),
    })
    return frame, change


def _ldf_metrics(name, rows, base, labels):
    if "age" in labels:
        ages = labels["age"].reindex(rows.index)
    else:
        ages = pd.Series(pd.NA, index=rows.index)
    before, after = [], []
    for value, age in zip(rows[name], ages):
        if pd.isna(age):
            before.append(np.nan)
            after.append(np.nan)
        else:
            before.append(base[name].get(int(age), np.nan))
            after.append(value.get(int(age), np.nan))
    frame = pd.DataFrame(
        {f"{name}_before": before, f"{name}_after": after}, index=rows.index
    )
    return frame, frame[f"{name}_after"] - frame[f"{name}_before"]


class GridSearch(BaseEstimator):
    """
    Exhaustive search over specified parameter values for an estimator.
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
    ibnr: callable
        Built-in scorer that records total IBNR from the final step. In
        ``compare_results`` it adds the change in total IBNR and its share
        of the base.
    ldf: callable
        Built-in scorer that records the ``ldf_`` of the final step by age.
        In ``compare_results`` it adds the factor at the dropped age before
        and after the drop.

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

    ibnr = _Scorer(_ibnr, _ibnr_metrics)
    ldf = _Scorer(_ldf, _ldf_metrics)

    def __init__(
        self,
        estimator,
        param_grid,
        scoring,
        verbose=0,
        error_score="raise",
        n_jobs=None,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.verbose = verbose
        self.error_score = error_score
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        """
        Fit the model with X.

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

        results_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                self.estimator, fit_params, X, y, scoring, item
            )
            for item in grid
        )
        self.results_ = pd.DataFrame(results_)
        return self

    @staticmethod
    def leave_one_out_grid(X, step="dev"):
        """
        Build a ``param_grid`` that leaves out one link ratio at a time.

        Parameters
        ----------
        X: Triangle
            A cumulative triangle with one index and one column.
        step: str, default="dev"
            The name of the ``Development`` step in the pipeline that
            ``GridSearch`` fits.

        Returns
        -------
        LeaveOneOutGrid
            A ``{step}__drop`` grid. Its first value is ``None``, the fit
            that keeps every ratio, and each later value is the
            ``(origin, age)`` pair of one observed link ratio, including
            ages that have only one ratio.

        Raises
        ------
        ValueError
            If ``X`` has more than one index or column.

        Examples
        --------
        .. testsetup::

            import chainladder as cl

        .. testcode::

            raa = cl.load_sample("raa")
            grid = cl.GridSearch.leave_one_out_grid(raa)
            print(len(grid["dev__drop"]), grid["dev__drop"][:3])

        .. testoutput::

            46 [None, ('1981', 12), ('1981', 24)]
        """
        if X.shape[:2] != (1, 1):
            raise ValueError(
                "leave_one_out_grid needs a triangle with one index and one "
                f"column, but X has {X.shape[0]} and {X.shape[1]}."
            )
        link_ratio = X.link_ratio
        values = link_ratio.values[0, 0]
        rows = [
            {
                "label": "base",
                "origin": None,
                "age": pd.NA,
                "link_ratio": np.nan,
                "value": None,
            }
        ]
        for i, j in zip(*np.nonzero(~np.isnan(values))):
            origin = str(link_ratio.origin[i])
            age = int(X.development[j])
            rows.append({
                "label": f"{origin}, {age}",
                "origin": origin,
                "age": age,
                "link_ratio": values[i, j],
                "value": (origin, age),
            })
        labels = pd.DataFrame(rows).set_index("label")
        labels.index.name = None
        labels["age"] = labels["age"].astype("Int64")
        return LeaveOneOutGrid(f"{step}__drop", labels)

    def compare_results(self, base):
        """
        Compare every candidate in ``results_`` with one base candidate.

        Parameters
        ----------
        base: label
            The candidate to compare with. For a grid from
            ``leave_one_out_grid`` this is ``"base"``. For any other grid it
            is an index of ``results_``.

        Returns
        -------
        DataFrame
            One row per other candidate. Rows from a ``leave_one_out_grid``
            carry the label of the dropped ratio as their index and its
            ``origin``, ``age`` and ``link_ratio`` as columns, and rows from
            any other grid keep the index of ``results_``. Each built-in
            scorer in ``scoring`` adds its own columns, and any other scorer
            keeps its column from ``results_``. Rows run from the largest
            change to the smallest under the first built-in scorer.

        Examples
        --------
        .. testsetup::

            import chainladder as cl

        .. testcode::

            raa = cl.load_sample("raa")
            pipe = cl.Pipeline(
                [("dev", cl.Development()), ("model", cl.Chainladder())]
            )
            search = cl.GridSearch(
                pipe,
                cl.GridSearch.leave_one_out_grid(raa),
                scoring={"ibnr": cl.GridSearch.ibnr, "ldf": cl.GridSearch.ldf},
            ).fit(raa)
            table = search.compare_results("base")
            print(table[["ibnr_change", "ldf_before", "ldf_after"]].head(3).round(4))

        .. testoutput::

                      ibnr_change  ldf_before  ldf_after
            1985, 60    3362.8852      1.1134     1.1525
            1981, 96    2818.0421      1.0169     1.0331
            1981, 12    2467.6774      2.9994     3.4016
        """
        results = self.results_.copy()
        grid = self.param_grid
        if isinstance(grid, LeaveOneOutGrid):
            lookup = dict(zip(grid.labels["value"], grid.labels.index))
            results.index = pd.Index([lookup[value] for value in results[grid.key]])
            labels = grid.labels.drop(columns="value")
        else:
            labels = pd.DataFrame(index=results.index)
        base_row = results.loc[base]
        rows = results.drop(index=base)
        if isinstance(self.scoring, dict):
            scoring = self.scoring
        else:
            scoring = {"score": self.scoring}
        parts = [labels.loc[rows.index]]
        order = None
        for name, scorer in scoring.items():
            if isinstance(scorer, _Scorer):
                frame, change = scorer.metrics(name, rows, base_row, labels)
                parts.append(frame)
                if order is None:
                    order = change.abs().sort_values(ascending=False, kind="stable")
            else:
                parts.append(rows[[name]])
        table = pd.concat(parts, axis=1)
        if order is not None:
            table = table.loc[order.index]
        return table


class Pipeline(PipelineSL, EstimatorIO):
    """
    This is a near direct of copy the scikit-learn Pipeline class.

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
        return json.dumps([
            {
                "name": item[0],
                "params": item[1].get_params(),
                "__class__": item[1].__class__.__name__,
            }
            for item in self.steps
        ])
