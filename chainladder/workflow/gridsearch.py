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


class _Comparisons:
    """
    Comparison callables for ``GridSearch.compare_results``.

    Each comparison takes the ``scoring`` key, the compared rows of
    ``results_``, the base row and the label columns, and returns a named
    Series with one value per compared row.
    """

    @staticmethod
    def change(name, rows, base, labels):
        """
        Measure how far each candidate's score moves from the base score.

        Parameters
        ----------
        name: str
            The ``scoring`` key whose column the comparison reads.
        rows: DataFrame
            The rows of ``results_`` that ``compare_results`` compares.
        base: Series
            The row of ``results_`` for the base candidate.
        labels: DataFrame
            The label columns that ``result_labels`` built, or no columns.
            This comparison does not read them.

        Returns
        -------
        Series
            ``{name}_change``, the score of each row less the base score.
            The score must be a number, such as the one ``GridSearch.ibnr``
            returns.
        """
        change = rows[name].astype(float) - float(base[name])
        return change.rename(f"{name}_change")

    @staticmethod
    def share(name, rows, base, labels):
        """
        Measure each candidate's change in score as a share of the base score.

        Parameters
        ----------
        name: str
            The ``scoring`` key whose column the comparison reads.
        rows: DataFrame
            The rows of ``results_`` that ``compare_results`` compares.
        base: Series
            The row of ``results_`` for the base candidate.
        labels: DataFrame
            The label columns that ``result_labels`` built, or no columns.
            This comparison does not read them.

        Returns
        -------
        Series
            ``{name}_share``, the change divided by the base score. The score
            must be a number, such as the one ``GridSearch.ibnr`` returns.
        """
        change = rows[name].astype(float) - float(base[name])
        return (change / float(base[name])).rename(f"{name}_share")

    @staticmethod
    def _factor_at_age(name, rows, factors, labels, suffix):
        if "age" in labels:
            ages = labels["age"].reindex(rows.index)
        else:
            ages = pd.Series(pd.NA, index=rows.index)
        values = [
            np.nan if pd.isna(age) else factor.get(int(age), np.nan)
            for factor, age in zip(factors, ages)
        ]
        return pd.Series(values, index=rows.index, name=f"{name}_{suffix}", dtype=float)

    @staticmethod
    def base(name, rows, base, labels):
        """
        Read the base factor at the age that each row names.

        Parameters
        ----------
        name: str
            The ``scoring`` key whose column the comparison reads. Its scores
            must be Series indexed by age, such as the ones
            ``GridSearch.ldf`` returns.
        rows: DataFrame
            The rows of ``results_`` that ``compare_results`` compares.
        base: Series
            The row of ``results_`` for the base candidate.
        labels: DataFrame
            The label columns that ``result_labels`` built. The comparison
            reads the ``age`` column, which
            ``GridSearch.leave_one_out_grid_labels`` supplies.

        Returns
        -------
        Series
            ``{name}_base``, the base factor at each row's age. A row gets
            NaN when ``labels`` has no ``age`` column or the base has no
            factor at that age.
        """
        return _Comparisons._factor_at_age(
            name, rows, [base[name]] * len(rows), labels, "base"
        )

    @staticmethod
    def scenario(name, rows, base, labels):
        """
        Read each candidate's own factor at the age that its row names.

        Parameters
        ----------
        name: str
            The ``scoring`` key whose column the comparison reads. Its scores
            must be Series indexed by age, such as the ones
            ``GridSearch.ldf`` returns.
        rows: DataFrame
            The rows of ``results_`` that ``compare_results`` compares.
        base: Series
            The row of ``results_`` for the base candidate. This comparison
            does not read it.
        labels: DataFrame
            The label columns that ``result_labels`` built. The comparison
            reads the ``age`` column, which
            ``GridSearch.leave_one_out_grid_labels`` supplies.

        Returns
        -------
        Series
            ``{name}_scenario``, the candidate's factor at its row's age. A
            row gets NaN when ``labels`` has no ``age`` column or the
            candidate has no factor at that age, as when it drops the only
            ratio at that age.
        """
        return _Comparisons._factor_at_age(name, rows, rows[name], labels, "scenario")


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
        return a single value. Two built-in scorers read the final step of
        the fitted estimator. ``GridSearch.ibnr`` returns the total of its
        ``ibnr_`` as a float, and ``GridSearch.ldf`` returns its ``ldf_``
        as a Series indexed by the starting age of each development
        interval.
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

    comparisons = _Comparisons

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
    def ibnr(model):
        """
        Score a fitted estimator by the total IBNR of its final step.

        Parameters
        ----------
        model: estimator
            A fitted estimator or ``Pipeline`` whose final step has ``ibnr_``.

        Returns
        -------
        float
            The sum of ``ibnr_`` over every cell.
        """
        final = model.steps[-1][1] if isinstance(model, PipelineSL) else model
        return float(final.ibnr_.sum())

    @staticmethod
    def ldf(model):
        """
        Score a fitted estimator by the ``ldf_`` of its final step.

        Parameters
        ----------
        model: estimator
            A fitted estimator or ``Pipeline`` whose final step has ``ldf_``.

        Returns
        -------
        Series
            The factors of the first index and column, indexed by the starting
            age of each development interval.
        """
        final = model.steps[-1][1] if isinstance(model, PipelineSL) else model
        ldf = final.ldf_
        ages = [int(str(interval).split("-")[0]) for interval in ldf.development]
        return pd.Series(ldf.values[0, 0, 0], index=ages)

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
        dict
            A ``{step}__drop`` grid. Its first value is ``None``, the fit
            that keeps every ratio. Each later value is the
            ``(origin, age)`` of one observed link ratio, including ages
            that have only one ratio.

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
            print(len(grid["dev__drop"]), grid["dev__drop"][1])

        .. testoutput::

            46 ('1981', 12)
        """
        if X.shape[:2] != (1, 1):
            raise ValueError(
                "leave_one_out_grid needs a triangle with one index and one "
                f"column, but X has {X.shape[0]} and {X.shape[1]}."
            )
        link_ratio = X.link_ratio
        values = link_ratio.values[0, 0]
        drops = [None]
        for i, j in zip(*np.nonzero(~np.isnan(values))):
            drops.append((str(link_ratio.origin[i]), int(X.development[j])))
        return {f"{step}__drop": drops}

    @staticmethod
    def leave_one_out_grid_labels(value):
        """
        Label one value of a ``leave_one_out_grid``.

        Pass this function as ``result_labels`` to ``compare_results``.

        Parameters
        ----------
        value: tuple or None
            One value of the grid. ``None`` is the base fit. A tuple holds
            the origin and age of the dropped ratio.

        Returns
        -------
        Series
            The ``origin`` and ``age`` of the dropped ratio, named
            ``"origin, age"``. The base fit gets the name ``"base"``.

        Examples
        --------
        .. testsetup::

            import chainladder as cl

        .. testcode::

            label = cl.GridSearch.leave_one_out_grid_labels(("1985", 60))
            print(label.name, label.tolist())

        .. testoutput::

            1985, 60 ['1985', 60]
        """
        if value is None:
            return pd.Series({"origin": None, "age": pd.NA}, name="base")
        origin, age = value
        return pd.Series({"origin": origin, "age": age}, name=f"{origin}, {age}")

    def compare_results(self, base, result_labels=None, result_comparisons=None):
        """
        Compare every candidate in ``results_`` with one base candidate.

        Parameters
        ----------
        base: label
            The candidate to compare with. With
            ``leave_one_out_grid_labels`` as ``result_labels`` this is
            ``"base"``. Without ``result_labels`` it is an index of
            ``results_``.
        result_labels: callable, optional
            Takes one value of the single ``param_grid`` key in ``results_``
            and returns the label columns for its row, as a dict or Series.
            When it returns a named Series, the name becomes the row label.
        result_comparisons: dict, optional
            Maps a ``scoring`` key to a list of comparison callables, such as
            ``{"ibnr": [cl.GridSearch.comparisons.change]}``. Each comparison
            takes the key, the compared rows, the base row and the label
            columns, and returns a named Series. A ``scoring`` key without
            comparisons keeps its column from ``results_``. Four comparisons
            are built in.

            - ``GridSearch.comparisons.change`` gives ``{key}_change``, each
              score less the base score.
            - ``GridSearch.comparisons.share`` gives ``{key}_share``, that
              change divided by the base score.
            - ``GridSearch.comparisons.base`` gives ``{key}_base``, the base
              factor at the ``age`` of each row.
            - ``GridSearch.comparisons.scenario`` gives ``{key}_scenario``,
              the candidate's own factor at that age.

            ``change`` and ``share`` need numeric scores, such as the ones
            ``GridSearch.ibnr`` returns. ``base`` and ``scenario`` need
            scores indexed by age, such as the ones ``GridSearch.ldf``
            returns, and an ``age`` label column, such as the one
            ``leave_one_out_grid_labels`` supplies. Without an ``age``
            column they give NaN, and ``scenario`` also gives NaN where the
            candidate has no factor at that age.

        Returns
        -------
        DataFrame
            One row per candidate other than the base, in the order of
            ``results_``, with the label columns first and then the comparison
            or score columns of each ``scoring`` key. Sort the table to rank
            the candidates.

        Raises
        ------
        ValueError
            If ``result_comparisons`` names a key that ``scoring`` lacks, if
            ``result_labels`` meets more than one ``param_grid`` key, or if
            ``result_labels`` gives two rows the same label.

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
            table = search.compare_results(
                base="base",
                result_labels=cl.GridSearch.leave_one_out_grid_labels,
                result_comparisons={
                    "ibnr": [cl.GridSearch.comparisons.change],
                    "ldf": [
                        cl.GridSearch.comparisons.base,
                        cl.GridSearch.comparisons.scenario,
                    ],
                },
            )
            table = table.sort_values("ibnr_change", key=abs, ascending=False)
            print(table[["ibnr_change", "ldf_base", "ldf_scenario"]].head(3).round(4))

        .. testoutput::

                      ibnr_change  ldf_base  ldf_scenario
            1985, 60    3362.8852    1.1134        1.1525
            1981, 96    2818.0421    1.0169        1.0331
            1981, 12    2467.6774    2.9994        3.4016
        """
        results = self.results_.copy()
        if isinstance(self.scoring, dict):
            scoring = self.scoring
        else:
            scoring = {"score": self.scoring}
        result_comparisons = {} if result_comparisons is None else result_comparisons
        unknown = [name for name in result_comparisons if name not in scoring]
        if unknown:
            raise ValueError(
                f"result_comparisons names {unknown}, which scoring does not have."
            )
        if result_labels is None:
            labels = pd.DataFrame(index=results.index)
        else:
            keys = [column for column in results.columns if column not in scoring]
            if len(keys) != 1:
                raise ValueError(
                    "result_labels needs results_ with one param_grid key, "
                    f"but results_ has {keys}."
                )
            labels = pd.DataFrame([result_labels(value) for value in results[keys[0]]])
            if not labels.index.is_unique:
                raise ValueError("result_labels gave two rows the same label.")
            results.index = labels.index
        base_row = results.loc[base]
        rows = results.drop(index=base)
        parts = [labels.loc[rows.index].infer_objects()]
        for name in scoring:
            if name in result_comparisons:
                parts.extend(
                    comparison(name, rows, base_row, labels)
                    for comparison in result_comparisons[name]
                )
            else:
                parts.append(rows[name])
        return pd.concat(parts, axis=1)


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
