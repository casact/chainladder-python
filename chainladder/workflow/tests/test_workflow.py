import chainladder as cl
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone


def test_grid(clrd):
    # Load Data
    medmal_paid = clrd.groupby("LOB").sum().loc["medmal"]["CumPaidLoss"]
    medmal_prem = (
        clrd.groupby("LOB").sum().loc["medmal"]["EarnedPremDIR"].latest_diagonal
    )

    # Pipeline
    dev = cl.Development()
    tail = cl.TailCurve()
    benk = cl.Benktander()

    steps = [("dev", dev), ("tail", tail), ("benk", benk)]
    pipe = cl.Pipeline(steps)

    # Prep Benktander Grid Search with various assumptions, and a scoring function
    param_grid = dict(benk__n_iters=[250], benk__apriori=[1.00])
    scoring = {"IBNR": lambda x: x.named_steps.benk.ibnr_.sum()}

    grid = cl.GridSearch(pipe, param_grid, scoring=scoring)
    # Perform Grid Search
    grid.fit(medmal_paid, benk__sample_weight=medmal_prem)
    assert (
        grid.results_["IBNR"][0]
        == cl
        .Benktander(n_iters=250, apriori=1)
        .fit(
            cl.TailCurve().fit_transform(cl.Development().fit_transform(medmal_paid)),
            sample_weight=medmal_prem,
        )
        .ibnr_.sum()
    )


@pytest.fixture
def tri(clrd):
    tri = clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss", "EarnedPremDIR"]]
    tri["CaseIncurredLoss"] = tri["IncurLoss"] - tri["CumPaidLoss"]
    return tri


dev = [
    cl.Development,
    cl.ClarkLDF,
    cl.Trend,
    cl.IncrementalAdditive,
    lambda: cl.MunichAdjustment(paid_to_incurred=("CumPaidLoss", "CaseIncurredLoss")),
    lambda: cl.CaseOutstanding(paid_to_incurred=("CumPaidLoss", "CaseIncurredLoss")),
]
tail = [cl.TailCurve, cl.TailConstant, cl.TailBondy, cl.TailClark]
ibnr = [
    cl.Chainladder,
    cl.BornhuetterFerguson,
    lambda: cl.Benktander(n_iters=2),
    cl.CapeCod,
]


@pytest.mark.parametrize("dev", dev)
@pytest.mark.parametrize("tail", tail)
@pytest.mark.parametrize("ibnr", ibnr)
def test_pipeline(tri, dev, tail, ibnr):
    X = tri[["CumPaidLoss", "CaseIncurredLoss"]]
    sample_weight = tri["EarnedPremDIR"].latest_diagonal
    cl.Pipeline(steps=[("dev", dev()), ("tail", tail()), ("ibnr", ibnr())]).fit_predict(
        X, sample_weight=sample_weight
    ).ibnr_.sum("origin").sum("columns").sum()


@pytest.fixture(scope="module")
def loo_pipe():
    return cl.Pipeline([("dev", cl.Development()), ("model", cl.Chainladder())])


@pytest.fixture(scope="module")
def loo_search(loo_pipe):
    raa = cl.load_sample("raa")
    return cl.GridSearch(
        loo_pipe,
        cl.GridSearch.leave_one_out_grid(raa),
        scoring={"ibnr": cl.GridSearch.ibnr, "ldf": cl.GridSearch.ldf},
    ).fit(raa)


LOO_METRICS = {
    "ibnr": [cl.GridSearch.metrics.change, cl.GridSearch.metrics.share],
    "ldf": [cl.GridSearch.metrics.before, cl.GridSearch.metrics.after],
}


@pytest.fixture(scope="module")
def loo_table(loo_search):
    return loo_search.compare_results(
        base="base",
        result_labels=cl.GridSearch.leave_one_out_grid_labels,
        result_metrics=LOO_METRICS,
    )


def test_leave_one_out_base_row_matches_plain_fit(loo_pipe, loo_search):
    """The base candidate drops nothing, so it reproduces a plain fit."""
    raa = cl.load_sample("raa")
    plain = clone(loo_pipe).fit(raa)
    assert loo_search.results_["dev__drop"][0] is None
    assert loo_search.results_["ibnr"][0] == float(plain.named_steps.model.ibnr_.sum())


def test_leave_one_out_grid_covers_every_observed_ratio():
    """The grid holds the base plus one (origin, age, link_ratio) per observed ratio."""
    raa = cl.load_sample("raa")
    grid = cl.GridSearch.leave_one_out_grid(raa)
    observed = int(np.sum(~np.isnan(raa.link_ratio.values)))
    assert list(grid) == ["dev__drop"]
    assert len(grid["dev__drop"]) == observed + 1
    origin, age, link_ratio = grid["dev__drop"][1]
    assert (origin, age) == ("1981", 12)
    assert link_ratio == raa.link_ratio.values[0, 0, 0, 0]
    assert list(cl.GridSearch.leave_one_out_grid(raa, step="ldf")) == ["ldf__drop"]


def test_leave_one_out_grid_ratio_does_not_change_the_drop():
    """Development drops the same ratio whether or not the value carries the ratio."""
    raa = cl.load_sample("raa")
    origin, age, link_ratio = cl.GridSearch.leave_one_out_grid(raa)["dev__drop"][5]
    with_ratio = cl.Development(drop=(origin, age, link_ratio)).fit(raa)
    without_ratio = cl.Development(drop=(origin, age)).fit(raa)
    assert np.array_equal(
        with_ratio.ldf_.values, without_ratio.ldf_.values, equal_nan=True
    )


def test_leave_one_out_grid_rejects_more_than_one_column(clrd):
    """A triangle with several columns has no single set of link ratios."""
    with pytest.raises(ValueError):
        cl.GridSearch.leave_one_out_grid(clrd.sum()[["CumPaidLoss", "IncurLoss"]])


def test_leave_one_out_grid_survives_clone(loo_search):
    """clone keeps the grid, so GridSearch stays an sklearn estimator."""
    copied = clone(loo_search)
    assert copied.param_grid == loo_search.param_grid
    assert set(loo_search.get_params(deep=False)) == {
        "error_score",
        "estimator",
        "n_jobs",
        "param_grid",
        "scoring",
        "verbose",
    }


def test_leave_one_out_grid_labels():
    """The labels name each value by origin and age and fill a missing ratio with NaN."""
    base = cl.GridSearch.leave_one_out_grid_labels(None)
    assert base.name == "base"
    assert np.isnan(base["link_ratio"])
    full = cl.GridSearch.leave_one_out_grid_labels(("1985", 60, 1.0087))
    assert full.name == "1985, 60"
    assert full.tolist() == ["1985", 60, 1.0087]
    pair = cl.GridSearch.leave_one_out_grid_labels(("1985", 60))
    assert pair.name == "1985, 60"
    assert np.isnan(pair["link_ratio"])


def test_compare_results_keeps_results_order(loo_search, loo_table):
    """compare_results leaves the rows in results_ order and labels them by origin and age."""
    assert len(loo_table) == 45
    assert list(loo_table.columns) == [
        "origin",
        "age",
        "link_ratio",
        "ibnr_change",
        "ibnr_share",
        "ldf_before",
        "ldf_after",
    ]
    assert loo_table.index[0] == "1981, 12"
    assert list(loo_table["link_ratio"]) == [
        value[2] for value in loo_search.results_["dev__drop"][1:]
    ]


def test_compare_results_sorted_by_ibnr_change(loo_table):
    """Sorting by the absolute IBNR change puts 1985 at 60 first."""
    table = loo_table.sort_values("ibnr_change", key=abs, ascending=False)
    top = table.iloc[0]
    assert table.index[0] == "1985, 60"
    assert (top["origin"], top["age"]) == ("1985", 60)
    assert top["ibnr_change"] == pytest.approx(3362.8852, abs=1e-3)
    assert top["ldf_before"] == pytest.approx(1.1134, abs=1e-4)
    assert top["ldf_after"] == pytest.approx(1.1525, abs=1e-4)


def test_compare_results_marks_single_ratio_ages(loo_table):
    """Dropping the only ratio at an age leaves no factor, so ldf_after is NaN."""
    row = loo_table.loc["1981, 108"]
    assert np.isnan(row["ldf_after"])
    assert row["ldf_before"] == pytest.approx(1.0092, abs=1e-4)


def test_compare_results_without_metrics_keeps_scores(loo_search):
    """A scoring key without metrics keeps its column from results_."""
    table = loo_search.compare_results(
        "base",
        result_labels=cl.GridSearch.leave_one_out_grid_labels,
        result_metrics={"ldf": [cl.GridSearch.metrics.after]},
    )
    assert list(table.columns) == ["origin", "age", "link_ratio", "ibnr", "ldf_after"]
    assert table["ibnr"].tolist() == loo_search.results_["ibnr"][1:].tolist()


def test_compare_results_keeps_raw_index_for_other_grids(loo_pipe):
    """An ordinary grid keeps its results_ index and its own scorer columns."""
    raa = cl.load_sample("raa")
    search = cl.GridSearch(
        loo_pipe,
        {"dev__average": ["volume", "simple"]},
        scoring={"ibnr": cl.GridSearch.ibnr, "n": lambda model: 1},
    ).fit(raa)
    table = search.compare_results(0, result_metrics={"ibnr": LOO_METRICS["ibnr"]})
    assert list(table.index) == [1]
    assert list(table.columns) == ["ibnr_change", "ibnr_share", "n"]


def test_compare_results_ldf_on_an_ordinary_grid_has_no_age(loo_pipe):
    """Without age labels there is no dropped age, so the ldf columns are NaN."""
    raa = cl.load_sample("raa")
    search = cl.GridSearch(
        loo_pipe,
        {"dev__average": ["volume", "simple"]},
        scoring={"ldf": cl.GridSearch.ldf},
    ).fit(raa)
    table = search.compare_results(0, result_metrics={"ldf": LOO_METRICS["ldf"]})
    assert list(table.index) == [1]
    assert list(table.columns) == ["ldf_before", "ldf_after"]
    assert table.isna().all().all()


def test_compare_results_accepts_a_single_scorer(loo_pipe):
    """A scoring callable that is not a dict keeps its score column and the results_ order."""
    raa = cl.load_sample("raa")
    search = cl.GridSearch(
        loo_pipe,
        {"dev__average": ["volume", "simple", "regression"]},
        scoring=lambda model: float(model.named_steps.model.ibnr_.sum()),
    ).fit(raa)
    table = search.compare_results(0)
    assert list(table.index) == [1, 2]
    assert list(table.columns) == ["score"]
    assert table["score"].equals(search.results_["score"].iloc[1:])


def test_compare_results_accepts_labels_as_dicts(loo_pipe):
    """A result_labels callable that returns a dict keeps the results_ index."""
    raa = cl.load_sample("raa")
    search = cl.GridSearch(
        loo_pipe,
        {"dev__average": ["volume", "simple"]},
        scoring={"ibnr": cl.GridSearch.ibnr},
    ).fit(raa)
    table = search.compare_results(0, result_labels=lambda value: {"average": value})
    assert list(table.index) == [1]
    assert table.loc[1, "average"] == "simple"


def test_compare_results_rejects_unknown_metric_key(loo_search):
    """result_metrics may only name keys that scoring has."""
    with pytest.raises(ValueError, match="scoring does not have"):
        loo_search.compare_results(
            "base", result_metrics={"ultimate": [cl.GridSearch.metrics.change]}
        )


def test_compare_results_labels_need_one_grid_key(loo_pipe):
    """result_labels reads one param_grid key, so two keys raise an error."""
    raa = cl.load_sample("raa")
    search = cl.GridSearch(
        loo_pipe,
        {"dev__average": ["volume", "simple"], "dev__n_periods": [-1, 5]},
        scoring={"ibnr": cl.GridSearch.ibnr},
    ).fit(raa)
    with pytest.raises(ValueError, match="one param_grid key"):
        search.compare_results(0, result_labels=lambda value: {"value": value})


def test_compare_results_rejects_duplicate_labels(loo_search):
    """Two rows with the same label cannot be compared."""
    with pytest.raises(ValueError, match="same label"):
        loo_search.compare_results(
            "base", result_labels=lambda value: pd.Series({"x": 1}, name="same")
        )
