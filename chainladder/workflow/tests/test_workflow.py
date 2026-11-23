import chainladder as cl
import numpy as np
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


def test_leave_one_out_base_row_matches_plain_fit(loo_pipe, loo_search):
    """The base candidate drops nothing, so it reproduces a plain fit."""
    raa = cl.load_sample("raa")
    plain = clone(loo_pipe).fit(raa)
    assert loo_search.results_["dev__drop"][0] is None
    assert loo_search.results_["ibnr"][0] == float(plain.named_steps.model.ibnr_.sum())


def test_leave_one_out_grid_covers_every_observed_ratio():
    """The grid holds the base plus one entry per observed link ratio."""
    raa = cl.load_sample("raa")
    grid = cl.GridSearch.leave_one_out_grid(raa)
    observed = int(np.sum(~np.isnan(raa.link_ratio.values)))
    assert len(grid["dev__drop"]) == observed + 1
    assert grid.key == "dev__drop"
    assert grid["dev__drop"][1] == ("1981", 12)
    assert list(cl.GridSearch.leave_one_out_grid(raa, step="ldf")) == ["ldf__drop"]


def test_leave_one_out_grid_rejects_more_than_one_column(clrd):
    """A triangle with several columns has no single set of link ratios."""
    with pytest.raises(ValueError):
        cl.GridSearch.leave_one_out_grid(clrd.sum()[["CumPaidLoss", "IncurLoss"]])


def test_leave_one_out_grid_survives_clone(loo_search):
    """clone keeps the grid type and labels, so GridSearch stays an sklearn estimator."""
    copied = clone(loo_search)
    assert isinstance(copied.param_grid, cl.workflow.gridsearch.LeaveOneOutGrid)
    assert copied.param_grid.labels.equals(loo_search.param_grid.labels)
    assert set(loo_search.get_params(deep=False)) == {
        "error_score",
        "estimator",
        "n_jobs",
        "param_grid",
        "scoring",
        "verbose",
    }


def test_compare_results_orders_by_ibnr_change(loo_search):
    """The ratio that moves IBNR the most comes first, labelled by origin and age."""
    table = loo_search.compare_results("base")
    assert len(table) == 45
    top = table.iloc[0]
    assert table.index[0] == "1985, 60"
    assert (top["origin"], top["age"]) == ("1985", 60)
    assert top["ibnr_change"] == pytest.approx(3362.8852, abs=1e-3)
    assert top["ldf_before"] == pytest.approx(1.1134, abs=1e-4)
    assert top["ldf_after"] == pytest.approx(1.1525, abs=1e-4)
    assert table["ibnr_change"].abs().is_monotonic_decreasing


def test_compare_results_marks_single_ratio_ages(loo_search):
    """Dropping the only ratio at an age leaves no factor, so ldf_after is NaN."""
    row = loo_search.compare_results("base").loc["1981, 108"]
    assert np.isnan(row["ldf_after"])
    assert row["ldf_before"] == pytest.approx(1.0092, abs=1e-4)


def test_compare_results_keeps_raw_index_for_other_grids(loo_pipe):
    """An ordinary grid keeps its results_ index and its own scorer columns."""
    raa = cl.load_sample("raa")
    search = cl.GridSearch(
        loo_pipe,
        {"dev__average": ["volume", "simple"]},
        scoring={"ibnr": cl.GridSearch.ibnr, "n": lambda model: 1},
    ).fit(raa)
    table = search.compare_results(0)
    assert list(table.index) == [1]
    assert list(table.columns) == ["ibnr_change", "ibnr_share", "n"]
