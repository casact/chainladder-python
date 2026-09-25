from __future__ import annotations

import chainladder as cl
import pytest
from functools import partial

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle
    from typing import Callable, Any


def test_grid(clrd: Triangle) -> None:
    """
    Test that GridSearch mirrors chaining chainladder estimators

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set fixture

    Returns
    -------
    None
    """
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
def tri(clrd: Triangle) -> Triangle:
    """
    Fixture for getting the desired aggregated slice of clrd

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set fixture

    Returns
    -------
    Triangle
    """
    tri = clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss", "EarnedPremDIR"]]
    tri["CaseIncurredLoss"] = tri["IncurLoss"] - tri["CumPaidLoss"]
    return tri


tri_sel = partial(cl.TriangleSelector, col="CumPaidLoss")
dev = [
    [tri_sel, cl.Development],
    [tri_sel, cl.ClarkLDF],
    [tri_sel, cl.Trend],
    [tri_sel, cl.IncrementalAdditive],
    [
        partial(
            cl.MunichAdjustment, paid_to_incurred=("CumPaidLoss", "CaseIncurredLoss")
        )
    ],
    [partial(cl.CaseOutstanding, paid_to_incurred=("CumPaidLoss", "CaseIncurredLoss"))],
]
tail = [cl.TailCurve, cl.TailConstant, cl.TailBondy, cl.TailClark]
ibnr = [
    cl.Chainladder,
    cl.BornhuetterFerguson,
    partial(cl.Benktander, n_iters=2),
    cl.CapeCod,
]


@pytest.mark.parametrize("dev", dev)
@pytest.mark.parametrize("tail", tail)
@pytest.mark.parametrize("ibnr", ibnr)
def test_pipeline(
    tri: Triangle,
    dev: list[Callable[[], Any]],
    tail: Callable[[], Any],
    ibnr: Callable[[], Any],
) -> None:
    """
    Test that Pipeline works across a wide combination of estimators

    Parameters
    ----------
    tri: Triangle
        Bespoke fixture for this test
    dev: list[Triangle Transformers]
        Develoopment Transformer, may be accompanied by other helper Transformers
    tail: Triangle Transformer
        Tail Curve Transformer
    ibnr: Triangle Predictor
        IBNR Predictor
    Returns
    -------
    None
    """
    X = tri[["CumPaidLoss", "CaseIncurredLoss"]]
    sample_weight = tri["EarnedPremDIR"].latest_diagonal
    (
        cl
        .Pipeline(
            steps=[
                *[(f"dev_{i}", x()) for i, x in enumerate(dev)],
                ("tail", tail()),
                ("ibnr", ibnr()),
            ]
        )
        .fit_predict(X, sample_weight=sample_weight)
        .ibnr_.sum("origin")
        .sum("columns")
        .sum()
    )
