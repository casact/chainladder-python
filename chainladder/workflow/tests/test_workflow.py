import chainladder as cl
import pytest

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
        == cl.Benktander(n_iters=250, apriori=1)
        .fit(
            cl.TailCurve().fit_transform(cl.Development().fit_transform(medmal_paid)),
            sample_weight=medmal_prem,
        )
        .ibnr_.sum()
    )


@pytest.fixture
def tri(clrd):
    tri = clrd.groupby('LOB').sum()[['CumPaidLoss', 'IncurLoss', 'EarnedPremDIR']]
    tri['CaseIncurredLoss'] = tri['IncurLoss'] - tri['CumPaidLoss']
    return tri

dev = [cl.Development, cl.ClarkLDF, cl.Trend, cl.IncrementalAdditive,
       lambda : cl.MunichAdjustment(paid_to_incurred=('CumPaidLoss', 'CaseIncurredLoss')),
       lambda :cl.CaseOutstanding(paid_to_incurred=('CumPaidLoss', 'CaseIncurredLoss'))]
tail = [cl.TailCurve, cl.TailConstant, cl.TailBondy, cl.TailClark]
ibnr = [cl.Chainladder,  cl.BornhuetterFerguson,
        lambda : cl.Benktander(n_iters=2), cl.CapeCod]

@pytest.mark.parametrize('dev', dev)
@pytest.mark.parametrize('tail', tail)
@pytest.mark.parametrize('ibnr', ibnr)
def test_pipeline(tri, dev, tail, ibnr):
    X = tri[['CumPaidLoss', 'CaseIncurredLoss']]
    sample_weight = tri['EarnedPremDIR'].latest_diagonal
    cl.Pipeline(
        steps=[('dev', dev()), ('tail', tail()), ('ibnr', ibnr())]
    ).fit_predict(X, sample_weight=sample_weight).ibnr_.sum('origin').sum('columns').sum()
