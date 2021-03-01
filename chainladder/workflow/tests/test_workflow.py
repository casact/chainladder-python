import chainladder as cl
import itertools

def test_grid():
    # Load Data
    clrd = cl.load_sample("clrd")
    medmal_paid = clrd.groupby("LOB").sum().loc["medmal"]["CumPaidLoss"]
    medmal_prem = (
        clrd.groupby("LOB").sum().loc["medmal"]["EarnedPremDIR"].latest_diagonal
    )
    medmal_prem.rename("development", ["premium"])

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

def test_pipeline():
    tri = cl.load_sample('clrd').groupby('LOB').sum()[['CumPaidLoss', 'IncurLoss', 'EarnedPremDIR']]
    tri['CaseIncurredLoss'] = tri['IncurLoss'] - tri['CumPaidLoss']

    X = tri[['CumPaidLoss', 'CaseIncurredLoss']]
    sample_weight = tri['EarnedPremDIR'].latest_diagonal
    X_gt = X.copy()
    sample_weight_gt = sample_weight.copy()

    dev = [cl.Development(), cl.ClarkLDF(), cl.Trend(), cl.IncrementalAdditive(),
             cl.MunichAdjustment(paid_to_incurred=('CumPaidLoss', 'CaseIncurredLoss')),
             cl.CaseOutstanding(paid_to_incurred=('CumPaidLoss', 'CaseIncurredLoss'))]
    tail = [cl.TailCurve(), cl.TailConstant(), cl.TailBondy(), cl.TailClark()]
    ibnr = [cl.Chainladder(),  cl.BornhuetterFerguson(), cl.Benktander(n_iters=2), cl.CapeCod()]

    for model in list(itertools.product(dev, tail, ibnr)):
        assert X == X_gt
        assert sample_weight == sample_weight_gt
        cl.Pipeline(
            steps=[('dev', model[0]),
                   ('tail', model[1]),
                   ('ibnr', model[2])]
        ).fit_predict(X, sample_weight=sample_weight).ibnr_.sum('origin').sum('columns').sum()
