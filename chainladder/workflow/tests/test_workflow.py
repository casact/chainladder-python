import chainladder as cl

def test_grid():
    # Load Data
    clrd = cl.load_sample('clrd')
    medmal_paid = clrd.groupby('LOB').sum().loc['medmal']['CumPaidLoss']
    medmal_prem = clrd.groupby('LOB').sum().loc['medmal']['EarnedPremDIR'].latest_diagonal
    medmal_prem.rename('development',['premium'])

    # Pipeline
    dev = cl.Development()
    tail = cl.TailCurve()
    benk = cl.Benktander()

    steps = [('dev',dev), ('tail',tail), ('benk', benk)]
    pipe = cl.Pipeline(steps)

    # Prep Benktander Grid Search with various assumptions, and a scoring function
    param_grid = dict(benk__n_iters=[250],
                      benk__apriori=[1.00])
    scoring = {'IBNR':lambda x: x.named_steps.benk.ibnr_.sum()}

    grid = cl.GridSearch(pipe, param_grid, scoring=scoring)
    # Perform Grid Search
    grid.fit(medmal_paid, benk__sample_weight=medmal_prem)
    assert grid.results_['IBNR'][0] == \
        cl.Benktander(n_iters=250, apriori=1).fit(cl.TailCurve().fit_transform(cl.Development().fit_transform(medmal_paid)), sample_weight=medmal_prem).ibnr_.sum()
