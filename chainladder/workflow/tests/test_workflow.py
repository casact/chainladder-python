import chainladder as cl

def test_grid():
    # Load Data
    clrd = cl.load_dataset('clrd')
    medmal_paid = clrd.groupby('LOB').sum().loc['medmal']['CumPaidLoss']
    medmal_prem = clrd.groupby('LOB').sum().loc['medmal']['EarnedPremDIR'].latest_diagonal
    medmal_prem.rename(development='premium')

    # Pipeline
    dev = cl.Development()
    tail = cl.TailCurve()
    benk = cl.Benktander()

    steps = [('dev',dev), ('tail',tail), ('benk', benk)]
    pipe = cl.Pipeline(steps)

    # Prep Benktander Grid Search with various assumptions, and a scoring function
    param_grid = dict(n_iters=[250],
                      apriori=[1.00])
    scoring = {'IBNR':lambda x: x.ibnr_.sum()[0]}

    grid = cl.GridSearch(benk, param_grid, scoring=scoring)
    # Perform Grid Search
    grid.fit(medmal_paid, sample_weight=medmal_prem)
    assert grid.results_['IBNR'][0] == cl.Benktander(n_iters=250, apriori=1).fit(medmal_paid, sample_weight=medmal_prem).ibnr_.sum()[0]
