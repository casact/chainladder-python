import chainladder as cl

def test_trend1():
    tri = cl.load_sample('clrd')[['CumPaidLoss', 'EarnedPremDIR']].sum()
    assert (
        cl.CapeCod(.05).fit(tri['CumPaidLoss'], sample_weight=tri['EarnedPremDIR'].latest_diagonal).ibnr_ ==
        cl.CapeCod().fit(cl.Trend(.05).fit_transform(tri['CumPaidLoss']), sample_weight=tri['EarnedPremDIR'].latest_diagonal).ibnr_)

def test_trend2():
    tri = cl.load_sample('raa')
    assert abs(
        cl.Trend(trends=[.05, .05], dates=[(None, '1985'), ('1985', None)], axis='origin').fit(tri).trend_*tri -
        tri.trend(.05, axis='origin')).sum().sum() < 1e-6
