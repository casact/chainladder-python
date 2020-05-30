import chainladder as cl

def test_struhuss():
    X = cl.load_sample('cc_sample')['loss']
    X = cl.TailConstant(tail=1/0.85).fit_transform(cl.Development().fit_transform(X))
    sample_weight = cl.load_sample('cc_sample')['exposure'].latest_diagonal
    ibnr = int(cl.CapeCod(trend=0.07, decay=0.75).fit(X, sample_weight=sample_weight).ibnr_.sum())
    assert ibnr == 17052
