import chainladder as cl
import numpy as np

def test_struhuss():
    X = cl.load_sample("cc_sample")["loss"]
    X = cl.TailConstant(tail=1 / 0.85).fit_transform(cl.Development().fit_transform(X))
    sample_weight = cl.load_sample("cc_sample")["exposure"].latest_diagonal
    ibnr = int(
        cl.CapeCod(trend=0.07, decay=0.75)
        .fit(X, sample_weight=sample_weight)
        .ibnr_.sum()
    )
    assert ibnr == 17052


def test_groupby(clrd):
    clrd = clrd[clrd['LOB']=='comauto']
    # But only the top 10 get their own CapeCod aprioris. Smaller companies get grouped together
    top_10 = clrd['EarnedPremDIR'].groupby('GRNAME').sum().latest_diagonal
    top_10 = top_10.loc[..., '1997', :].to_frame(origin_as_datetime=True).nlargest(10)
    cc_groupby = clrd.index['GRNAME'].map(lambda x: x if x in top_10.index else 'Remainder')
    idx = clrd.index
    idx['Top 10'] = cc_groupby
    clrd.index = idx

    # All companies share the same development factors regardless of size
    X = cl.Development().fit(clrd['CumPaidLoss'].sum()).transform(clrd['CumPaidLoss'])
    sample_weight=clrd['EarnedPremDIR'].latest_diagonal
    a = cl.CapeCod(groupby='Top 10', decay=0.98, trend=0.02).fit(X, sample_weight=sample_weight).ibnr_.groupby('Top 10').sum().sort_index()
    b = cl.CapeCod(decay=0.98, trend=0.02).fit(X.groupby('Top 10').sum(), sample_weight=sample_weight.groupby('Top 10').sum()).ibnr_.sort_index()
    xp = a.get_array_module()
    b = b.set_backend(a.array_backend)
    xp.allclose(xp.nan_to_num(a.values), xp.nan_to_num(b.values), atol=1e-5)


def test_capecod_zero_tri(raa):
    premium = raa.latest_diagonal * 0 + 50000
    raa.loc[:,:,'1987',48] = 0
    assert cl.CapeCod().fit(raa, sample_weight=premium).ultimate_.loc[:,:,'1987'].sum() > 0


def test_capecod_predict1(prism):
    """ github issue #400 
    Test whether we can make predictions at a more granular level than is fitted
    """
    prism = prism[['reportedCount', 'Paid']]

    cc_pipe = cl.Pipeline(
        [('dev', cl.Development()),
        ('model', cl.CapeCod())]
    )
    cc_pipe.fit(
        X=prism.groupby('Line')['Paid'].sum(), 
        sample_weight=prism.groupby('Line')['reportedCount'].sum().sum('development'))

    assert abs(cc_pipe.predict(prism['Paid'], sample_weight=prism['reportedCount'].sum('development')).ultimate_.sum() - 
            cc_pipe.named_steps.model.ultimate_.sum()).sum() < 1e-6
            

def test_capecod_predict2(prism):
    """ github issue #400 
    Test whether predictions between groupby with estimator and
    groupby outside estimator match
    """
    prism = prism[['reportedCount', 'Paid']]

    pipe1 = cl.Pipeline(
        [('dev', cl.Development(groupby='Line')),
        ('model', cl.CapeCod(groupby='Line'))]
    )
    pipe1.fit(
        X=prism['Paid'], 
        sample_weight=prism['reportedCount'].sum('development'))

    pipe2 = cl.Pipeline(
        [('dev', cl.Development()),
        ('model', cl.CapeCod())]
    )
    pipe2.fit(
        X=prism.groupby('Line')['Paid'].sum(), 
        sample_weight=prism.groupby('Line')['reportedCount'].sum().sum('development'))

    pred1 = pipe1.named_steps.model.ultimate_.sum()
    pred2 = pipe2.predict(prism['Paid'], sample_weight=prism['reportedCount'].sum('development')).ultimate_.sum()

    assert np.nan_to_num(abs(pred1 - pred2).values).sum() <= 1e-6