import chainladder as cl


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


def test_groupby():
    clrd = cl.load_sample('clrd')
    clrd = clrd[clrd['LOB']=='comauto']
    # But only the top 10 get their own CapeCod aprioris. Smaller companies get grouped together
    top_10 = clrd['EarnedPremDIR'].groupby('GRNAME').sum().latest_diagonal
    top_10 = top_10.loc[..., '1997', :].to_frame().nlargest(10)
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
