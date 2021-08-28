import chainladder as cl

raa = cl.load_sample("RAA")
raa_1989 = raa[raa.valuation < raa.valuation_date]
cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
apriori = cl_ult * 0 + (float(cl_ult.sum()) / 10)  # Mean Chainladder Ultimate
apriori_1989 = apriori[apriori.origin < "1990"]


def test_cc_predict():
    cc = cl.CapeCod().fit(raa_1989, sample_weight=apriori_1989)
    cc.predict(raa, sample_weight=apriori)


def test_bf_predict():
    cc = cl.BornhuetterFerguson().fit(raa_1989, sample_weight=apriori_1989)
    cc.predict(raa, sample_weight=apriori)


def test_mack_predict():
    mack = cl.MackChainladder().fit(raa_1989)
    mack.predict(raa_1989)
    # mack.predict(raa)


def test_bs_random_state_predict():
    tri = (
        cl.load_sample("clrd")
        .groupby("LOB")
        .sum()
        .loc["wkcomp", ["CumPaidLoss", "EarnedPremNet"]]
    )
    X = cl.BootstrapODPSample(random_state=100).fit_transform(tri["CumPaidLoss"])
    bf = cl.BornhuetterFerguson(apriori=0.6, apriori_sigma=0.1, random_state=42).fit(
        X, sample_weight=tri["EarnedPremNet"].latest_diagonal
    )
    assert (
        abs(
            bf.predict(X, sample_weight=tri["EarnedPremNet"].latest_diagonal)
            .ibnr_.sum()
            .sum()
            / bf.ibnr_.sum().sum()
            - 1
        )
        < 5e-3
    )


def test_basic_transform():
    tri = cl.load_sample("raa")
    cl.Development().fit_transform(tri)
    cl.ClarkLDF().fit_transform(tri)
    cl.TailClark().fit_transform(tri)
    cl.TailBondy().fit_transform(tri)
    cl.TailConstant().fit_transform(tri)
    cl.TailCurve().fit_transform(tri)
    cl.BootstrapODPSample().fit_transform(tri)
    cl.IncrementalAdditive().fit_transform(tri, sample_weight=tri.latest_diagonal)

def test_misaligned_index():
    prism = cl.load_sample('prism')['Paid']
    model = cl.Chainladder().fit(cl.Development(groupby=['Line', 'Type']).fit_transform(prism))
    a = model.ultimate_.loc[prism.index.iloc[:10]].sum().sum()
    b = model.predict(prism.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
   

def test_misaligned_index2():
    clrd = cl.load_sample('clrd')['CumPaidLoss']
    w = cl.load_sample('clrd')['EarnedPremDIR']
    bcl = cl.Chainladder().fit(cl.Development(groupby=['LOB']).fit_transform(clrd))
    bbk = cl.Benktander().fit(cl.Development(groupby=['LOB']).fit_transform(clrd), sample_weight=w)
    bcc = cl.CapeCod().fit(cl.Development(groupby=['LOB']).fit_transform(clrd), sample_weight=w)

    a = bcl.ultimate_.loc[clrd.index.iloc[:10]].sum().sum()
    b = bcl.predict(clrd.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bbk.ultimate_.loc[clrd.index.iloc[:10]].sum().sum()
    b = bbk.predict(clrd.iloc[:10], sample_weight=w.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bcc.ultimate_.loc[clrd.index.iloc[:10]].sum().sum()
    b = bcc.predict(clrd.iloc[:10], sample_weight=w.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5

    X = clrd.iloc[150:153]
    sample_weight=w.iloc[150:153]
    assert (bcl.predict(X).ultimate_.index == X.index).values.all()
    assert (bbk.predict(X, sample_weight=sample_weight).ultimate_.index == X.index).values.all()
    assert (bcc.predict(X, sample_weight=sample_weight).ultimate_.index == X.index).values.all()
    X = clrd.iloc[150:152]
    sample_weight=w.iloc[150:152]
    assert (bcl.predict(X).ultimate_.index == X.index).values.all()
    assert (bbk.predict(X, sample_weight=sample_weight).ultimate_.index == X.index).values.all()
    assert (bcc.predict(X, sample_weight=sample_weight).ultimate_.index == X.index).values.all()
    X = clrd.iloc[150]
    sample_weight=w.iloc[150]
    assert (bcl.predict(X).ultimate_.index == X.index).values.all()
    assert (bbk.predict(X, sample_weight=sample_weight).ultimate_.index == X.index).values.all()
    assert (bcc.predict(X, sample_weight=sample_weight).ultimate_.index == X.index).values.all()