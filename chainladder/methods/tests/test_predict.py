import chainladder as cl
import pandas as pd
import pytest 

raa = cl.load_sample("RAA")
raa_1989 = raa[raa.valuation < raa.valuation_date]
cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
apriori = cl_ult * 0 + (float(cl_ult.sum()) / 10)  # Mean Chainladder Ultimate
apriori_1989 = apriori[apriori.origin < "1990"]


def test_cc_predict():
    cc = cl.CapeCod().fit(raa_1989, sample_weight=apriori_1989)
    assert cc.predict(raa, sample_weight=apriori)

def test_bf_predict():
    bf = cl.BornhuetterFerguson().fit(raa_1989, sample_weight=apriori_1989)
    assert bf.predict(raa, sample_weight=apriori)

def test_el_predict():
    bf = cl.ExpectedLoss().fit(raa_1989, sample_weight=apriori_1989)
    assert bf.predict(raa, sample_weight=apriori)

def test_mack_predict():
    mack = cl.MackChainladder().fit(raa_1989)
    assert mack.predict(raa_1989)

def test_capecod_fit_weight():
    """
    Test validation of sample_weight requirement. Should raise a value error if no weight is supplied.
    """
    with pytest.raises(ValueError):
        cl.CapeCod().fit(raa_1989)

def test_capecod_predict_weight():
    """
    Test validation of sample_weight requirement. Should raise a value error if no weight is supplied.
    """
    with pytest.raises(ValueError):
        cc = cl.CapeCod().fit(raa_1989, sample_weight=apriori_1989)
        cc.predict(raa)

def test_bs_random_state_predict(clrd):
    tri = clrd.groupby("LOB").sum().loc["wkcomp", ["CumPaidLoss", "EarnedPremNet"]]
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


def test_basic_transform(raa):
    cl.Development().fit_transform(raa)
    cl.ClarkLDF().fit_transform(raa)
    cl.TailClark().fit_transform(raa)
    cl.TailBondy().fit_transform(raa)
    cl.TailConstant().fit_transform(raa)
    cl.TailCurve().fit_transform(raa)
    cl.BootstrapODPSample().fit_transform(raa)
    cl.IncrementalAdditive().fit_transform(raa, sample_weight=raa.latest_diagonal)


def test_misaligned_index(prism):
    prism = prism["Paid"]
    model = cl.Chainladder().fit(
        cl.Development(groupby=["Line", "Type"]).fit_transform(prism)
    )
    a = model.ultimate_.loc[prism.index.iloc[:10]].sum().sum()
    b = model.predict(prism.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5


def test_misaligned_index2(clrd):
    clrd = clrd["CumPaidLoss"]
    w = cl.load_sample("clrd")["EarnedPremDIR"].latest_diagonal
    bcl = cl.Chainladder().fit(cl.Development(groupby=["LOB"]).fit_transform(clrd))
    bbk = cl.Benktander().fit(
        cl.Development(groupby=["LOB"]).fit_transform(clrd), sample_weight=w
    )
    bcc = cl.CapeCod().fit(
        cl.Development(groupby=["LOB"]).fit_transform(clrd), sample_weight=w
    )

    a = bcl.ultimate_.iloc[:10].sum().sum()
    b = bcl.predict(clrd.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bbk.ultimate_.iloc[:10].sum().sum()
    b = bbk.predict(clrd.iloc[:10], sample_weight=w.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bcc.ultimate_.iloc[:10].sum().sum()
    b = bcc.predict(clrd.iloc[:10], sample_weight=w.iloc[:10]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5

    a = bcl.ultimate_.iloc[150:153].sum().sum()
    b = bcl.predict(clrd.iloc[150:153]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bbk.ultimate_.iloc[150:153].sum().sum()
    b = (
        bbk.predict(clrd.iloc[150:153], sample_weight=w.iloc[150:153])
        .ultimate_.sum()
        .sum()
    )
    assert abs(a - b) < 1e-5
    a = bcc.ultimate_.iloc[150:153].sum().sum()
    b = (
        bcc.predict(clrd.iloc[150:153], sample_weight=w.iloc[150:153])
        .ultimate_.sum()
        .sum()
    )
    assert abs(a - b) < 1e-5

    a = bcl.ultimate_.iloc[150:152].sum().sum()
    b = bcl.predict(clrd.iloc[150:152]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bbk.ultimate_.iloc[150:152].sum().sum()
    b = (
        bbk.predict(clrd.iloc[150:152], sample_weight=w.iloc[150:152])
        .ultimate_.sum()
        .sum()
    )
    assert abs(a - b) < 1e-5
    a = bcc.ultimate_.iloc[150:152].sum().sum()
    b = (
        bcc.predict(clrd.iloc[150:152], sample_weight=w.iloc[150:152])
        .ultimate_.sum()
        .sum()
    )
    assert abs(a - b) < 1e-5

    a = bcl.ultimate_.iloc[150].sum().sum()
    b = bcl.predict(clrd.iloc[150]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bbk.ultimate_.iloc[150].sum().sum()
    b = bbk.predict(clrd.iloc[150], sample_weight=w.iloc[150]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bcc.ultimate_.iloc[150].sum().sum()
    b = bcc.predict(clrd.iloc[150], sample_weight=w.iloc[150]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5


def test_align_cdfs():
    ld = cl.load_sample("raa").latest_diagonal * 0 + 40000
    model = cl.BornhuetterFerguson().fit(cl.load_sample("raa"), sample_weight=ld)
    a = model.ultimate_.iloc[..., :4, :]
    b = model.predict(
        cl.load_sample("raa").dev_to_val().iloc[..., :4, -1].val_to_dev(),
        sample_weight=ld.iloc[..., :4, :],
    ).ultimate_
    assert a == b
    model = cl.Chainladder().fit(cl.load_sample("raa"), sample_weight=ld)
    a = model.ultimate_.iloc[..., :4, :]
    b = model.predict(
        cl.load_sample("raa").dev_to_val().iloc[..., :4, -1].val_to_dev(),
        sample_weight=ld.iloc[..., :4, :],
    ).ultimate_
    assert a == b


def test_check_val_tri_cl(raa):
    model = cl.Chainladder().fit(raa.dev_to_val())
    assert model.predict(raa.latest_diagonal).ultimate_ == model.ultimate_

def test_odd_shaped_triangle():
    df = pd.DataFrame({
        "claim_year": 2000 + pd.Series([0] * 8 + [1] * 4),
        "claim_month": [1, 4, 7, 10] * 3,
        "dev_year": 2000 + pd.Series([0] * 4 + [1] * 8),
        "dev_month": [1, 4, 7, 10] * 3,
        "payment": [1] * 12,
    })
    tr = cl.Triangle(
        df,
        origin=["claim_year", "claim_month"],
        development=["dev_year", "dev_month"],
        columns="payment",
        cumulative=False,
    )
    ult1 = cl.Chainladder().fit(cl.Development(average="volume").fit_transform(tr.grain("OYDQ"))).ultimate_.sum()
    ult2 = cl.Chainladder().fit(cl.Development(average="volume").fit_transform(tr)).ultimate_.grain("OYDQ").sum()
    assert abs(ult1 - ult2) < 1e-5