import chainladder as cl
import numpy as np
import pandas as pd
import pytest

raa = cl.load_sample("RAA")
raa_1989 = raa[raa.valuation < raa.valuation_date]
cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
apriori = cl_ult * 0 + (float(cl_ult.sum()) / 10)  # Mean Chainladder Ultimate
apriori_1989 = apriori[apriori.origin < "1990"]


@pytest.mark.parametrize(
    "estimators",
    [
        cl.CapeCod,
        cl.BornhuetterFerguson,
        cl.ExpectedLoss,
        cl.Benktander,
        cl.Chainladder,
    ],
)
def test_predict_and_weights(estimators, atol):
    est = estimators().fit(raa_1989, sample_weight=apriori_1989)
    pred = est.predict(raa, sample_weight=apriori)
    assert pred
    assert np.allclose(
        raa_1989.latest_diagonal.values.swapaxes(0, 1),
        cl.model_diagnostics(est)["Latest"].values,
        atol=atol,
        equal_nan=True,
    )
    assert np.allclose(
        raa.latest_diagonal.values.swapaxes(0, 1),
        cl.model_diagnostics(pred)["Latest"].values,
        atol=atol,
        equal_nan=True,
    )
    assert np.allclose(
        est.ultimate_.values.swapaxes(0, 1),
        cl.model_diagnostics(est)["Ultimate"].values,
        atol=atol,
        equal_nan=True,
    )
    assert np.allclose(
        pred.ultimate_.values.swapaxes(0, 1),
        cl.model_diagnostics(pred)["Ultimate"].values,
        atol=atol,
        equal_nan=True,
    )
    # Test validation of sample_weight requirement. Should raise a value error if no weight is supplied.
    if estimators in [
        cl.CapeCod,
        cl.BornhuetterFerguson,
        cl.ExpectedLoss,
        cl.Benktander,
    ]:
        assert np.allclose(
            est.expectation_.values.swapaxes(0, 1),
            cl.model_diagnostics(est)["Apriori"].values,
            atol=atol,
            equal_nan=True,
        )
        assert np.allclose(
            pred.expectation_.values.swapaxes(0, 1),
            cl.model_diagnostics(pred)["Apriori"].values,
            atol=atol,
            equal_nan=True,
        )
        with pytest.raises(ValueError):
            estimators().fit(raa_1989)
        with pytest.raises(ValueError):
            estimators().fit(raa_1989, sample_weight=apriori_1989).predict(raa)


def test_mack_predict():
    mack = cl.MackChainladder().fit(raa_1989)
    assert mack.predict(raa_1989)


def test_bs_random_state_predict(clrd):
    tri = clrd.groupby("LOB").sum().loc["wkcomp", ["CumPaidLoss", "EarnedPremNet"]]
    X = cl.BootstrapODPSample(random_state=100).fit_transform(tri["CumPaidLoss"])
    bf = cl.BornhuetterFerguson(apriori=0.6, apriori_sigma=0.1, random_state=42).fit(
        X, sample_weight=tri["EarnedPremNet"].latest_diagonal
    )
    assert (
        abs(
            bf
            .predict(X, sample_weight=tri["EarnedPremNet"].latest_diagonal)
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
    w = clrd["EarnedPremDIR"].latest_diagonal
    clrd = clrd["CumPaidLoss"]
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
        bbk
        .predict(clrd.iloc[150:153], sample_weight=w.iloc[150:153])
        .ultimate_.sum()
        .sum()
    )
    assert abs(a - b) < 1e-5
    a = bcc.ultimate_.iloc[150:153].sum().sum()
    b = (
        bcc
        .predict(clrd.iloc[150:153], sample_weight=w.iloc[150:153])
        .ultimate_.sum()
        .sum()
    )
    assert abs(a - b) < 1e-5

    a = bcl.ultimate_.iloc[150:152].sum().sum()
    b = bcl.predict(clrd.iloc[150:152]).ultimate_.sum().sum()
    assert abs(a - b) < 1e-5
    a = bbk.ultimate_.iloc[150:152].sum().sum()
    b = (
        bbk
        .predict(clrd.iloc[150:152], sample_weight=w.iloc[150:152])
        .ultimate_.sum()
        .sum()
    )
    assert abs(a - b) < 1e-5
    a = bcc.ultimate_.iloc[150:152].sum().sum()
    b = (
        bcc
        .predict(clrd.iloc[150:152], sample_weight=w.iloc[150:152])
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


def test_predict_rejects_index_values_the_model_never_saw(clrd):
    """github issue #1288

    intersection() narrows both operands to their shared index, so a group the
    model was never fit on used to be dropped from the result instead of
    predicted, silently on Chainladder and as a numpy shape error on the others.
    """
    tri = clrd["CumPaidLoss"]
    sample_weight = clrd["EarnedPremDIR"].latest_diagonal
    seen = tri.index["LOB"] != "wkcomp"
    train = tri[seen].groupby("LOB").sum()
    train_weight = sample_weight[seen].groupby("LOB").sum()

    models = [
        cl.Chainladder().fit(cl.Development().fit_transform(train)),
        cl.BornhuetterFerguson(apriori=0.7).fit(train, sample_weight=train_weight),
        cl.CapeCod().fit(train, sample_weight=train_weight),
    ]
    for model in models:
        with pytest.raises(ValueError, match="index values the model was not fit on"):
            if isinstance(model, cl.Chainladder):
                model.predict(tri)
            else:
                model.predict(tri, sample_weight=sample_weight)


def test_predict_rejects_a_different_single_group(clrd):
    """github issue #1288

    Both sides being one row is the case intersection() short circuits on, so
    the pattern from one group used to be applied to another silently, with the
    result labelled as the group that was predicted on.
    """
    lob = clrd["CumPaidLoss"].groupby("LOB").sum()
    fitted_on = lob[lob.index["LOB"] == "comauto"]
    predicted_on = lob[lob.index["LOB"] == "othliab"]
    assert len(fitted_on) == len(predicted_on) == 1

    weight = clrd["EarnedPremDIR"].latest_diagonal.groupby("LOB").sum()
    fitted_weight = weight[weight.index["LOB"] == "comauto"]
    predicted_weight = weight[weight.index["LOB"] == "othliab"]

    model = cl.Chainladder().fit(cl.Development().fit_transform(fitted_on))
    with pytest.raises(ValueError, match="index values the model was not fit on"):
        model.predict(predicted_on)
    # the same single group is still fine
    assert model.predict(fitted_on).ultimate_.shape[0] == 1

    for cls in (cl.BornhuetterFerguson, cl.CapeCod):
        exposure_model = cls().fit(fitted_on, sample_weight=fitted_weight)
        with pytest.raises(ValueError, match="index values the model was not fit on"):
            exposure_model.predict(predicted_on, sample_weight=predicted_weight)


def test_predict_rejects_a_pattern_finer_than_the_triangle(clrd):
    """github issue #1288

    The reverse direction: a pattern fit per company cannot be applied to a
    triangle that has aggregated companies away. intersection() leaves the ldf_
    at the fitted grain, which used to surface as a 775 row ultimate_ from a
    6 row input.
    """
    tri = clrd["CumPaidLoss"]
    sample_weight = clrd["EarnedPremDIR"].latest_diagonal
    coarse = tri.groupby("LOB").sum()
    coarse_weight = sample_weight.groupby("LOB").sum()

    models = [
        cl.Chainladder().fit(cl.Development().fit_transform(tri)),
        cl.BornhuetterFerguson(apriori=0.7).fit(tri, sample_weight=sample_weight),
        cl.CapeCod().fit(tri, sample_weight=sample_weight),
    ]
    for model in models:
        with pytest.raises(ValueError, match="has index levels that X does not"):
            if isinstance(model, cl.Chainladder):
                model.predict(coarse)
            else:
                model.predict(coarse, sample_weight=coarse_weight)


def test_predict_rejects_columns_the_model_was_not_fit_on(clrd):
    """github issue #1288

    The same mismatch on the columns axis. A paid pattern applied to an incurred
    triangle used to come back labelled incurred, overstating the ultimate by
    about half on clrd.
    """
    paid = clrd["CumPaidLoss"].groupby("LOB").sum()
    incurred = clrd["IncurLoss"].groupby("LOB").sum()
    both = clrd[["CumPaidLoss", "IncurLoss"]].groupby("LOB").sum()

    model = cl.Chainladder().fit(cl.Development().fit_transform(paid))
    with pytest.raises(ValueError, match="columns the model was not fit on"):
        model.predict(incurred)
    with pytest.raises(ValueError, match="columns the model was not fit on"):
        model.predict(both)

    # the column it was fit on is still fine
    assert model.predict(paid).ultimate_.shape[0] == paid.shape[0]


def test_predict_still_allows_an_aggregate_pattern(clrd):
    """github issue #1288

    A pattern fit on a fully aggregated triangle carries the "(All)" sentinel
    rather than any group identity, so it may be broadcast to any grain. Both
    directions matter: test_different_backends relies on the first.
    """
    tri = clrd["CumPaidLoss"]
    sample_weight = clrd["EarnedPremDIR"].latest_diagonal
    model = cl.BornhuetterFerguson().fit(tri.sum(), sample_weight=sample_weight.sum())

    per_company = model.predict(tri, sample_weight=sample_weight)
    assert per_company.ultimate_.shape[0] == tri.shape[0]

    by_lob = model.predict(
        tri.groupby("LOB").sum(), sample_weight=sample_weight.groupby("LOB").sum()
    )
    assert by_lob.ultimate_.shape[0] == tri.groupby("LOB").sum().shape[0]


def test_predict_checks_before_the_index_is_borrowed(clrd):
    """github issue #1288

    MethodBase.predict adds a zeroed slice of self.X_ to X, and for single row
    operands whose key_labels differ that addition takes the model's index. The
    check has to run before it, or by the time it looks the caller's identity is
    already the model's and the two agree.
    """
    tri = clrd["CumPaidLoss"]
    by_company = tri.groupby("GRNAME").sum()
    by_lob = tri.groupby("LOB").sum()
    fitted_on = by_company[by_company.index["GRNAME"] == "Aegis Grp"]
    predicted_on = by_lob[by_lob.index["LOB"] == "othliab"]
    assert fitted_on.key_labels != predicted_on.key_labels

    model = cl.Chainladder().fit(cl.Development().fit_transform(fitted_on))
    with pytest.raises(ValueError):
        model.predict(predicted_on)


def test_predict_still_allows_a_pattern_coarser_than_the_triangle(clrd):
    """github issue #1288

    The #400 flow must keep working: a pattern fit at a coarser grain broadcasts
    down to the triangle, so validate_ldf has to stay quiet here.
    """
    tri = clrd["CumPaidLoss"]
    sample_weight = clrd["EarnedPremDIR"].latest_diagonal
    train = tri.groupby("LOB").sum()
    train_weight = sample_weight.groupby("LOB").sum()

    bcl = cl.Chainladder().fit(cl.Development().fit_transform(train)).predict(tri)
    assert bcl.ultimate_.shape[0] == tri.shape[0]

    bcc = cl.CapeCod().fit(train, sample_weight=train_weight)
    assert (
        bcc.predict(tri, sample_weight=sample_weight).apriori_.shape[0]
        == train.shape[0]
    )


def test_align_cdfs(raa):
    ld = raa.latest_diagonal * 0 + 40000
    model = cl.BornhuetterFerguson().fit(raa, sample_weight=ld)
    a = model.ultimate_.iloc[..., :4, :]
    b = model.predict(
        raa.dev_to_val().iloc[..., :4, -1].val_to_dev(),
        sample_weight=ld.iloc[..., :4, :],
    ).ultimate_
    assert a == b
    model = cl.Chainladder().fit(raa, sample_weight=ld)
    a = model.ultimate_.iloc[..., :4, :]
    b = model.predict(
        raa.dev_to_val().iloc[..., :4, -1].val_to_dev(),
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
    ult1 = (
        cl
        .Chainladder()
        .fit(cl.Development(average="volume").fit_transform(tr.grain("OYDQ")))
        .ultimate_.sum()
    )
    ult2 = (
        cl
        .Chainladder()
        .fit(cl.Development(average="volume").fit_transform(tr))
        .ultimate_.grain("OYDQ")
        .sum()
    )
    assert abs(ult1 - ult2) < 1e-5
