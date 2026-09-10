import chainladder as cl
import numpy as np


def test_struhuss():
    X = cl.load_sample("cc_sample")["loss"]
    X = cl.TailConstant(tail=1 / 0.85).fit_transform(cl.Development().fit_transform(X))
    sample_weight = cl.load_sample("cc_sample")["exposure"].latest_diagonal
    ibnr = int(
        cl
        .CapeCod(trend=0.07, decay=0.75)
        .fit(X, sample_weight=sample_weight)
        .ibnr_.sum()
    )
    assert ibnr == 17052


def test_groupby(clrd):
    clrd = clrd[clrd["LOB"] == "comauto"]
    # But only the top 10 get their own CapeCod aprioris. Smaller companies get grouped together
    top_10 = clrd["EarnedPremDIR"].groupby("GRNAME").sum().latest_diagonal
    top_10 = top_10.loc[..., "1997", :].to_frame(origin_as_datetime=True).nlargest(10)
    cc_groupby = clrd.index["GRNAME"].map(
        lambda x: x if x in top_10.index else "Remainder"
    )
    idx = clrd.index
    idx["Top 10"] = cc_groupby
    clrd.index = idx

    # All companies share the same development factors regardless of size
    X = cl.Development().fit(clrd["CumPaidLoss"].sum()).transform(clrd["CumPaidLoss"])
    sample_weight = clrd["EarnedPremDIR"].latest_diagonal
    a = (
        cl
        .CapeCod(groupby="Top 10", decay=0.98, trend=0.02)
        .fit(X, sample_weight=sample_weight)
        .ibnr_.groupby("Top 10")
        .sum()
        .sort_index()
    )
    b = (
        cl
        .CapeCod(decay=0.98, trend=0.02)
        .fit(
            X.groupby("Top 10").sum(),
            sample_weight=sample_weight.groupby("Top 10").sum(),
        )
        .ibnr_.sort_index()
    )
    xp = a.get_array_module()
    b = b.set_backend(a.array_backend)
    xp.allclose(xp.nan_to_num(a.values), xp.nan_to_num(b.values), atol=1e-5)


def test_capecod_zero_tri(raa):
    premium = raa.latest_diagonal * 0 + 50000
    raa.at["Total", "values", "1987", 48] = 0
    assert (
        cl.CapeCod().fit(raa, sample_weight=premium).ultimate_.loc[:, :, "1987"].sum()
        > 0
    )


def test_capecod_predict1(prism):
    """
    github issue #400
    Test whether we can make predictions at a more granular level than is fitted
    """
    prism = prism[["reportedCount", "Paid"]]

    cc_pipe = cl.Pipeline([("dev", cl.Development()), ("model", cl.CapeCod())])
    cc_pipe.fit(
        X=prism.groupby("Line")["Paid"].sum(),
        sample_weight=prism.groupby("Line")["reportedCount"].sum().sum("development"),
    )

    assert (
        abs(
            cc_pipe.predict(
                prism["Paid"], sample_weight=prism["reportedCount"].sum("development")
            ).ultimate_.sum()
            - cc_pipe.named_steps.model.ultimate_.sum()
        ).sum()
        < 1e-6
    )


def test_capecod_predict2(prism):
    """
    github issue #400
    Test whether predictions between groupby with estimator and
    groupby outside estimator match
    """
    prism = prism[["reportedCount", "Paid"]]

    pipe1 = cl.Pipeline([
        ("dev", cl.Development(groupby="Line")),
        ("model", cl.CapeCod(groupby="Line")),
    ])
    pipe1.fit(X=prism["Paid"], sample_weight=prism["reportedCount"].sum("development"))

    pipe2 = cl.Pipeline([("dev", cl.Development()), ("model", cl.CapeCod())])
    pipe2.fit(
        X=prism.groupby("Line")["Paid"].sum(),
        sample_weight=prism.groupby("Line")["reportedCount"].sum().sum("development"),
    )

    pred1 = pipe1.named_steps.model.ultimate_.sum()
    pred2 = pipe2.predict(
        prism["Paid"], sample_weight=prism["reportedCount"].sum("development")
    ).ultimate_.sum()

    assert np.nan_to_num(abs(pred1 - pred2).values).sum() <= 1e-6


def test_capecod_predict_one_extra_index_level(clrd):
    """
    github issue #1265

    predict() aggregates the prediction data up to the grain the model was fit
    at. test_capecod_predict2 covers that path with prism, whose triangle has
    five index levels more than the fitted model. This covers the case of a
    single extra level, which clrd gives.
    """
    tri = clrd["CumPaidLoss"]
    sample_weight = clrd["EarnedPremDIR"].latest_diagonal

    model = cl.CapeCod().fit(
        tri.groupby("LOB").sum(), sample_weight=sample_weight.groupby("LOB").sum()
    )
    pred = model.predict(tri, sample_weight=sample_weight)

    assert set(sample_weight.key_labels) - set(model.apriori_.key_labels) == {"GRNAME"}
    assert np.allclose(pred.apriori_.values, model.apriori_.values)
    assert abs(pred.ultimate_.sum().sum() - model.ultimate_.sum().sum()) < 1e-6


def test_capecod_onlevel_friedland_exhibit_ii():
    """
    github issue #1308, #1319
    Verify CapeCod on-leveling with ParallelogramOLF on both loss (tort reform)
    and exposure (rate changes) against Friedland Chapter 10 Exhibit II.
    """
    import os
    import pandas as pd

    xyz = cl.load_sample("friedland_xyz_auto_bi")
    xyz_reported = xyz["Reported Claims"]
    xyz_premium = xyz["Earned Premium"].latest_diagonal

    _data_dir = os.path.join(os.path.dirname(cl.__file__), "utils", "data")
    with open(os.path.join(_data_dir, "friedland_ch7_xyz_reported.json")) as f:
        xyz_reported_dev = cl.read_json(f.read())
    xyz_reported_dev.ldf_ = xyz_reported_dev.ldf_.round(3)

    rate_history = pd.DataFrame({
        "date": [
            "1/1/1999",
            "1/1/2000",
            "1/1/2001",
            "1/1/2002",
            "1/1/2003",
            "1/1/2004",
            "1/1/2005",
            "1/1/2006",
            "1/1/2007",
            "1/1/2008",
        ],
        "rate_change": [0.02, 0.02, 0.02, 0.02, 0.05, 0.075, 0.15, 0.10, -0.20, -0.20],
    })
    tort_history = pd.DataFrame({
        "date": ["1/1/2006", "1/1/2007"],
        "rate_change": [-0.1067, -0.25],
    })

    onlevel = cl.ParallelogramOLF(
        rate_history, change_col="rate_change", date_col="date", vertical_line=True
    )
    tort = cl.ParallelogramOLF(
        tort_history, change_col="rate_change", date_col="date", vertical_line=True
    )

    floored_patterns = dict(
        zip(
            [int(age) for age in xyz_reported_dev.cdf_.ddims],
            np.maximum(xyz_reported_dev.cdf_.values.flatten(), 1.0),
        )
    )

    sample_weight = onlevel.fit_transform(xyz_premium)
    cc = (
        cl
        .Pipeline([
            ("tort", tort),
            ("dev", cl.DevelopmentConstant(patterns=floored_patterns, style="cdf")),
            ("capecod", cl.CapeCod(trend=0.0342)),
        ])
        .fit(xyz_reported, sample_weight=sample_weight)
        .named_steps["capecod"]
    )

    # Exhibit II, Sheet 1 - all-years adjusted claim ratio is 0.708
    assert np.isclose(cc.apriori_.values[0, 0, 0, 0], 0.708, atol=2e-3)

    # Exhibit II, Sheet 1 - detrended (unadjusted) claim ratios, Column 15
    expected_detrended = [
        0.746,
        0.757,
        0.767,
        0.778,
        0.789,
        0.777,
        0.747,
        0.672,
        0.565,
        0.547,
        0.708,
    ]
    assert np.allclose(
        cc.detrended_apriori_.values.flatten(), expected_detrended, atol=2e-3
    )

    # Exhibit II, Sheet 2 - projected ultimate claims total 504,300
    assert np.isclose(cc.ultimate_.sum().sum(), 504300, rtol=5e-3)
