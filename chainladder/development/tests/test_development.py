import numpy as np
import chainladder as cl
import pytest


def test_full_slice():
    assert (
        cl.Development().fit_transform(cl.load_sample("GenIns")).ldf_
        == cl.Development(n_periods=1000).fit_transform(cl.load_sample("GenIns")).ldf_
    )


def test_full_slice2():
    assert (
        cl.Development().fit_transform(cl.load_sample("GenIns")).ldf_
        == cl.Development(n_periods=[1000] * (cl.load_sample("GenIns").shape[3] - 1))
        .fit_transform(cl.load_sample("GenIns"))
        .ldf_
    )


def test_drop1(raa):
    assert (
        cl.Development(drop=("1982", 12)).fit(raa).ldf_.values[0, 0, 0, 0]
        == cl.Development(drop_high=[True] + [False] * 8)
        .fit(raa)
        .ldf_.values[0, 0, 0, 0]
    )


def test_drop2(raa):
    assert (
        cl.Development(drop_valuation="1981").fit(raa).ldf_.values[0, 0, 0, 0]
        == cl.Development(drop_low=[True] + [False] * 8)
        .fit(raa)
        .ldf_.values[0, 0, 0, 0]
    )


def test_n_periods():
    d = cl.load_sample("usauto")["incurred"]
    xp = np if d.array_backend == "sparse" else d.get_array_module()
    return xp.all(
        xp.around(
            xp.unique(
                cl.Development(n_periods=3, average="volume").fit(d).ldf_.values,
                axis=-2,
            ),
            3,
        ).flatten()
        == xp.array([1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.0])
    )


def test_drophighlow():
    raa = cl.load_sample("raa")

    lhs = np.round(cl.Development(drop_high=0).fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [8.9202, 2.974, 1.8318, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_high=[True, False, True, False]).fit(raa).cdf_.values, 4
    ).flatten()
    rhs = np.array(
        [8.0595, 2.8613, 1.7624, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_high=[2, 3, 3, 3], drop_low=[0, 1, 0], preserve=2)
        .fit(raa)
        .cdf_.values,
        4,
    ).flatten()
    rhs = np.array(
        [
            5.7403,
            2.2941,
            1.5617,
            1.3924,
            1.2302,
            1.1049,
            1.0604,
            1.0263,
            1.0092,
        ]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(cl.Development(drop_high=1).fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [7.2190, 2.5629, 1.6592, 1.3570, 1.1734, 1.0669, 1.0419, 1.0121, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_high=1, drop_low=1).fit(raa).cdf_.values, 4
    ).flatten()
    rhs = np.array(
        [9.0982, 2.8731, 1.8320, 1.4713, 1.2522, 1.0963, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_high=[2, 1, 1], drop_low=1).fit(raa).cdf_.values, 4
    ).flatten()
    rhs = np.array(
        [8.4905, 3.0589, 1.9504, 1.5664, 1.3142, 1.1403, 1.0822, 1.0426, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_high=1, drop_low=1, n_periods=5).fit(raa).cdf_.values, 4
    ).flatten()
    rhs = np.array(
        [16.3338, 3.2092, 1.8124, 1.4793, 1.2522, 1.0963, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    tri = cl.load_sample("prism")["Paid"].sum().grain("OYDQ")
    no_drop = cl.Development().fit_transform(tri).cdf_.to_frame().values
    drop_high = cl.Development(drop_high=True).fit_transform(tri).cdf_.to_frame().values
    drop_low = cl.Development(drop_low=True).fit_transform(tri).cdf_.to_frame().values
    assert (drop_low >= no_drop).all()
    assert (no_drop >= drop_high).all()


def test_dropabovebelow():
    raa = cl.load_sample("raa")

    lhs = np.round(cl.Development(drop_above=40.0).fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [8.3771, 2.9740, 1.8318, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(cl.Development(drop_above=1.2).fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [7.6859, 2.5625, 1.5784, 1.4072, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_above=1.2, drop_below=1.05).fit(raa).cdf_.values, 4
    ).flatten()
    rhs = np.array(
        [8.4983, 2.8334, 1.7452, 1.5560, 1.3602, 1.1802, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_above=[40.0], drop_below=[0.0, 0.0, 1.05, 1.7])
        .fit(raa)
        .cdf_.values,
        4,
    ).flatten()
    rhs = np.array(
        [8.3771, 2.9740, 1.8318, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)

    lhs = np.round(
        cl.Development(drop_above=[40.0], drop_below=[0.0, 0.0, 1.05, 1.2])
        .fit(raa)
        .cdf_.values,
        4,
    ).flatten()
    rhs = np.array(
        [8.9773, 3.1871, 1.9631, 1.5447, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)


def test_drop_valuation():
    raa = cl.load_sample("raa")
    assert (
        cl.Development(drop_valuation="1981-12-31").fit_transform(raa).cdf_
        != cl.Development(drop_valuation="1982-12-31").fit_transform(raa).cdf_
    )
    assert (
        cl.Development(drop_valuation="1982-12-31").fit_transform(raa).cdf_
        != cl.Development(drop_valuation="1983-12-31").fit_transform(raa).cdf_
    )
    assert (
        cl.Development(drop_valuation="1981-12-31").fit_transform(raa).cdf_
        != cl.Development(drop_valuation="1983-12-31").fit_transform(raa).cdf_
    )

    quarterly = cl.load_sample("quarterly")
    assert (
        cl.Development().fit_transform(quarterly["incurred"]).cdf_
        != cl.Development(drop_valuation="1995-03-31")
        .fit_transform(quarterly["incurred"])
        .cdf_
    )
    assert (
        cl.Development().fit_transform(quarterly["incurred"]).cdf_
        != cl.Development(drop_valuation="1995-06-30")
        .fit_transform(quarterly["incurred"])
        .cdf_
    )
    assert (
        cl.Development(drop_valuation="1995-03-31")
        .fit_transform(quarterly["incurred"])
        .cdf_
        != cl.Development(drop_valuation="1995-06-30")
        .fit_transform(quarterly["incurred"])
        .cdf_
    )
    assert (
        cl.Development(drop_valuation="1995-06-30")
        .fit_transform(quarterly["incurred"])
        .cdf_
        != cl.Development(drop_valuation="1995-09-30")
        .fit_transform(quarterly["incurred"])
        .cdf_
    )


def test_assymetric_development(atol):
    quarterly = cl.load_sample("quarterly")["paid"]
    xp = np if quarterly.array_backend == "sparse" else quarterly.get_array_module()
    dev = cl.Development(n_periods=1, average="simple").fit(quarterly)
    dev2 = cl.Development(n_periods=1, average="regression").fit(quarterly)
    assert xp.allclose(dev.ldf_.values, dev2.ldf_.values, atol=atol)


def test_hilo_multiple_indices(clrd):
    tri = clrd.groupby("LOB")["CumPaidLoss"].sum()
    assert (
        cl.Development(n_periods=5).fit(tri).ldf_.loc["wkcomp"]
        == cl.Development(n_periods=5).fit(tri.loc["wkcomp"]).ldf_
    )
    assert (
        cl.Development(drop_low=2).fit(tri).ldf_.loc["wkcomp"]
        == cl.Development(drop_low=2).fit(tri.loc["wkcomp"]).ldf_
    )


def test_new_drop_1(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # n_periods
    compare_new_drop(cl.Development(n_periods=4).fit(clrd), clrd)


def test_new_drop_2(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # single drop and drop_valuation
    compare_new_drop(
        cl.Development(drop=("1992", 12), drop_valuation=1993).fit(clrd), clrd
    )


def test_new_drop_3(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # multiple drop and drop_valuation
    compare_new_drop(
        cl.Development(
            drop=[("1992", 12), ("1996", 24)], drop_valuation=[1993, 1995]
        ).fit(clrd),
        clrd,
    )


def test_new_drop_4(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_hi/low without preserve
    compare_new_drop(cl.Development(drop_high=1, drop_low=1).fit(clrd), clrd)


def test_new_drop_5(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_hi/low without preserve
    compare_new_drop(
        cl.Development(drop_high=1, drop_low=1, preserve=3).fit(clrd), clrd
    )


def test_new_drop_5a(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_hi/low without preserve
    assert np.array_equal(
        cl.Development(drop_high=1, drop_low=1, preserve=3)
        ._set_weight_func(clrd.age_to_age, clrd.age_to_age)
        .values,
        cl.Development(
            drop_high=True,
            drop_low=[True, True, True, True, True, True, True, True, True],
            preserve=3,
        )
        ._set_weight_func(clrd.age_to_age)
        .values,
        True,
    )


def test_new_drop_6(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_above/below without preserve
    compare_new_drop(cl.Development(drop_above=1.01, drop_below=0.95).fit(clrd), clrd)


def test_new_drop_7(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_above/below with preserve
    compare_new_drop(
        cl.Development(drop_above=1.01, drop_below=0.95, preserve=3).fit(clrd), clrd
    )


def test_new_drop_8():
    tri = cl.load_sample("prism")["Paid"].sum().grain("OYDQ")

    try:
        cl.Development(drop_high=False).fit_transform(tri)
    except:
        assert False


def test_new_drop_9():
    tri = cl.load_sample("prism")["Paid"].sum().grain("OYDQ")

    lhs = cl.Development(drop_high=True).fit(tri).cdf_.to_frame().fillna(0).values
    rhs = cl.Development(drop_high=1).fit(tri).cdf_.to_frame().fillna(0).values
    assert (lhs == rhs).all()


@pytest.mark.xfail
def test_new_drop_10():
    data = {
        "valuation": [
            1981,
            1982,
            1983,
            1984,
            1985,
            1982,
            1983,
            1984,
            1985,
        ],
        "origin": [
            1981,
            1982,
            1983,
            1984,
            1985,
            1981,
            1982,
            1983,
            1984,
        ],
        "values": [
            100,
            200,
            300,
            400,
            500,
            200,
            200,
            300,
            800,
        ],
    }

    tri = cl.Triangle(
        pd.DataFrame(data),
        origin="origin",
        development="valuation",
        columns=["values"],
        cumulative=True,
    )

    assert np.round(
        cl.Development(drop_high=1).fit(tri).cdf_.to_frame().values.flatten()[0], 4
    ) == (200 + 300 + 800) / (200 + 300 + 400)

    assert (
        np.round(
            cl.Development(drop_high=2).fit(tri).cdf_.to_frame().values.flatten()[0], 4
        )
        == 1.0000
    )


def compare_new_drop(dev, tri):
    assert np.array_equal(
        dev._set_weight_func(tri.age_to_age, tri.age_to_age).values,
        dev.transform(tri).age_to_age.values * 0 + 1,
        True,
    )


def test_4d_drop(clrd):
    clrd = clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss"]]
    assert (
        cl.Development(n_periods=4).fit_transform(clrd.iloc[0, 0]).link_ratio
        == cl.Development(n_periods=4).fit_transform(clrd).link_ratio.iloc[0, 0]
    )


def test_pipeline(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    dev1 = cl.Development(
        n_periods=7,
        drop_valuation=1995,
        drop=("1992", 12),
        drop_above=1.05,
        drop_below=0.95,
        drop_high=1,
        drop_low=1,
    ).fit(clrd)
    pipe = cl.Pipeline(
        steps=[
            ("n_periods", cl.Development(n_periods=7)),
            ("drop_valuation", cl.Development(drop_valuation=1995)),
            ("drop", cl.Development(drop=("1992", 12))),
            ("drop_abovebelow", cl.Development(drop_above=1.05, drop_below=0.95)),
            ("drop_hilo", cl.Development(drop_high=1, drop_low=1)),
        ]
    )
    dev2 = pipe.fit(X=clrd)
    assert np.array_equal(
        dev1.w_v2_.values, dev2.named_steps.drop_hilo.w_v2_.values, True
    )
