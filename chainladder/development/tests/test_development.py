import numpy as np
import chainladder as cl
import pytest

class _FutureDevelopment(cl.TriangleWeight):
    '''
    An internal class to assist with the testing of the TriangleWeight utility class
    '''
    def __init__(self,dev):
        super().__init__(
            n_periods=dev.n_periods,
            drop=dev.drop,
            drop_high=dev.drop_high,
            drop_low=dev.drop_low,
            preserve=dev.preserve,
            drop_valuation=dev.drop_valuation,
            drop_above=dev.drop_above,
            drop_below=dev.drop_below
        )    
        self.average = dev.average
        self.dev = dev
    
    def fit(self, X, y: None = None, sample_weight: None = None):
        if hasattr(X,'age_to_age'):
            super().fit(X.incr_to_cum().age_to_age)
            xp = X.get_array_module()
            indices = X.values.shape[0]
            columns = X.values.shape[1]
            origins = X.age_to_age.values.shape[2]
            reg_x = X.incr_to_cum().values[...,:origins,:-1]
            reg_y = X.incr_to_cum().values[...,:origins,1:]
            dev_len = reg_x.shape[3]
            average_param = self._cascade_param(dev_len, self.average, "volume")
            average_param = np.tile(average_param,(indices,columns,1,1))
            params = cl.WeightedRegression(axis=2, thru_orig=True, xp=xp).fit(
                reg_x, reg_y, self.w_.values, average_param
            )
            self.ldf_ = self.dev._param_property(X, xp.swapaxes(params.slope_, 2, 3), 0)
        return self
    
def test_full_slice(genins):
    dev1 = cl.Development()
    dev2 = cl.Development(n_periods=1000)
    assert (
        dev1.fit_transform(genins).ldf_
        == dev2.fit_transform(genins).ldf_
    )
    assert (
        dev1.fit_transform(genins).ldf_
        == _FutureDevelopment(dev1).fit(genins).ldf_
    )
    assert (
        _FutureDevelopment(dev1).fit(genins).ldf_
        == _FutureDevelopment(dev2).fit(genins).ldf_
    )

def test_full_slice2(genins):
    dev1 = cl.Development()
    dev2 = cl.Development(n_periods=[1000] * (genins.shape[3] - 1))
    assert (
        dev1.fit_transform(genins).ldf_
        == dev2.fit_transform(genins).ldf_
    )
    assert (
        dev1.fit_transform(genins).ldf_
        == _FutureDevelopment(dev1).fit(genins).ldf_
    )
    assert (
        _FutureDevelopment(dev1).fit(genins).ldf_
        == _FutureDevelopment(dev2).fit(genins).ldf_
    )

def test_drop1(raa):
    dev1 = cl.Development(drop=("1982", 12))
    dev2 = cl.Development(drop_high=[True] + [False] * 8)
    assert (
        dev1.fit(raa).ldf_.values[0, 0, 0, 0]
        == dev2.fit(raa).ldf_.values[0, 0, 0, 0]
    )
    assert (
        dev1.fit_transform(raa).ldf_.values[0, 0, 0, 0]
        == _FutureDevelopment(dev1).fit(raa).ldf_.values[0, 0, 0, 0]
    )
    assert (
        _FutureDevelopment(dev1).fit(raa).ldf_.values[0, 0, 0, 0]
        == _FutureDevelopment(dev2).fit(raa).ldf_.values[0, 0, 0, 0]
    )

def test_drop2(raa):
    dev1 = cl.Development(drop_valuation="1981")
    dev2 = cl.Development(drop_low=[True] + [False] * 8)
    assert (
        dev1.fit(raa).ldf_.values[0, 0, 0, 0]
        == dev2.fit(raa).ldf_.values[0, 0, 0, 0]
    )
    assert (
        dev1.fit_transform(raa).ldf_.values[0, 0, 0, 0]
        == _FutureDevelopment(dev1).fit(raa).ldf_.values[0, 0, 0, 0]
    )
    assert (
        _FutureDevelopment(dev1).fit(raa).ldf_.values[0, 0, 0, 0]
        == _FutureDevelopment(dev2).fit(raa).ldf_.values[0, 0, 0, 0]
    )


def test_n_periods():
    d = cl.load_sample("usauto")["incurred"]
    xp = np if d.array_backend == "sparse" else d.get_array_module()
    dev = cl.Development(n_periods=3, average="volume")
    assert xp.all(
        xp.around(
            xp.unique(
                dev.fit(d).ldf_.values,
                axis=-2,
            ),
            3,
        ).flatten()
        == xp.array([1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.0])
    )
    assert (
        dev.fit_transform(d).ldf_
        == _FutureDevelopment(dev).fit(d).ldf_
    )

def test_drophighlow(raa):
    dev = cl.Development(drop_high=0)
    lhs = np.round(dev.fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [8.9202, 2.974, 1.8318, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert (
        dev.fit_transform(raa).ldf_
        == _FutureDevelopment(dev).fit(raa).ldf_
    )

    dev = cl.Development(drop_high=[True, False, True, False])
    lhs = np.round(dev.fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [8.0595, 2.8613, 1.7624, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert (
        dev.fit_transform(raa).ldf_
        == _FutureDevelopment(dev).fit(raa).ldf_
    )

    dev = cl.Development(
            drop_high=[2, 3, 3, 3], drop_low=[0, 1, 0], preserve=2
        )
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        tr = dev.fit_transform(raa)
        tw = _FutureDevelopment(dev).fit(raa)
    lhs = np.round(tr.cdf_.values, 4).flatten()
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
    assert tr.ldf_ == tw.ldf_

    dev = cl.Development(drop_high=1)
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        tr = dev.fit_transform(raa)
        tw = _FutureDevelopment(dev).fit(raa)
    lhs = np.round(tr.cdf_.values, 4).flatten()
    rhs = np.array(
        [7.2190, 2.5629, 1.6592, 1.3570, 1.1734, 1.0669, 1.0419, 1.0121, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert tr.ldf_ == tw.ldf_

    dev = cl.Development(drop_high=1, drop_low=1)
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        tr = dev.fit_transform(raa)
        tw = _FutureDevelopment(dev).fit(raa)
    lhs = np.round(tr.cdf_.values, 4).flatten()
    rhs = np.array(
        [9.0982, 2.8731, 1.8320, 1.4713, 1.2522, 1.0963, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert tr.ldf_ == tw.ldf_

    dev = cl.Development(drop_high=[2, 1, 1], drop_low=1)
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        tr = dev.fit_transform(raa)
        tw = _FutureDevelopment(dev).fit(raa)
    lhs = np.round(tr.cdf_.values, 4).flatten()
    rhs = np.array(
        [8.4905, 3.0589, 1.9504, 1.5664, 1.3142, 1.1403, 1.0822, 1.0426, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert tr.ldf_ == tw.ldf_

    dev = cl.Development(drop_high=1, drop_low=1, n_periods=5)
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        tr = dev.fit_transform(raa)
        tw = _FutureDevelopment(dev).fit(raa)
    lhs = np.round(tr.cdf_.values, 4).flatten()
    rhs = np.array(
        [16.3338, 3.2092, 1.8124, 1.4793, 1.2522, 1.0963, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert tr.ldf_ == tw.ldf_


def test_drophighlow_inequal(prism,atol):
    tri = prism["Paid"].sum().grain("OYDQ")
    dev0 = cl.Development()
    dev1 = cl.Development(drop_high=True)
    dev2 = cl.Development(drop_low=True)
    no_drop = dev0.fit_transform(tri).cdf_.to_frame().values
    drop_high = dev1.fit_transform(tri).cdf_.to_frame().values
    drop_low = dev2.fit_transform(tri).cdf_.to_frame().values
    assert (drop_low >= no_drop).all()
    assert (no_drop >= drop_high).all()
    assert (_FutureDevelopment(dev2).fit(tri).ldf_.values >= _FutureDevelopment(dev0).fit(tri).ldf_.values).all()
    assert (_FutureDevelopment(dev0).fit(tri).ldf_.values >= _FutureDevelopment(dev1).fit(tri).ldf_.values).all()


def test_dropabovebelow(raa):
    dev = cl.Development(drop_above=40.0)
    lhs = np.round(dev.fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [8.3771, 2.9740, 1.8318, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert dev.fit(raa).ldf_ == _FutureDevelopment(dev).fit(raa).ldf_

    dev = cl.Development(drop_above=1.2)
    lhs = np.round(dev.fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [7.6859, 2.5625, 1.5784, 1.4072, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert dev.fit(raa).ldf_ == _FutureDevelopment(dev).fit(raa).ldf_

    dev = cl.Development(drop_above=1.2, drop_below=1.05)
    lhs = np.round(dev.fit(raa).cdf_.values, 4).flatten()
    rhs = np.array(
        [8.4983, 2.8334, 1.7452, 1.5560, 1.3602, 1.1802, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert dev.fit(raa).ldf_ == _FutureDevelopment(dev).fit(raa).ldf_

    dev = cl.Development(drop_above=[40.0], drop_below=[0.0, 0.0, 1.05, 1.7])
    lhs = np.round(dev.fit(raa).cdf_.values,4,).flatten()
    rhs = np.array(
        [8.3771, 2.9740, 1.8318, 1.4414, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert dev.fit(raa).ldf_ == _FutureDevelopment(dev).fit(raa).ldf_

    dev = cl.Development(drop_above=[40.0], drop_below=[0.0, 0.0, 1.05, 1.2])
    lhs = np.round(dev.fit(raa).cdf_.values,4,).flatten()
    rhs = np.array(
        [8.9773, 3.1871, 1.9631, 1.5447, 1.2302, 1.1049, 1.0604, 1.0263, 1.0092]
    )
    assert np.all(lhs == rhs)
    assert dev.fit(raa).ldf_ == _FutureDevelopment(dev).fit(raa).ldf_


def test_drop_valuation_1(raa):
    dev1 = cl.Development(drop_valuation="1981-12-31")
    dev2 = cl.Development(drop_valuation="1982-12-31")
    dev3 = cl.Development(drop_valuation="1983-12-31")
    assert (
        dev1.fit_transform(raa).cdf_
        != dev2.fit_transform(raa).cdf_
    )
    assert (
        dev2.fit_transform(raa).cdf_
        != dev3.fit_transform(raa).cdf_
    )
    assert (
        dev1.fit_transform(raa).cdf_
        != dev3.fit_transform(raa).cdf_
    )
    assert (
        _FutureDevelopment(dev1).fit(raa).ldf_
        != _FutureDevelopment(dev2).fit(raa).ldf_
    )
    assert (
        _FutureDevelopment(dev2).fit(raa).ldf_
        != _FutureDevelopment(dev3).fit(raa).ldf_
    )
    assert (
        _FutureDevelopment(dev1).fit(raa).ldf_
        != _FutureDevelopment(dev3).fit(raa).ldf_
    )


def test_drop_valuation_2(qtr):
    dev1 = cl.Development()
    dev2 = cl.Development(drop_valuation="1995-03-31")
    dev3 = cl.Development(drop_valuation="1995-06-30")
    dev4 = cl.Development(drop_valuation="1995-09-30")
    assert (
        dev1.fit_transform(qtr["incurred"]).cdf_
        != dev2.fit_transform(qtr["incurred"]).cdf_
    )
    assert (
        dev1.fit_transform(qtr["incurred"]).cdf_
        != dev3.fit_transform(qtr["incurred"]).cdf_
    )
    assert (
        dev2.fit_transform(qtr["incurred"]).cdf_
        != dev3.fit_transform(qtr["incurred"]).cdf_
    )
    assert (
        dev3.fit_transform(qtr["incurred"]).cdf_
        != dev4.fit_transform(qtr["incurred"]).cdf_
    )
    assert (
        _FutureDevelopment(dev1).fit(qtr["incurred"]).ldf_
        != _FutureDevelopment(dev2).fit(qtr["incurred"]).ldf_
    )
    assert (
        _FutureDevelopment(dev1).fit(qtr["incurred"]).ldf_
        != _FutureDevelopment(dev3).fit(qtr["incurred"]).ldf_
    )
    assert (
        _FutureDevelopment(dev2).fit(qtr["incurred"]).ldf_
        != _FutureDevelopment(dev3).fit(qtr["incurred"]).ldf_
    )
    assert (
        _FutureDevelopment(dev3).fit(qtr["incurred"]).ldf_
        != _FutureDevelopment(dev4).fit(qtr["incurred"]).ldf_
    )


def test_assymetric_development(qtr,atol):
    quarterly = qtr["paid"]
    xp = np if quarterly.array_backend == "sparse" else quarterly.get_array_module()
    dev1 = cl.Development(n_periods=1, average="simple")
    dev2 = cl.Development(n_periods=1, average="regression")
    assert xp.allclose(
        dev1.fit(quarterly).ldf_.values, 
        dev2.fit(quarterly).ldf_.values, 
        atol=atol
    )
    assert xp.allclose(
        _FutureDevelopment(dev1).fit(quarterly).ldf_.values,
        _FutureDevelopment(dev2).fit(quarterly).ldf_.values,
        atol=atol
    )

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
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        dev = cl.Development(drop_high=1, drop_low=1, preserve=3).fit(clrd)
    compare_new_drop(dev, clrd)


def test_new_drop_5a(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_hi/low without preserve
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        lhs = (
            cl.Development(drop_high=1, drop_low=1, preserve=3)
            ._set_weight_func(clrd.age_to_age, clrd.age_to_age)
            .values
        )
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        rhs = (
            cl.Development(
                drop_high=True,
                drop_low=[True, True, True, True, True, True, True, True, True],
                preserve=3,
            )
            ._set_weight_func(clrd.age_to_age)
            .values
        )
    assert np.array_equal(lhs, rhs, True)


def test_new_drop_6(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_above/below without preserve
    compare_new_drop(cl.Development(drop_above=1.01, drop_below=0.95).fit(clrd), clrd)


def test_new_drop_7(clrd):
    clrd = clrd.groupby("LOB")[["IncurLoss", "CumPaidLoss"]].sum()
    # drop_above/below with preserve
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        dev = cl.Development(drop_above=1.01, drop_below=0.95, preserve=3).fit(clrd)
    compare_new_drop(dev, clrd)


def test_new_drop_8(prism):
    tri = prism["Paid"].sum().grain("OYDQ")
    try:
        cl.Development(drop_high=False).fit_transform(tri)
    except:
        assert False


def test_new_drop_9(prism):
    tri = prism["Paid"].sum().grain("OYDQ")

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


def test_geometric_avg():
    tri = cl.load_sample("friedland_us_industry_auto")["Reported Claims"]
    df = tri.link_ratio.to_frame()

    lhs = np.round(
        cl.Development(n_periods=4, average="geometric")
        .fit_transform(tri)
        .ldf_.to_frame()
        .values.flatten(),
        6,
    )

    def geo_lastn(s, n):
        vals = s.dropna().tail(n)
        return vals.prod() ** (1 / len(vals)) if len(vals) > 0 else np.nan

    geo_means = df.apply(lambda s: geo_lastn(s, 4))
    rhs = np.round(geo_means.values.flatten(), 6)

    assert np.all(lhs == rhs)


def test_simple_avg():
    tri = cl.load_sample("friedland_us_industry_auto")["Reported Claims"]
    df = tri.link_ratio.to_frame()

    lhs = np.round(
        cl.Development(n_periods=4, average="simple")
        .fit_transform(tri)
        .ldf_.to_frame()
        .values.flatten(),
        6,
    )

    def sim_lastn(s, n):
        vals = s.dropna().tail(n)
        return vals.mean() if len(vals) > 0 else np.nan

    avg_means = df.apply(lambda s: sim_lastn(s, 4))
    rhs = np.round(avg_means.values.flatten(), 6)

    assert np.all(lhs == rhs)


def test_simple_geometric_avg():
    tri = cl.load_sample("friedland_us_industry_auto")["Reported Claims"]
    df = tri.link_ratio.to_frame()

    lhs = np.round(
        cl.Development(
            n_periods=4,
            average=[
                "geometric",
                "simple",
                "geometric",
                "simple",
                "geometric",
                "simple",
                "geometric",
                "simple",
                "geometric",
            ],
        )
        .fit_transform(tri)
        .ldf_.to_frame()
        .values.flatten(),
        6,
    )

    def sim_lastn(s, n):
        vals = s.dropna().tail(n)
        return vals.mean() if len(vals) > 0 else np.nan

    def geo_lastn(s, n):
        vals = s.dropna().tail(n)
        return vals.prod() ** (1 / len(vals)) if len(vals) > 0 else np.nan

    sim_avg = df.apply(lambda s: s.dropna().tail(4).mean())
    geo_avg = df.apply(
        lambda s: s.dropna().tail(4).prod() ** (1 / len(s.dropna().tail(4)))
    )

    methods = np.array(
        [
            "geometric",
            "simple",
            "geometric",
            "simple",
            "geometric",
            "simple",
            "geometric",
            "simple",
            "geometric",
        ]
    )

    rhs = np.round(np.where(methods == "geometric", geo_avg.values, sim_avg.values), 6)

    assert np.all(lhs == rhs)


def test_simple_geometric_avg2():
    tri = cl.load_sample("friedland_us_industry_auto")["Reported Claims"]
    df = tri.link_ratio.to_frame()

    lhs = np.round(
        cl.Development(
            n_periods=4,
            average=[
                "simple",
                "geometric",
                "simple",
                "geometric",
                "simple",
                "geometric",
                "simple",
                "geometric",
                "simple",
            ],
        )
        .fit_transform(tri)
        .ldf_.to_frame()
        .values.flatten(),
        6,
    )

    def sim_lastn(s, n):
        vals = s.dropna().tail(n)
        return vals.mean() if len(vals) > 0 else np.nan

    def geo_lastn(s, n):
        vals = s.dropna().tail(n)
        return vals.prod() ** (1 / len(vals)) if len(vals) > 0 else np.nan

    sim_avg = df.apply(lambda s: s.dropna().tail(4).mean())
    geo_avg = df.apply(
        lambda s: s.dropna().tail(4).prod() ** (1 / len(s.dropna().tail(4)))
    )

    methods = np.array(
        [
            "simple",
            "geometric",
            "simple",
            "geometric",
            "simple",
            "geometric",
            "simple",
            "geometric",
            "simple",
        ]
    )

    rhs = np.round(np.where(methods == "geometric", geo_avg.values, sim_avg.values), 6)

    assert np.all(lhs == rhs)


def test_sigma():
    tri = cl.load_sample("friedland_us_industry_auto")["Reported Claims"]
    sigma = np.round(
        cl.Development(
            n_periods=4,
            average="simple",
        )
        .fit_transform(tri)
        .sigma_.to_frame()
        .values.flatten(),
        6,
    )
    sigma_expected = [
        0.006371,
        0.001693,
        0.001274,
        0.001823,
        0.000612,
        0.000349,
        0.000371,
        0.000212,
        0.000128,
    ]

    assert np.all(sigma == sigma_expected)


def test_stderror():
    tri = cl.load_sample("friedland_us_industry_auto")["Reported Claims"]
    std_error = np.round(
        cl.Development(
            n_periods=4,
            average="simple",
        )
        .fit_transform(tri)
        .std_err_.to_frame()
        .values.flatten(),
        6,
    )
    std_error_expected = [
        0.003186,
        0.000847,
        0.000637,
        0.000912,
        0.000306,
        0.000175,
        0.000214,
        0.00015,
        0.000128,
    ]
    assert np.all(std_error == std_error_expected)


def test_std_residuals():
    tri = cl.load_sample("friedland_us_industry_auto")["Reported Claims"]
    std_residuals = np.round(
        cl.Development(
            n_periods=4,
            average="simple",
        )
        .fit_transform(tri)
        .std_residuals_.to_frame()
        .values,
        6,
    )
    std_residuals_expected = [
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.342157, -1.144874, 0.707107, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.847023, 0.576855, 0.702623, -0.707107, np.nan],
        [0.0, 0.0, 0.0, -0.74519, 1.213508, 0.917912, 0.442251, np.nan, np.nan],
        [0.0, 0.0, -0.251713, 1.337416, 0.426182, -0.15261, np.nan, np.nan, np.nan],
        [0.0, 1.426642, 1.023636, 0.194113, -0.792667, np.nan, np.nan, np.nan, np.nan],
        [
            -0.19898,
            -0.056297,
            0.505912,
            -0.78634,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            -0.727376,
            -0.791472,
            -1.277835,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [-0.537388, -0.578874, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [1.463744, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]

    assert np.array_equal(std_residuals, std_residuals_expected, equal_nan=True)


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
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
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
    with pytest.warns(UserWarning, match="exclusions have been ignored"):
        dev2 = pipe.fit(X=clrd)
    assert np.array_equal(
        dev1.w_v2_.values, dev2.named_steps.drop_hilo.w_v2_.values, True
    )
