import numpy as np
import pytest
import chainladder as cl

try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r

    CL = importr("ChainLadder")
except:
    pass


def mack_r(data, alpha, est_sigma):
    return r(
        'mack<-MackChainLadder({},alpha={}, est.sigma="{}")'.format(
            data, alpha, est_sigma
        )
    )


def mack_p(data, average, est_sigma):
    return cl.Development(average=average, sigma_interpolation=est_sigma).fit_transform(
        cl.load_sample(data)
    )


data = ["RAA", "GenIns", "MW2014"]
averages = [("simple", 0), ("volume", 1), ("regression", 2)]
est_sigma = [("mack", "Mack"), ("log-linear", "log-linear")]


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


@pytest.mark.r
@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
def test_mack_ldf(data, averages, est_sigma, atol):
    p = mack_p(data, averages[0], est_sigma[0]).ldf_
    xp = p.get_array_module()
    r = xp.array(mack_r(data, averages[1], est_sigma[1]).rx("f"))[:, :-1]
    assert xp.allclose(r, p.values[0, 0, :, :], atol=atol)


@pytest.mark.r
@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
def test_mack_sigma(data, averages, est_sigma, atol):
    p = mack_p(data, averages[0], est_sigma[0]).sigma_.set_backend(
        "numpy", inplace=True
    )
    xp = p.get_array_module()
    r = xp.array(mack_r(data, averages[1], est_sigma[1]).rx("sigma"))
    assert xp.allclose(r, p.values[0, 0, :, :], atol=atol)


@pytest.mark.r
@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
def test_mack_std_err(data, averages, est_sigma, atol):
    p = mack_p(data, averages[0], est_sigma[0]).std_err_
    xp = p.get_array_module()
    r = xp.array(mack_r(data, averages[1], est_sigma[1]).rx("f.se"))
    assert xp.allclose(r, p.values[0, 0, :, :], atol=atol)


def test_assymetric_development(atol):
    quarterly = cl.load_sample("quarterly")["paid"]
    xp = np if quarterly.array_backend == "sparse" else quarterly.get_array_module()
    dev = cl.Development(n_periods=1, average="simple").fit(quarterly)
    dev2 = cl.Development(n_periods=1, average="regression").fit(quarterly)
    assert xp.allclose(dev.ldf_.values, dev2.ldf_.values, atol=atol)
