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
