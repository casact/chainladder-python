import numpy as np
import pytest
import chainladder as cl
from rpy2.robjects.packages import importr
from rpy2.robjects import r

CL = importr("ChainLadder")


@pytest.fixture
def atol():
    return 1e-5


def mack_r(data, alpha, est_sigma, tail):
    if tail:
        return r(
            'mack<-MackChainLadder({},alpha={}, est.sigma="{}", tail=TRUE)'.format(
                data, alpha, est_sigma
            )
        )
    else:
        return r(
            'mack<-MackChainLadder({},alpha={}, est.sigma="{}")'.format(
                data, alpha, est_sigma
            )
        )


def mack_p(data, average, est_sigma, tail):
    if tail:
        return cl.MackChainladder().fit(
            cl.TailCurve(curve="exponential").fit_transform(
                cl.Development(
                    average=average, sigma_interpolation=est_sigma
                ).fit_transform(cl.load_sample(data))
            )
        )
    else:
        return cl.MackChainladder().fit(
            cl.Development(
                average=average, sigma_interpolation=est_sigma
            ).fit_transform(cl.load_sample(data))
        )


data = ["ABC", "MW2008"]
tail = [True, False]
averages = [("simple", 0), ("volume", 1), ("regression", 2)]
est_sigma = [("log-linear", "log-linear"), ("mack", "Mack")]


def test_mack_to_triangle():
    assert (
        cl.MackChainladder()
        .fit(
            cl.TailConstant().fit_transform(
                cl.Development().fit_transform(cl.load_sample("ABC"))
            )
        )
        .summary_
        == cl.MackChainladder()
        .fit(cl.Development().fit_transform(cl.load_sample("ABC")))
        .summary_
    )


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
@pytest.mark.parametrize("tail", tail)
def test_mack_full_std_err(data, averages, est_sigma, tail, atol):
    df = mack_r(data, averages[1], est_sigma[1], tail).rx("F.se")
    p = mack_p(data, averages[0], est_sigma[0], tail).full_std_err_
    xp = p.get_array_module()
    p = p.values[0, 0, :, :][:, :-1] if not tail else p.values[0, 0, :, :]
    r = xp.array(df[0])
    assert xp.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
@pytest.mark.parametrize("tail", tail)
def test_mack_process_risk(data, averages, est_sigma, tail, atol):
    df = mack_r(data, averages[1], est_sigma[1], tail).rx("Mack.ProcessRisk")
    p = mack_p(data, averages[0], est_sigma[0], tail).process_risk_
    xp = p.get_array_module()
    p = p.values[0, 0, :, :][:, :-1] if not tail else p.values[0, 0, :, :]
    r = xp.array(df[0])
    assert xp.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
@pytest.mark.parametrize("tail", tail)
def test_mack_parameter_risk(data, averages, est_sigma, tail, atol):
    df = mack_r(data, averages[1], est_sigma[1], tail).rx("Mack.ParameterRisk")
    p = mack_p(data, averages[0], est_sigma[0], tail).parameter_risk_
    xp = p.get_array_module()
    p = p.values[0, 0, :, :][:, :-1] if not tail else p.values[0, 0, :, :]
    r = xp.array(df[0])
    assert xp.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
@pytest.mark.parametrize("tail", tail)
def test_mack_total_process_risk(data, averages, est_sigma, tail, atol):
    df = mack_r(data, averages[1], est_sigma[1], tail).rx("Total.ProcessRisk")
    p = mack_p(data, averages[0], est_sigma[0], tail).total_process_risk_
    xp = p.get_array_module()
    p = p.values[0, 0, :, :][:, :-1] if not tail else p.values[0, 0, :, :]
    r = xp.array(df[0])[None, ...]
    assert xp.allclose(r, xp.nan_to_num(p), atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
@pytest.mark.parametrize("tail", tail)
def test_mack_total_parameter_risk(data, averages, est_sigma, tail, atol):
    df = mack_r(data, averages[1], est_sigma[1], tail).rx("Total.ParameterRisk")
    p = mack_p(data, averages[0], est_sigma[0], tail).total_parameter_risk_
    xp = p.get_array_module()
    p = p.values[0, 0, :, :][:, :-1] if not tail else p.values[0, 0, :, :]
    r = xp.array(df[0])[None]
    assert xp.allclose(r, xp.nan_to_num(p), atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("averages", averages)
@pytest.mark.parametrize("est_sigma", est_sigma)
@pytest.mark.parametrize("tail", tail)
def test_mack_mack_std_err_(data, averages, est_sigma, tail, atol):
    df = mack_r(data, averages[1], est_sigma[1], tail).rx("Mack.S.E")
    p = mack_p(data, averages[0], est_sigma[0], tail).mack_std_err_
    xp = p.get_array_module()
    p = p.values[0, 0, :, :][:, :-1] if not tail else p.values[0, 0, :, :]
    r = xp.array(df[0])
    assert xp.allclose(r, xp.nan_to_num(p), atol=atol)


def test_mack_asymmetric():
    r("Paid <- matrix(NA, 45, 45)")
    r("Paid[seq(1,45,4),] <- qpaid")
    out = r("M <- MackChainLadder(Paid)")
    tri = cl.load_sample("quarterly")["paid"]
    xp = tri.get_array_module()
    assert round(float(xp.array(out.rx("Mack.S.E")[0])[-1, -1]), 2) == round(
        float(cl.MackChainladder().fit(tri).summary_.to_frame().iloc[-1, -1]), 2
    )
