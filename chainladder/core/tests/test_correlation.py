import numpy as np
import pytest
import chainladder as cl
from rpy2.robjects.packages import importr
from rpy2.robjects import r

CL = importr("ChainLadder")


@pytest.fixture
def atol():
    return 1e-5


def dev_corr_r(data, ci):
    return r("out<-dfCorTest({},ci={})".format(data, ci))


def dev_corr_p(data, ci):
    return cl.load_sample(data).development_correlation(p_critical=ci)


def val_corr_r(data, ci):
    return r("out<-cyEffTest({},ci={})".format(data, ci))


def val_corr_p(data, ci):
    return cl.load_sample(data).valuation_correlation(p_critical=ci, total=True)


data = ["RAA", "GenIns", "MW2014"]
ci = [0.5, 0.75]


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("ci", ci)
def test_dev_corr(data, ci, atol):
    p = dev_corr_p(data, ci)
    r = dev_corr_r(data, ci).rx("T_stat")[0]
    p = p.t_expectation.values[0]
    assert np.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("ci", ci)
def test_dev_corr_var(data, ci, atol):
    p = dev_corr_p(data, ci)
    r = dev_corr_r(data, ci).rx("Var")[0]
    p = np.array([p.t_variance])
    assert np.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("ci", ci)
def test_dev_corr_range(data, ci, atol):
    p = dev_corr_p(data, 1 - ci)
    r = dev_corr_r(data, ci).rx("Range")[0]
    p = np.array(p.range)
    assert np.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("ci", ci)
def test_val_corr_z(data, ci, atol):
    p = val_corr_p(data, ci)
    r = val_corr_r(data, ci).rx("Z")[0]
    p = p.z.values[0]
    assert np.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("ci", ci)
def test_val_corr_e(data, ci, atol):
    p = val_corr_p(data, ci)
    r = val_corr_r(data, ci).rx("E")[0]
    p = p.z_expectation.values[0]
    assert np.allclose(r, p, atol=atol)


@pytest.mark.parametrize("data", data)
@pytest.mark.parametrize("ci", ci)
def test_val_corr_var(data, ci, atol):
    p = val_corr_p(data, ci)
    r = val_corr_r(data, ci).rx("Var")[0]
    p = p.z_variance.values[0]
    assert np.allclose(r, p, atol=atol)
