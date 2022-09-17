### Building out a dev environment with a working copy
### of R ChainLadder is difficult.  These tests are 
### Currently inactive, but available should the compatibility
### of the installs improve at a later date.

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
