import numpy as np
import pytest
from numpy.testing import assert_allclose
import chainladder as cl
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r

pandas2ri.activate()
CL = importr('ChainLadder')


@pytest.fixture
def atol():
    return 1e-5


def mack_r(data, alpha, est_sigma):
    return r(f'mack<-MackChainLadder({data},alpha={alpha}, est.sigma="{est_sigma}")')


def mack_p(data, average, est_sigma):
    return cl.Development(average=average, sigma_interpolation=est_sigma).fit(cl.load_dataset(data))


data = ['RAA', 'ABC', 'GenIns', 'M3IR5', 'MW2008', 'MW2014']
averages = [('simple', 0), ('volume', 1), ('regression', 2)]
est_sigma = [('mack', 'Mack'), ('log-linear', 'log-linear')]


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_ldf(data, averages, est_sigma, atol):
    r = np.array(mack_r(data, averages[1], est_sigma[1]).rx('f'))[:, :-1]
    p = mack_p(data, averages[0], est_sigma[0]).ldf_.triangle[0, 0, :, :]
    assert_allclose(r, p, atol=atol)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_sigma(data, averages, est_sigma, atol):
    r = np.array(mack_r(data, averages[1], est_sigma[1]).rx('sigma'))
    p = mack_p(data, averages[0], est_sigma[0]).sigma_.triangle[0, 0, :, :]
    assert_allclose(r, p, atol=atol)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_std_err(data, averages, est_sigma, atol):
    r = np.array(mack_r(data, averages[1], est_sigma[1]).rx('f.se'))
    p = mack_p(data, averages[0], est_sigma[0]).std_err_.triangle[0, 0, :, :]
    assert_allclose(r, p, atol=atol)
