import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import chainladder as cl
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r

pandas2ri.activate()
CL = importr('ChainLadder')

@pytest.fixture
def atol():
    return 1e-5

def mack_r(data, alpha, est_sigma):
    return r(f'mack<-MackChainLadder({data},alpha={alpha}, est.sigma="{est_sigma}", tail=TRUE)')


def mack_p(data, average, est_sigma):
    return cl.Exponential().fit(cl.Development(average=average, sigma_interpolation=est_sigma).fit(cl.load_dataset(data)))

def mack_p_no_tail(data, average, est_sigma):
    return cl.Development(average=average, sigma_interpolation=est_sigma).fit(cl.load_dataset(data))

data = ['RAA', 'ABC', 'GenIns', 'MW2008', 'MW2014']
# M3IR5 in R fails silently on exponential tail. Python actually computes it.
averages = [('simple', 0), ('volume', 1), ('regression', 2)]
est_sigma = [('mack', 'Mack'), ('log-linear', 'log-linear')]


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_tail_ldf(data, averages, est_sigma, atol):
    r = np.array(mack_r(data, averages[1], est_sigma[1]).rx('f'))
    p = mack_p(data, averages[0], est_sigma[0]).ldf_.triangle[0, 0, :, :]
    assert_allclose(r, p, atol=atol)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_tail_sigma(data, averages, est_sigma, atol):
    r = np.array(mack_r(data, averages[1], est_sigma[1]).rx('sigma'))
    p = mack_p(data, averages[0], est_sigma[0]).sigma_.triangle[0, 0, :, :]
    assert_allclose(r, p, atol=atol)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_tail_std_err(data, averages, est_sigma, atol):
    r = np.array(mack_r(data, averages[1], est_sigma[1]).rx('f.se'))
    p = mack_p(data, averages[0], est_sigma[0]).std_err_.triangle[0, 0, :, :]
    assert_allclose(r, p, atol=atol)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages[0:1])
@pytest.mark.parametrize('est_sigma', est_sigma[0:1])
def test_tail_doesnt_mutate_std_err(data, averages, est_sigma):
    p = mack_p(data, averages[0], est_sigma[0]).std_err_.triangle[:, :, :, :-1]
    p_no_tail = mack_p_no_tail(data, averages[0], est_sigma[0]).std_err_.triangle
    assert_equal(p_no_tail, p)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages[0:1])
@pytest.mark.parametrize('est_sigma', est_sigma[0:1])
def test_tail_doesnt_mutate_ldf_(data, averages, est_sigma):
    p = mack_p(data, averages[0], est_sigma[0]).ldf_.triangle[:, :, :, :-1]
    p_no_tail = mack_p_no_tail(data, averages[0], est_sigma[0]).ldf_.triangle
    assert_equal(p_no_tail, p)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages[0:1])
@pytest.mark.parametrize('est_sigma', est_sigma[0:1])
def test_tail_doesnt_mutate_sigma_(data, averages, est_sigma):
    p = mack_p(data, averages[0], est_sigma[0]).sigma_.triangle[:, :, :, :-1]
    p_no_tail = mack_p_no_tail(data, averages[0], est_sigma[0]).sigma_.triangle
    assert_equal(p_no_tail, p)
