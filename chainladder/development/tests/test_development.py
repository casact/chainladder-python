import numpy as np
from chainladder.utils.cupy import cp
import pytest
import chainladder as cl
from rpy2.robjects.packages import importr
from rpy2.robjects import r


CL = importr('ChainLadder')


@pytest.fixture
def atol():
    return 1e-5


def mack_r(data, alpha, est_sigma):
    return r('mack<-MackChainLadder({},alpha={}, est.sigma="{}")'.format(data, alpha, est_sigma))


def mack_p(data, average, est_sigma):
    return cl.Development(average=average, sigma_interpolation=est_sigma).fit_transform(cl.load_dataset(data))


data = ['RAA', 'GenIns', 'MW2014']
averages = [('simple', 0), ('volume', 1), ('regression', 2)]
est_sigma = [('mack', 'Mack'), ('log-linear', 'log-linear')]


def test_full_slice():
    assert cl.Development().fit_transform(cl.load_dataset('GenIns')).ldf_ == \
        cl.Development(n_periods=1000).fit_transform(cl.load_dataset('GenIns')).ldf_


def test_full_slice2():
    assert cl.Development().fit_transform(cl.load_dataset('GenIns')).ldf_ == \
        cl.Development(n_periods=[1000]*(cl.load_dataset('GenIns').shape[3]-1)).fit_transform(cl.load_dataset('GenIns')).ldf_

def test_drop1():
    raa = cl.load_dataset('raa')
    assert cl.Development(drop=('1982', 12)).fit(raa).ldf_.values[0, 0, 0, 0] == \
           cl.Development(drop_high=[True]+[False]*8).fit(raa).ldf_.values[0, 0, 0, 0]

def test_drop2():
    raa = cl.load_dataset('raa')
    assert cl.Development(drop_valuation='1981').fit(raa).ldf_.values[0, 0, 0, 0] == \
           cl.Development(drop_low=[True]+[False]*8).fit(raa).ldf_.values[0, 0, 0, 0]

def test_n_periods():
    d = cl.load_dataset('usauto')['incurred']
    xp = cp.get_array_module(d.values)
    return xp.all(xp.around(xp.unique(
        cl.Development(n_periods=3, average='volume').fit(d).ldf_.values,
        axis=-2), 3).flatten() ==
        xp.array([1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.0]))

@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_ldf(data, averages, est_sigma, atol):
    p = mack_p(data, averages[0], est_sigma[0]).ldf_.values[0, 0, :, :]
    xp = cp.get_array_module(p)
    r = xp.array(mack_r(data, averages[1], est_sigma[1]).rx('f'))[:, :-1]
    p = xp.unique(p, axis=-2)
    xp.testing.assert_allclose(r, p, atol=atol)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_sigma(data, averages, est_sigma, atol):
    p = mack_p(data, averages[0], est_sigma[0]).sigma_.values[0, 0, :, :]
    xp = cp.get_array_module(p)
    p = xp.unique(p, axis=-2)
    r = xp.array(mack_r(data, averages[1], est_sigma[1]).rx('sigma'))
    xp.testing.assert_allclose(r, p, atol=atol)


@pytest.mark.parametrize('data', data)
@pytest.mark.parametrize('averages', averages)
@pytest.mark.parametrize('est_sigma', est_sigma)
def test_mack_std_err(data, averages, est_sigma, atol):
    p = mack_p(data, averages[0], est_sigma[0]).std_err_.values[0, 0, :, :]
    xp = cp.get_array_module(p)
    r = xp.array(mack_r(data, averages[1], est_sigma[1]).rx('f.se'))
    p = xp.unique(p, axis=-2)
    xp.testing.assert_allclose(r, p, atol=atol)

def test_assymetric_development():
    quarterly = cl.load_dataset('quarterly')['paid']
    xp = cp.get_array_module(quarterly.values)
    dev = cl.Development(n_periods=1, average='simple').fit(quarterly)
    dev2 = cl.Development(n_periods=1, average='regression').fit(quarterly)
    xp.testing.assert_allclose(dev.ldf_.values, dev2.ldf_.values, atol=1e-5)
