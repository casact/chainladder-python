import numpy as np
import pytest
from numpy.testing import assert_allclose

data = ['RAA', 'ABC', 'GenIns', 'M3IR5', 'MW2008', 'MW2014']
# Mortgage 2 fail

@pytest.mark.parametrize('data', data)
def test_mack_simple_sigma(mack_r_simple, mack_p_simple, data, atol):
    assert_allclose(np.array(mack_r_simple[data].rx('sigma')),
                    mack_p_simple[data].sigma_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_volume_sigma(mack_r_volume, mack_p_volume, data, atol):
    assert_allclose(np.array(mack_r_volume[data].rx('sigma')),
                    mack_p_volume[data].sigma_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_reg_sigma(mack_r_reg, mack_p_reg, data, atol):
    assert_allclose(np.array(mack_r_reg[data].rx('sigma')),
                    mack_p_reg[data].sigma_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_simple_f(mack_r_simple, mack_p_simple, data, atol):
    assert_allclose(np.array(mack_r_simple[data].rx('f'))[:, :-1],
                    mack_p_simple[data].ldf_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_volume_f(mack_r_volume, mack_p_volume, data, atol):
    assert_allclose(np.array(mack_r_volume[data].rx('f'))[:, :-1],
                    mack_p_volume[data].ldf_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_reg_f(mack_r_reg, mack_p_reg, data, atol):
    assert_allclose(np.array(mack_r_reg[data].rx('f'))[:, :-1],
                    mack_p_reg[data].ldf_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_simple_fse(mack_r_simple, mack_p_simple, data, atol):
    assert_allclose(np.array(mack_r_simple[data].rx('f.se')),
                    mack_p_simple[data].std_err_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_volume_fse(mack_r_volume, mack_p_volume, data, atol):
    assert_allclose(np.array(mack_r_volume[data].rx('f.se')),
                    mack_p_volume[data].std_err_.triangle[0, 0, :, :],
                    atol=atol)


@pytest.mark.parametrize('data', data)
def test_mack_reg_fse(mack_r_reg, mack_p_reg, data, atol):
    assert_allclose(np.array(mack_r_reg[data].rx('f.se')),
                    mack_p_reg[data].std_err_.triangle[0, 0, :, :],
                    atol=atol)
