import chainladder as cl
import numpy as np
from numpy.testing import assert_allclose


def test_constant_cdf():
    dev = cl.Development().fit(cl.load_dataset('raa'))
    link_ratios = {(num+1)*12: item
                   for num, item in enumerate(dev.ldf_.values[0,0,0,:])}
    dev_c = cl.DevelopmentConstant(
        patterns=link_ratios, style='ldf').fit(cl.load_dataset('raa'))
    assert_allclose(dev.cdf_.values, dev_c.cdf_.values, atol=1e-5)


def test_constant_ldf():
    dev = cl.Development().fit(cl.load_dataset('raa'))
    link_ratios = {(num+1)*12: item
                   for num, item in enumerate(dev.ldf_.values[0, 0, 0, :])}
    dev_c = cl.DevelopmentConstant(
        patterns=link_ratios, style='ldf').fit(cl.load_dataset('raa'))
    assert_allclose(dev.ldf_.values, dev_c.ldf_.values, atol=1e-5)
