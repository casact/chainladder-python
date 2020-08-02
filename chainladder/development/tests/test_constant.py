import chainladder as cl
import numpy as np
from chainladder.utils.cupy import cp

def test_constant_cdf():
    dev = cl.Development().fit(cl.load_sample('raa'))
    xp = cp.get_array_module(dev.ldf_.values)
    link_ratios = {(num+1)*12: item
                   for num, item in enumerate(dev.ldf_.values[0,0,0,:])}
    dev_c = cl.DevelopmentConstant(
        patterns=link_ratios, style='ldf').fit(cl.load_sample('raa'))
    assert xp.allclose(dev.cdf_.values, dev_c.cdf_.values, atol=1e-5)


def test_constant_ldf():
    dev = cl.Development().fit(cl.load_sample('raa'))
    xp = cp.get_array_module(dev.ldf_.values)
    link_ratios = {(num+1)*12: item
                   for num, item in enumerate(dev.ldf_.values[0, 0, 0, :])}
    dev_c = cl.DevelopmentConstant(
        patterns=link_ratios, style='ldf').fit(cl.load_sample('raa'))
    assert xp.allclose(dev.ldf_.values, dev_c.ldf_.values, atol=1e-5)
