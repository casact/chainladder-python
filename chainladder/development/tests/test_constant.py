import chainladder as cl
import numpy as np


def test_constant_cdf(raa):
    dev = cl.Development().fit(raa)
    xp = dev.ldf_.get_array_module()
    link_ratios = {
        (num + 1) * 12: item for num, item in enumerate(dev.ldf_.values[0, 0, 0, :])
    }
    dev_c = cl.DevelopmentConstant(patterns=link_ratios, style="ldf").fit(raa)
    assert xp.allclose(dev.cdf_.values, dev_c.cdf_.values, atol=1e-5)


def test_constant_ldf(raa):
    dev = cl.Development().fit(raa)
    xp = dev.ldf_.get_array_module()
    link_ratios = {
        (num + 1) * 12: item for num, item in enumerate(dev.ldf_.values[0, 0, 0, :])
    }
    dev_c = cl.DevelopmentConstant(patterns=link_ratios, style="ldf").fit(raa)
    assert xp.allclose(dev.ldf_.values, dev_c.ldf_.values, atol=1e-5)
