import chainladder as cl
import numpy as np
import pandas as pd


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

def test_constant_callable(clrd, atol):
    agway = clrd.loc['Agway Ins Co', 'CumPaidLoss']
    def paid_cdfs(x):
        """ A function that returns different CDFs depending on a specified LOB """
        cdfs = {
        'comauto': [3.832, 1.874, 1.386, 1.181, 1.085, 1.043, 1.022, 1.013, 1.007, 1],
        'medmal': [24.168, 4.127, 2.103, 1.528, 1.275, 1.161, 1.088, 1.047, 1.018, 1],
        'othliab': [10.887, 3.416, 1.957, 1.433, 1.231, 1.119, 1.06, 1.031, 1.011, 1],
        'ppauto': [2.559, 1.417, 1.181, 1.084, 1.04, 1.019, 1.009, 1.004, 1.001, 1],
        'prodliab': [13.703, 5.613, 2.92, 1.765, 1.385, 1.177, 1.072, 1.034, 1.008, 1],
        'wkcomp': [4.106, 1.865, 1.418, 1.234, 1.141, 1.09, 1.056, 1.03, 1.01, 1]}
        patterns = pd.DataFrame(cdfs, index=range(12, 132, 12)).T
        return patterns.loc[x.loc['LOB']].to_dict()
    model = cl.DevelopmentConstant(patterns=paid_cdfs, callable_axis=1, style='cdf')
    assert abs(model.fit_transform(agway).cdf_.loc['comauto'].iloc[..., 0].sum() - 3.832) < atol
