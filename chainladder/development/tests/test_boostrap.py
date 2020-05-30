import chainladder as cl
import numpy as np


def test_bs_sample():
    tri = cl.load_sample('raa')
    a = cl.Development().fit(cl.BootstrapODPSample(n_sims=40000).fit_transform(tri).mean()).ldf_
    b = cl.Development().fit_transform(tri).ldf_
    assert np.all(abs(((a-b)/b).values)<.005)
