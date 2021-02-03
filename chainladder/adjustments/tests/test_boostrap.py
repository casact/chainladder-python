import chainladder as cl
import numpy as np


def test_bs_sample():
    tri = cl.load_sample("raa")
    a = (
        cl.Development()
        .fit(cl.BootstrapODPSample(n_sims=40000).fit_transform(tri).mean())
        .ldf_
    )
    b = cl.Development().fit_transform(tri).ldf_
    assert tri.get_array_module().all(abs(((a - b) / b).values) < 0.005)
