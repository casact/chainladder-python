import chainladder as cl
import numpy as np
import pandas as pd


def test_bs_sample(raa):
    tri = raa
    a = (
        cl.Development()
        .fit(cl.BootstrapODPSample(n_sims=40000).fit_transform(tri).mean())
        .ldf_
    )
    b = cl.Development().fit_transform(tri).ldf_
    assert tri.get_array_module().all(abs(((a - b) / b).values) < 0.005)

def test_bs_multiple_cols():
    assert cl.BootstrapODPSample().fit_transform(
        cl.load_sample('berqsherm').iloc[0]).shape == (1000, 4, 8, 8)

def test_multi_index(clrd):
    tri = clrd['CumPaidLoss'].sum()
    resampled_triangles = cl.BootstrapODPSample().fit(tri).resampled_triangles_
    resampled_triangles.index
    assert np.all(resampled_triangles.index == pd.DataFrame(np.concat((np.array([['(All)','(All)']] * 1000),np.arange(1000).reshape(-1,1)),axis=1),columns=['GRNAME','LOB','Simulation_#']))