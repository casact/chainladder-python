import chainladder as cl
import numpy as np
import pandas as pd


def test_bs_sample(raa):
    tri = raa
    a = (
        cl
        .Development()
        .fit(cl.BootstrapODPSample(n_sims=40000).fit_transform(tri).mean())
        .ldf_
    )
    b = cl.Development().fit_transform(tri).ldf_
    assert tri.get_array_module().all(abs(((a - b) / b).values) < 0.005)


def test_bs_multiple_cols():
    assert cl.BootstrapODPSample().fit_transform(
        cl.load_sample("berqsherm").iloc[0]
    ).shape == (1000, 4, 8, 8)


def test_multi_index(clrd):
    tri = clrd["CumPaidLoss"].sum()
    resampled_triangles = cl.BootstrapODPSample().fit(tri).resampled_triangles_
    expected = pd.concat([tri.index] * 1000, ignore_index=True)
    expected["Simulation_#"] = np.arange(1000)
    pd.testing.assert_frame_equal(resampled_triangles.index, expected)


def test_bootstrap_odp_val_triangle_error(raa):
    import pytest

    with pytest.raises(
        ValueError, match="Triangle must be expressed with development lags"
    ):
        cl.BootstrapODPSample().fit(raa.dev_to_val())
