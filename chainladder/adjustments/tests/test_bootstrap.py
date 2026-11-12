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


def test_hat_adjustment_is_zero_at_corners():
    """
    Check that the hat adjustment factor is exactly 0 at the two corner cells.

    Each corner cell has a parameter of its own, so 1 - H_ii is 0 there and its
    residual stays out of the resampling pool (Shapland and Leong 2010,
    section 3.2). Any other value at those cells can change the draws for a fixed
    ``random_state``.
    """
    tri = cl.load_sample("genins")
    for _ in range(5):
        hat = cl.BootstrapODPSample(n_sims=10, random_state=42).fit(tri).hat_
        assert hat[0, -1] == 0
        assert hat[-1, 0] == 0
