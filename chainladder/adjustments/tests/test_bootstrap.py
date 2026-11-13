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
    Check that the hat adjustment factor is exactly 0 at the two corner cells,
    and that one ``random_state`` gives one hat matrix and one set of draws.

    Each corner cell has a parameter of its own, so 1 - H_ii is 0 there and its
    residual stays out of the resampling pool (Shapland and Leong 2010,
    section 3.2). Any other value at those cells can change the draws for a fixed
    ``random_state``, so every fit must match the first in all of ``hat_`` and
    ``resampled_triangles_``.
    """
    tri = cl.load_sample("genins")
    first = cl.BootstrapODPSample(n_sims=10, random_state=42).fit(tri)
    assert first.hat_[0, -1] == 0
    assert first.hat_[-1, 0] == 0
    for _ in range(4):
        model = cl.BootstrapODPSample(n_sims=10, random_state=42).fit(tri)
        np.testing.assert_array_equal(model.hat_, first.hat_)
        np.testing.assert_array_equal(
            model.resampled_triangles_.values, first.resampled_triangles_.values
        )
