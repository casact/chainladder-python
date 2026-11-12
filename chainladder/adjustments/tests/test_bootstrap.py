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


def test_hat_adjustment_is_zero_at_corners(genins):
    """
    Check that the hat adjustment factor is exactly 0 at the two corner cells,
    and that one ``random_state`` gives one hat matrix and one set of draws.

    Each corner cell has a parameter of its own, so 1 - H_ii is 0 there and its
    residual stays out of the resampling pool (Shapland and Leong 2010,
    section 3.2). Any other value at those cells can change the draws for a fixed
    ``random_state``, so every fit must match the first in all of ``hat_`` and
    ``resampled_triangles_``.
    """
    tri = genins
    first = cl.BootstrapODPSample(n_sims=10, random_state=42).fit(tri)
    assert first.hat_[0, -1] == 0
    assert first.hat_[-1, 0] == 0
    for _ in range(4):
        model = cl.BootstrapODPSample(n_sims=10, random_state=42).fit(tri)
        np.testing.assert_array_equal(model.hat_, first.hat_)
        np.testing.assert_array_equal(
            model.resampled_triangles_.values, first.resampled_triangles_.values
        )


def _bootstrap_run(tri, random_state):
    """Return the resampled values and the Chainladder ultimates for one run."""
    sims = cl.BootstrapODPSample(n_sims=20, random_state=random_state).fit_transform(
        tri
    )
    return sims.values, cl.Chainladder().fit(sims).ultimate_.values


def test_random_state_zero_reproduces_ultimates(raa):
    """
    Check that ``random_state=0`` reproduces the process-variance draws.

    Seed 0 used to give the process-variance step an unseeded generator, so
    the resampled triangles repeated and the ultimates did not (#1568).
    """
    first, second = _bootstrap_run(raa, 0), _bootstrap_run(raa, 0)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_random_state_none_uses_global_generator(raa):
    """
    Check that ``random_state=None`` draws from the global ``np.random``
    generator, as the docstring says, so ``np.random.seed`` repeats a run.
    """
    np.random.seed(1)
    first = _bootstrap_run(raa, None)
    np.random.seed(1)
    second = _bootstrap_run(raa, None)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_random_state_accepts_random_state_instance(raa):
    """
    Check that a ``RandomState`` instance is used as the generator.

    The resampled triangles match those from the integer seed the instance
    was built with.
    """
    from_instance = _bootstrap_run(raa, np.random.RandomState(42))
    np.testing.assert_array_equal(
        from_instance[0], _bootstrap_run(raa, np.random.RandomState(42))[0]
    )
    np.testing.assert_array_equal(from_instance[0], _bootstrap_run(raa, 42)[0])
