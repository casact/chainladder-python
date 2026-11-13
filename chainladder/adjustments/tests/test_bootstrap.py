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


def test_random_state_none_records_the_seed(raa):
    """
    Check that ``fit`` records the integer it draws for ``random_state=None``
    as ``random_state_``, and that passing it back repeats the run (#1568).
    """
    model = cl.BootstrapODPSample(n_sims=20).fit(raa)
    seed = model.random_state_
    assert isinstance(seed, int)
    sims = model.transform(raa)
    assert sims.random_state == seed
    ultimate = cl.Chainladder().fit(sims).ultimate_.values
    repeat = _bootstrap_run(raa, seed)
    np.testing.assert_array_equal(sims.values, repeat[0])
    np.testing.assert_array_equal(ultimate, repeat[1])


def test_random_state_given_is_recorded(raa):
    """Check that an integer or an instance passed in is kept as ``random_state_``."""
    assert cl.BootstrapODPSample(n_sims=5, random_state=42).fit(raa).random_state_ == 42
    generator = np.random.RandomState(42)
    model = cl.BootstrapODPSample(n_sims=5, random_state=generator).fit(raa)
    assert model.random_state_ is generator


def test_random_state_none_independent_columns():
    """
    Check that with several columns and ``random_state=None`` each column draws
    its own seed, so two identical columns resample differently, that
    ``np.random.seed`` repeats the run, and that an integer seed is still shared
    by every column.
    """
    tri = cl.load_sample("raa")
    tri["copy"] = tri["values"]
    np.random.seed(3)
    first = cl.BootstrapODPSample(n_sims=5).fit(tri).resampled_triangles_.values
    assert not np.allclose(np.nan_to_num(first[:, 0]), np.nan_to_num(first[:, 1]))
    np.random.seed(3)
    second = cl.BootstrapODPSample(n_sims=5).fit(tri).resampled_triangles_.values
    np.testing.assert_array_equal(first, second)
    seeded = cl.BootstrapODPSample(n_sims=5, random_state=7).fit(tri)
    seeded = seeded.resampled_triangles_.values
    np.testing.assert_array_equal(seeded[:, 0], seeded[:, 1])
