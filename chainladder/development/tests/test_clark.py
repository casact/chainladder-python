from __future__ import annotations

import numpy as np
import chainladder as cl
import pytest


@pytest.fixture
def clark_genins():
    """
    The ``genins`` triangle as printed in Clark (2003). The paper's table
    has a typo in the 1991 origin at age 48 (2,182,708 instead of the
    2,218,270 in ``genins``), so that cell is edited to match the paper's
    published results.
    """
    tri = cl.load_sample("genins").copy()
    tri.values[0, 0, 0, 3] = 2_182_708
    return tri


@pytest.fixture
def clark_premium(clark_genins):
    premium = clark_genins.latest_diagonal * 0
    premium.values = (10_000_000 + 400_000 * np.arange(10))[None, None, :, None]
    return premium


def test_clark_ldf_scale(clark_genins):
    """LDF method scale parameter matches Clark (2003), p. 65."""
    model = cl.ClarkLDF().fit(clark_genins)
    assert model.scale_.iloc[0, 0] == pytest.approx(65_029, abs=1)


def test_clark_cape_cod_scale(clark_genins, clark_premium):
    """
    Cape Cod scale parameter matches Clark (2003), p. 68. The fitted
    incrementals must use premium * ELR as the expected ultimate rather
    than the LDF-implied ultimate (GH #1501).
    """
    model = cl.ClarkLDF().fit(clark_genins, sample_weight=clark_premium)
    assert model.elr_.iloc[0, 0] == pytest.approx(0.5978, abs=1e-4)
    assert model.scale_.iloc[0, 0] == pytest.approx(61_577, abs=1)


def test_clark_cape_cod_incremental_fits(clark_genins, clark_premium):
    """Cape Cod fitted incrementals are premium * ELR * (G(y) - G(x))."""
    model = cl.ClarkLDF().fit(clark_genins, sample_weight=clark_premium)
    fits = model.incremental_fits_.values
    ultimate = clark_premium.values * model.elr_.iloc[0, 0]
    # Expected loss at age 12 for every origin is ultimate * G(6)
    g6 = model.G_(6.0).values[0, 0, 0, 0]
    assert np.allclose(fits[..., 0:1], ultimate * g6)
