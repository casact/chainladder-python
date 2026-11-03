from __future__ import annotations

import numpy as np
import chainladder as cl
import pytest

# ClarkLDF fits omega and theta numerically, and the optimizer stops at
# slightly different points across Python versions and numpy/scipy/pandas
# builds. That moves the scale parameter by a few units and the ELR in the
# fifth decimal, so fitted values are compared with the paper's rounded
# figures at a relative tolerance of 0.1%. A Cape Cod scale computed from
# the LDF-implied ultimate (53,822) is about 13% away and fails the check.
PAPER_RTOL = 1e-3


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
    """LDF method scale parameter matches Clark (2003), p. 61."""
    model = cl.ClarkLDF().fit(clark_genins)
    assert model.scale_.iloc[0, 0] == pytest.approx(65_029, rel=PAPER_RTOL)


def test_clark_cape_cod_scale(clark_genins, clark_premium):
    """
    Cape Cod ELR matches Clark (2003), p. 67, and the scale parameter
    matches p. 68. The fitted incrementals must use premium * ELR as the
    expected ultimate rather than the LDF-implied ultimate (GH #1501).
    """
    model = cl.ClarkLDF().fit(clark_genins, sample_weight=clark_premium)
    assert model.elr_.iloc[0, 0] == pytest.approx(0.5978, rel=PAPER_RTOL)
    assert model.scale_.iloc[0, 0] == pytest.approx(61_577, rel=PAPER_RTOL)


# Fitted incrementals for the Cape Cod method, Clark (2003), Table 2.1, p. 80.
# One row per origin (1991 to 2000), one column per 12-month development step.
CAPE_COD_FITTED = [
    [280569, 882582, 845554, 691227, 542171, 422833, 332202, 264171, 212900, 173860],
    [291792, 917885, 879376, 718876, 563858, 439746, 345490, 274738, 221416],
    [303015, 953188, 913198, 746525, 585545, 456660, 358778, 285305],
    [314238, 988491, 947020, 774174, 607232, 473573, 372066],
    [325460, 1023795, 980842, 801823, 628919, 490486],
    [336683, 1059098, 1014664, 829472, 650606],
    [347906, 1094401, 1048487, 857121],
    [359129, 1129704, 1082309],
    [370351, 1165008],
    [381574],
]


def test_clark_cape_cod_incremental_fits(clark_genins, clark_premium):
    """Cape Cod fitted incrementals match Clark (2003), Table 2.1, p. 80."""
    model = cl.ClarkLDF().fit(clark_genins, sample_weight=clark_premium)
    expected = np.full((10, 10), np.nan)
    for i, row in enumerate(CAPE_COD_FITTED):
        expected[i, : len(row)] = row
    assert model.incremental_fits_.values[0, 0] == pytest.approx(
        expected, rel=PAPER_RTOL, nan_ok=True
    )


def test_clark_cape_cod_sparse_premium(clark_genins, clark_premium):
    """A premium triangle on the sparse backend gives the same Cape Cod fit."""
    dense = cl.ClarkLDF().fit(clark_genins, sample_weight=clark_premium)
    sparse = cl.ClarkLDF().fit(
        clark_genins, sample_weight=clark_premium.set_backend("sparse")
    )
    assert sparse.scale_.iloc[0, 0] == pytest.approx(dense.scale_.iloc[0, 0])
    assert np.allclose(
        sparse.incremental_fits_.values,
        dense.incremental_fits_.values,
        equal_nan=True,
    )
