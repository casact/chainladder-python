"""
Tests for the one-year claims development result of MackChainladder.

Sources are cited by their keys in docs/library/references.bib.

What this module checks for each attribute, by source:

====================  =========  =========  =========  ============
Attribute             R mw2008   R mw2014   merz2008   per-triangle
====================  =========  =========  =========  ============
cdr_std_err_          published  published  published  unchanged
total_cdr_std_err_    published  published  published  unchanged
mack_std_err_         published  published  no test    no test
total_mack_std_err_   published  published  no test    no test
====================  =========  =========  =========  ============

published
    R mw2008 and R mw2014 are the CDR() examples printed on the reference page
    of gesmann2026, for MackChainLadder fits with est.sigma="Mack". Each value
    matches to half a unit in the last printed decimal (4 decimals for mw2008,
    7 for mw2014), for every origin and the total. merz2008 is the one-year
    column of Table 4, printed in whole units and checked to 2 units for the
    reason given in its test.
unchanged
    A fit on clrd summed by LOB, paid and incurred, gives each of its twelve
    triangles the values of a fit on that triangle alone.

The Mack rows confirm that sigma_interpolation="mack" reproduces R's
est.sigma="Mack"; test_mack.py checks the Mack attributes against mack1994 and
mack1999.

The R and paper figures are checked on the numpy and sparse backends. Other
tests check that an incremental triangle gives the cumulative values, that a
triangle with a missing cell gets NaN, that each unsupported development
pattern fits and then raises ValueError on reading the CDR attributes, and
that reading them leaves the Mack attributes as they were.
"""

from __future__ import annotations

import numpy as np

import chainladder as cl
import pytest

# gesmann2026, CDR() reference page: CDR(1)S.E. and Mack.S.E. for MW2008 by
# origin (1 to 9), then the totals.
MW2008_CDR = [
    0.0000,
    566.1744,
    1486.5603,
    3923.0986,
    9722.8598,
    28442.6216,
    20954.2870,
    28119.3180,
    53320.8210,
]
MW2008_MACK = [
    0.0000,
    566.1744,
    1563.8075,
    4157.2733,
    10536.4380,
    30319.4638,
    35967.0384,
    45090.1821,
    69552.3397,
]
MW2008_TOTALS = (81080.5468, 108401.3875)

# gesmann2026, CDR() reference page: the same columns for MW2014, origins 1
# to 17.
MW2014_CDR = [
    0.0000000,
    0.4083149,
    2.5393857,
    16.7232632,
    156.4022713,
    137.6522771,
    171.1812092,
    70.3161155,
    271.6352221,
    310.1268449,
    103.3834357,
    632.6388191,
    315.0489135,
    406.1424672,
    285.2076540,
    668.2337878,
    733.2222786,
]
MW2014_MACK = [
    0.0000000,
    0.4083149,
    2.5652899,
    16.8984949,
    157.2756452,
    207.1650862,
    261.9266093,
    292.2622285,
    390.5874717,
    502.0606072,
    486.0911099,
    806.9028971,
    793.9381916,
    891.6613403,
    916.4940218,
    1106.1262716,
    1295.6909824,
]
MW2014_TOTALS = (1842.8507073, 3233.6807352)

# merz2008, Table 4 (p. 562): the one-year column for accident years 1 to 8,
# the sample's origins 2002 to 2009, then the total.
TABLE4_CDR = [567, 1488, 3923, 9723, 28443, 20954, 28119, 53320]
TABLE4_TOTAL = 81080

UNSUPPORTED = {
    "tail": (
        lambda t: cl.TailCurve().fit_transform(cl.Development().fit_transform(t)),
        "without a tail",
    ),
    "n_periods": (lambda t: cl.Development(n_periods=5).fit_transform(t), "every"),
    "drop_high": (lambda t: cl.Development(drop_high=1).fit_transform(t), "every"),
    "drop": (lambda t: cl.Development(drop=("2003", 12)).fit_transform(t), "every"),
    "simple": (
        lambda t: cl.Development(average="simple").fit_transform(t),
        "volume-weighted",
    ),
    "fillna": (lambda t: cl.Development(fillna=1.0).fit_transform(t), "own link"),
    "not_square": (lambda t: t[t.origin < "2009"], "square"),
}


def mack_sigma_fit(tri):
    """MackChainladder with Mack's extrapolation of the last sigma."""
    dev = cl.Development(sigma_interpolation="mack").fit_transform(tri)
    return cl.MackChainladder().fit(dev)


def by_origin(triangle):
    """Values of a single-triangle per-origin Triangle as a numpy vector."""
    return triangle.set_backend("numpy").values[0, 0, :, -1]


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize(
    "sample, cdr, mack, totals, decimals",
    [
        ("mw2008", MW2008_CDR, MW2008_MACK, MW2008_TOTALS, 4),
        ("mw2014", MW2014_CDR, MW2014_MACK, MW2014_TOTALS, 7),
    ],
)
def test_matches_r(sample, cdr, mack, totals, decimals, backend):
    """
    One-year and Mack standard errors match R's CDR() output.

    The figures are the MW2008 and MW2014 examples of CDR() in gesmann2026,
    fitted with est.sigma="Mack"; the samples mw2008 and mw2014 hold the same
    triangles. Each value matches to half a unit in the last decimal R
    prints. R prints 0 for the oldest origin, which is
    fully developed; cdr_std_err_ and mack_std_err_ give NaN there.
    """
    model = mack_sigma_fit(cl.load_sample(sample).set_backend(backend))
    atol = 0.5 * 10.0**-decimals
    assert np.isnan(by_origin(model.cdr_std_err_)[0])
    assert np.isnan(by_origin(model.mack_std_err_)[0])
    np.testing.assert_allclose(
        by_origin(model.cdr_std_err_)[1:], cdr[1:], rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        by_origin(model.mack_std_err_)[1:], mack[1:], rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        model.total_cdr_std_err_.values[0, 0], totals[0], rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        model.total_mack_std_err_.values[0, 0], totals[1], rtol=0, atol=atol
    )


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
def test_matches_merz2008_table4(backend):
    """
    One-year standard errors match merz2008, Table 4, to 2 units.

    Table 4 prints whole units of $1,000, and Table 2 prints the sigma^2
    estimates to two decimals. Against the unrounded figures that R and this
    implementation share, the entries for accident years 1 and 2 are 0.83 and
    1.44 higher; every other entry and the total are within 1.
    """
    model = mack_sigma_fit(cl.load_sample("mw2008").set_backend(backend))
    np.testing.assert_allclose(
        by_origin(model.cdr_std_err_)[1:], TABLE4_CDR, rtol=0, atol=2
    )
    np.testing.assert_allclose(
        model.total_cdr_std_err_.values[0, 0], TABLE4_TOTAL, rtol=0, atol=2
    )


def test_multi_triangle_matches_single(clrd):
    """
    A fit on several triangles gives each the values of its own fit.

    clrd summed by LOB, paid and incurred, is twelve complete triangles. Both
    attributes of the joint fit match a fit on each triangle alone.
    """
    tri = clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss"]]
    model = cl.MackChainladder().fit(tri)
    per = model.cdr_std_err_.set_backend("numpy").values
    total = model.total_cdr_std_err_.values
    assert np.isfinite(total).all()
    for i in range(len(tri.index)):
        for j in range(len(tri.columns)):
            single = cl.MackChainladder().fit(tri.iloc[i, j])
            np.testing.assert_allclose(
                per[i, j, :, 0], by_origin(single.cdr_std_err_), rtol=1e-10
            )
            np.testing.assert_allclose(
                total[i, j], single.total_cdr_std_err_.values[0, 0], rtol=1e-10
            )


def test_incremental_matches_cumulative():
    """An incremental triangle gives the values of its cumulative form."""
    tri = cl.load_sample("mw2008")
    cumulative = mack_sigma_fit(tri)
    incremental = mack_sigma_fit(tri.cum_to_incr())
    np.testing.assert_allclose(
        by_origin(incremental.cdr_std_err_),
        by_origin(cumulative.cdr_std_err_),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        incremental.total_cdr_std_err_.values,
        cumulative.total_cdr_std_err_.values,
        rtol=1e-12,
    )


def test_missing_cell_gives_nan():
    """
    A triangle with a missing cell above the latest diagonal gets NaN.

    The estimators of merz2008 use every cell of the upper triangle. Removing
    one cell from mw2008 leaves MackChainladder able to fit, and both CDR
    attributes are NaN.
    """
    tri = cl.load_sample("mw2008")
    tri.values = tri.values.copy()
    tri.values[0, 0, 3, 1] = np.nan
    model = mack_sigma_fit(tri)
    assert np.isfinite(model.total_mack_std_err_.values).all()
    assert np.isnan(by_origin(model.cdr_std_err_)).all()
    assert np.isnan(model.total_cdr_std_err_.values).all()


@pytest.mark.parametrize("name", list(UNSUPPORTED))
def test_unsupported_pattern_raises_on_read(name):
    """
    Unsupported development patterns fit, then raise on reading the CDR.

    merz2008 assumes a square triangle, volume-weighted factors from every
    link ratio of the triangle and no tail. MackChainladder still fits each
    pattern here and gives its Mack standard errors; only the two CDR
    attributes raise ValueError.
    """
    transform, message = UNSUPPORTED[name]
    model = cl.MackChainladder().fit(transform(cl.load_sample("mw2008")))
    assert np.isfinite(model.total_mack_std_err_.values).all()
    with pytest.raises(ValueError, match=message):
        _ = model.cdr_std_err_
    with pytest.raises(ValueError, match=message):
        _ = model.total_cdr_std_err_


def test_reading_cdr_leaves_mack_unchanged():
    """Reading the CDR attributes leaves the fitted Mack attributes as they were."""
    model = mack_sigma_fit(cl.load_sample("mw2008"))
    summary = model.summary_.values.copy()
    total = model.total_mack_std_err_.values.copy()
    assert np.isfinite(by_origin(model.cdr_std_err_)[1:]).all()
    assert np.isfinite(model.total_cdr_std_err_.values).all()
    np.testing.assert_array_equal(model.summary_.values, summary)
    np.testing.assert_array_equal(model.total_mack_std_err_.values, total)
