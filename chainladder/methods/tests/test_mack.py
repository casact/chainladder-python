"""
Tests for MackChainladder.

test_mack1994_hardcode and test_mack1999_hardcode check the Mack attributes
against mack1994 and mack1999. The tests from test_cdr_matches_merz2008 on
cover the one-year claims development result (CDR) and cite sources by their
keys in docs/library/references.bib. What these CDR tests check for each
attribute, by source:

Attribute                 merz2008  merz2014  per-triangle
ldf_, sigma_              Table 2   Table 1   --
ibnr_                     --        Table 2   --
process_risk_             --        Table 2   --
parameter_risk_           --        Table 2   --
mack_std_err_             Table 4   Table 2   --
total_process_risk_       --        Table 2   --
total_parameter_risk_     --        Table 2   --
total_mack_std_err_       Table 4   Table 2   --
cdr_std_err_              Table 4   --        clrd
total_cdr_std_err_        Table 4   Table 3   clrd

merz2008
    The paper prints Tables 2 and 4 for its own triangle, the sample mw2008,
    with Mack's extrapolation of the last sigma (sigma_interpolation="mack").
    Table 2 (p. 560) gives the factors to 4 decimals and sigma^2 to 2. Table 4
    (p. 562) gives the one-year and Mack columns per accident year, the
    cov^(1/2) row and the totals, in whole units of $1,000.
merz2014
    Table 1 (p. 28) of this paper is the 17-year triangle of the sample
    mw2014, fitted with Mack's extrapolation too (p. 25). Page numbers are
    those of the version dated July 3, 2015. Table 1 gives the factors to 5
    decimals and sigma^2 to 4. Table 2 (p. 30) gives, per accident year and in
    total, the reserve and the square roots of the process variance, the
    parameter uncertainty and Mack's msep, in whole units, with the Mack
    total to 1 decimal. Table 3 (p. 31), row t = 17, gives the square root of
    the one-year uncertainty in total, 1,843. The paper prints no one-year
    figures per accident year, so the per-origin CDR is checked on mw2008 only.
per-triangle
    A fit on clrd summed by LOB, paid and incurred, gives each of its twelve
    triangles the values of a fit on that triangle alone.

test_cdr_matches_merz2008 and test_cdr_matches_merz2014 state the tolerance
for each printed figure. The mw2008, mw2014 and clrd fixtures run these checks
on the numpy and sparse backends. Other CDR tests check that an incremental
triangle gives the values of its cumulative form, that a triangle with a
missing cell gets NaN, that each unsupported development pattern fits and then
raises ValueError on reading the CDR attributes, and that reading them leaves
the Mack attributes as they were.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import chainladder as cl
import pytest

if TYPE_CHECKING:
    from chainladder import Triangle


def test_mack_to_triangle():
    assert (
        cl
        .MackChainladder()
        .fit(
            cl.TailConstant().fit_transform(
                cl.Development().fit_transform(cl.load_sample("ABC"))
            )
        )
        .summary_.equals(
            cl
            .MackChainladder()
            .fit(cl.Development().fit_transform(cl.load_sample("ABC")))
            .summary_
        )
    )


def test_mack_malformed(raa):
    raa_alt = raa.copy().iloc[:, :, :-1]
    x = cl.MackChainladder().fit(raa)
    y = cl.MackChainladder().fit(raa_alt)
    assert x.process_risk_.iloc[:, :, :-1].equals(y.process_risk_)


def test_multi_triangle_mack(clrd: Triangle, atol: float) -> None:
    """
    Tests that applying MackChainladder on a multi-triangle returns the same
    results as applying MackChainladder on each single-triangle individually
    """
    tri = clrd.groupby("LOB").sum()["IncurLoss", "CumPaidLoss"]
    mack = cl.MackChainladder().fit(tri)
    for i in range(len(tri.index)):
        for j in range(len(tri.columns)):
            assert np.all(
                abs(
                    mack.full_std_err_.iloc[i, j].values
                    - cl.MackChainladder().fit(tri.iloc[i, j]).full_std_err_.values
                )
                < atol
            )


def test_mack1999_hardcode():
    """
    Reconciles key MackChainladder statistics to values provided in the paper
    """
    # sourced from Table 1, p365 of Mack(1999)
    ldf_se = [2.24, 0.517, 0.122, 0.051, 0.042, 0.023, 0.015, 0.012]
    sigma = [1337, 988.5, 440.1, 207.0, 164.2, 74.6, 35.49, 16.89]
    # sourced from Table 2, p366 of Mack(1999)
    ibnr_se = [0, 61, 140, 319, 596, 1038, 1298, 1806, 2182]

    tri = cl.load_sample("mortgage")
    dev = cl.Development(sigma_interpolation="mack").fit_transform(tri)
    model = cl.MackChainladder().fit(dev)
    ldf_rhs = dev.std_err_.values.flatten()
    assert np.allclose(ldf_se[0], ldf_rhs[0], atol=0.01)
    assert np.allclose(ldf_se[1:], ldf_rhs[1:], atol=0.001)
    sigma_rhs = dev.sigma_.values.flatten()
    assert np.allclose(sigma[0], sigma_rhs[0], atol=1)
    assert np.allclose(sigma[1:], sigma_rhs[1:], atol=0.1)
    ibnr_rhs = model.summary_.values[0, 0, :, -1] / 1000
    assert np.allclose(ibnr_se, np.nan_to_num(ibnr_rhs, nan=0), atol=1, equal_nan=True)


def _assert_incremental_matches_cumulative(cum):
    """An incremental triangle gives the same Mack estimates as its cumulative form."""
    attrs = [
        "parameter_risk_",
        "process_risk_",
        "total_parameter_risk_",
        "total_process_risk_",
        "mack_std_err_",
        "full_std_err_",
    ]
    incr = cum.cum_to_incr()
    model_cum = cl.MackChainladder().fit(cum)
    model_incr = cl.MackChainladder().fit(incr)
    assert not model_incr.X_.is_cumulative
    pairs = [
        (model_cum, model_incr),
        (model_cum.predict(cum), model_cum.predict(incr)),
    ]
    for a, b in pairs:
        for attr in attrs:
            assert np.allclose(
                getattr(a, attr).set_backend("numpy").values,
                getattr(b, attr).set_backend("numpy").values,
                equal_nan=True,
            ), attr
        assert np.allclose(a.total_mack_std_err_.values, b.total_mack_std_err_.values)
    assert np.allclose(
        model_cum.summary_.set_backend("numpy").values,
        model_incr.summary_.set_backend("numpy").values,
        equal_nan=True,
    )


def test_mack_incremental_matches_cumulative_raa(raa):
    _assert_incremental_matches_cumulative(raa)


def test_mack_incremental_matches_cumulative_mw2008(mw2008):
    _assert_incremental_matches_cumulative(mw2008)


def test_mack_incremental_matches_cumulative_genins(genins):
    _assert_incremental_matches_cumulative(genins)


def test_mack1994_hardcode(raa):
    """
    Reconciles key MackChainladder statistics to values provided in the paper
    """
    # sourced from top table on p130 of Mack(1994)
    sigma_sq = [27883, 1109, 691, 61.2, 119, 40.8, 1.34, 7.88]
    # sourced from bottom table on p130 of Mack(1994)
    ibnr_se = [206, 623, 747, 1469, 2002, 2209, 5358, 6333, 24566]

    dev = cl.Development(sigma_interpolation="mack").fit_transform(raa)
    model = cl.MackChainladder().fit(dev)
    sigma_rhs = dev.sigma_.values.flatten() ** 2
    assert np.allclose(sigma_sq[:3], sigma_rhs[:3], atol=1)
    assert np.allclose(sigma_sq[3:4], sigma_rhs[3:4], atol=0.1)
    assert np.allclose(sigma_sq[4:5], sigma_rhs[4:5], atol=1)
    assert np.allclose(sigma_sq[5:6], sigma_rhs[5:6], atol=0.1)
    assert np.allclose(sigma_sq[6:8], sigma_rhs[6:8], atol=0.01)
    ibnr_rhs = model.summary_.values[0, 0, :, -1]
    assert np.allclose(ibnr_se, ibnr_rhs[1:], atol=1)


# merz2008, Table 2 (p. 560): f_j^I and sigma_j^2 for j = 0 to 7.
TABLE2_LDF = [1.4759, 1.0719, 1.0232, 1.0161, 1.0063, 1.0056, 1.0013, 1.0011]
TABLE2_SIGMA2 = [911.43, 189.82, 97.81, 178.75, 20.64, 3.23, 0.36, 0.04]

# merz2008, Table 4 (p. 562), accident years 1 to 8 (the sample's origins 2002
# to 2009): the one-year column, msep^(1/2) of the observable CDR around 0, and
# Mack's column. Then the cov^(1/2) row and the totals of the two columns.
TABLE4_CDR = [567, 1488, 3923, 9723, 28443, 20954, 28119, 53320]
TABLE4_MACK = [567, 1566, 4157, 10536, 30319, 35967, 45090, 69552]
TABLE4_COV = [39746, 50361]
TABLE4_TOTAL = [81080, 108401]

# fmt: off
# merz2014, Table 1 (p. 28): f_j and sigma_j^2 for j = 0 to 15, the last
# sigma_j^2 by Mack's extrapolation.
MW2014_LDF = [
    1.51105, 1.05369, 1.02681, 1.01709, 1.01284, 1.01098, 1.00292, 1.01098,
    1.00698, 1.00141, 1.00574, 1.00386, 1.00359, 1.00042, 1.00035, 1.00004,
]
MW2014_SIGMA2 = [
    29.4993, 16.8327, 2.6478, 5.4444, 3.2477, 11.4238, 0.1882, 2.6132,
    2.0863, 0.0542, 0.9079, 0.5397, 0.7202, 0.0079, 0.0002, 0.0000,
]

# merz2014, Table 2 (p. 30), accident years 2 to 17 (the sample's origins
# 2002 to 2017; accident year 1 prints 0 throughout): the reserve, then the
# square roots of the process variance, the parameter uncertainty and Mack's
# msep.
MW2014_TABLE2 = {
    "ibnr_": [
        1, 10, 21, 118, 223, 362, 469, 654,
        1009, 1012, 1407, 1493, 1918, 2458, 3384, 9597,
    ],
    "process_risk_": [
        0, 2, 15, 140, 186, 238, 262, 358,
        464, 452, 760, 756, 851, 877, 1067, 1256,
    ],
    "parameter_risk_": [
        0, 2, 9, 72, 92, 109, 129, 156,
        192, 179, 271, 242, 265, 265, 293, 319,
    ],
    "mack_std_err_": [
        0, 3, 17, 157, 207, 262, 292, 391,
        502, 486, 807, 794, 892, 916, 1106, 1296,
    ],
}
# fmt: on
# merz2014, Table 2 (p. 30): the totals of the same four columns.
MW2014_TABLE2_TOTAL = [24135, 2467, 2090, 3233.7]

# merz2014, Table 3 (p. 31), row t = 17: the square root of the one-year
# uncertainty in total.
MW2014_CDR_TOTAL = 1843

CDR_UNSUPPORTED = {
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
    """Values of a per-origin Triangle as an array, origin on the last axis."""
    return triangle.set_backend("numpy").values[..., -1]


def assert_within(actual, expected, tol):
    """Assert that every entry of actual is less than tol from expected."""
    gap = np.abs(np.asarray(actual) - expected)
    np.testing.assert_array_less(gap, np.broadcast_to(tol, gap.shape))


def test_cdr_matches_merz2008(mw2008):
    """
    Four copies of mw2008, two index rows by two columns, each give the
    figures of merz2008.

    Table 2 prints the factors to 4 decimals and sigma^2 to 2. The factors
    round to the printed ones, and the printed sigma^2 are within 0.015 of
    the estimates. Table 4 prints whole units of $1,000. Every entry is within
    1 except accident year 2, printed as 1,488 and 1,566 against estimates of
    1,486.6 and 1,563.8, and checked to 2.5. The cov^(1/2) row is the square
    root of the total's square less the sum of the per-origin squares.
    Accident year 1 has one development period left, so its one-year and Mack
    errors are equal (Table 4 prints 567 in both columns).
    """
    copy = mw2008.copy()
    copy.index = pd.DataFrame({mw2008.key_labels[0]: ["Copy"]})
    tri = cl.concat([mw2008, copy], axis=0)
    tri["copy"] = tri["values"]
    dev = cl.Development(sigma_interpolation="mack").fit_transform(tri)
    model = cl.MackChainladder().fit(dev)
    assert model.cdr_std_err_.shape == (2, 2, 9, 1)
    ldf = dev.ldf_.set_backend("numpy").values[..., 0, :]
    sigma2 = dev.sigma_.set_backend("numpy").values[..., 0, :] ** 2
    assert_within(ldf, TABLE2_LDF, 5e-5)
    assert_within(sigma2, TABLE2_SIGMA2, 0.015)

    cdr = by_origin(model.cdr_std_err_)
    mack = by_origin(model.mack_std_err_)
    assert np.isnan(cdr[..., 0]).all() and np.isnan(mack[..., 0]).all()
    tol = np.where(np.arange(1, 9) == 2, 2.5, 1.0)
    assert_within(cdr[..., 1:], TABLE4_CDR, tol)
    assert_within(mack[..., 1:], TABLE4_MACK, tol)
    total = np.stack(
        (model.total_cdr_std_err_.values, model.total_mack_std_err_.values), -1
    )
    assert_within(total, TABLE4_TOTAL, 1)
    squares = np.stack((np.nansum(cdr**2, -1), np.nansum(mack**2, -1)), -1)
    assert_within(np.sqrt(total**2 - squares), TABLE4_COV, 1)
    np.testing.assert_allclose(cdr[..., 1], mack[..., 1], rtol=1e-12)


def test_cdr_matches_merz2014(mw2014):
    """
    mw2014 gives the figures that merz2014 prints for it.

    Every figure is within half a unit in its last printed decimal: the
    factors of Table 1 to 5e-6 and sigma^2 to 5e-5; the reserve and the three
    errors of Table 2 per accident year, and their totals, to 0.5, except the
    Mack total, printed to 1 decimal and checked to 0.05; the one-year total
    of Table 3 to 0.5. Table 3 gives the one-year uncertainty for each
    accounting year t; the test reads its row t = 17, the latest diagonal, as
    total_cdr_std_err_. Table 2 checks process_risk_ and parameter_risk_ as
    the two parts of Mack's msep. Accident year 1 has no reserve: the paper
    prints 0, and chainladder gives 0 or NaN.
    """
    dev = cl.Development(sigma_interpolation="mack").fit_transform(mw2014)
    model = cl.MackChainladder().fit(dev)
    ldf = dev.ldf_.set_backend("numpy").values[..., 0, :]
    sigma2 = dev.sigma_.set_backend("numpy").values[..., 0, :] ** 2
    assert_within(ldf, MW2014_LDF, 5e-6)
    assert_within(sigma2, MW2014_SIGMA2, 5e-5)

    for attr, printed in MW2014_TABLE2.items():
        values = by_origin(getattr(model, attr))
        assert (np.nan_to_num(values[..., 0]) == 0).all()
        assert_within(values[..., 1:], printed, 0.5)
    totals = [
        np.nansum(by_origin(model.ibnr_)),
        by_origin(model.total_process_risk_).item(),
        by_origin(model.total_parameter_risk_).item(),
        model.total_mack_std_err_.values.item(),
    ]
    assert_within(totals, MW2014_TABLE2_TOTAL, [0.5, 0.5, 0.5, 0.05])
    assert_within(model.total_cdr_std_err_.values, MW2014_CDR_TOTAL, 0.5)


def test_cdr_multi_triangle_matches_single(clrd):
    """
    A fit on several triangles gives each the values of its own fit.

    clrd summed by LOB, paid and incurred, is twelve complete triangles. Both
    attributes of the joint fit match a fit on each triangle alone.
    """
    tri = clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss"]]
    model = cl.MackChainladder().fit(tri)
    per = by_origin(model.cdr_std_err_)
    total = model.total_cdr_std_err_.values
    assert np.isfinite(total).all()
    for i in range(len(tri.index)):
        for j in range(len(tri.columns)):
            single = cl.MackChainladder().fit(tri.iloc[i, j])
            np.testing.assert_allclose(
                per[i, j], by_origin(single.cdr_std_err_)[0, 0], rtol=1e-10
            )
            np.testing.assert_allclose(
                total[i, j], single.total_cdr_std_err_.values[0, 0], rtol=1e-10
            )


def test_cdr_incremental_matches_cumulative(mw2008):
    """
    An incremental triangle gives the values of its cumulative form.

    The estimators of merz2008 are defined on cumulative claims, and the CDR
    attributes accumulate an incremental fit before using it. mw2008 fitted
    as cum_to_incr() and as it is gives the same per-origin and total errors.
    """
    cumulative = mack_sigma_fit(mw2008)
    incremental = mack_sigma_fit(mw2008.cum_to_incr())
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


def test_cdr_missing_cell_gives_nan(mw2008):
    """
    A triangle with a missing cell above the latest diagonal gets NaN.

    The estimators of merz2008 use every cell of the upper triangle. Removing
    one cell from mw2008 leaves MackChainladder able to fit, and both CDR
    attributes are NaN.
    """
    tri = mw2008.set_backend("numpy")
    tri.values = tri.values.copy()
    tri.values[0, 0, 3, 1] = np.nan
    model = mack_sigma_fit(tri.set_backend(mw2008.array_backend))
    assert np.isfinite(model.total_mack_std_err_.values).all()
    assert np.isnan(by_origin(model.cdr_std_err_)).all()
    assert np.isnan(model.total_cdr_std_err_.values).all()


@pytest.mark.parametrize("name", list(CDR_UNSUPPORTED))
def test_cdr_unsupported_pattern_raises_on_read(name, mw2008):
    """
    Unsupported development patterns fit, then raise on reading the CDR.

    merz2008 assumes a square triangle, volume-weighted factors from every
    link ratio of the triangle and no tail. MackChainladder still fits each
    pattern here and gives its Mack standard errors; only the two CDR
    attributes raise ValueError.
    """
    transform, message = CDR_UNSUPPORTED[name]
    model = cl.MackChainladder().fit(transform(mw2008))
    assert np.isfinite(model.total_mack_std_err_.values).all()
    with pytest.raises(ValueError, match=message):
        _ = model.cdr_std_err_
    with pytest.raises(ValueError, match=message):
        _ = model.total_cdr_std_err_


def test_reading_cdr_leaves_mack_unchanged(mw2008):
    """
    Reading the CDR attributes leaves the fitted Mack attributes as they were.

    cdr_std_err_ and total_cdr_std_err_ are computed from the fitted model
    each time they are read. After reading both, summary_ and
    total_mack_std_err_ hold the values they had before.
    """
    model = mack_sigma_fit(mw2008)
    summary = model.summary_.set_backend("numpy").values.copy()
    total = model.total_mack_std_err_.values.copy()
    assert np.isfinite(by_origin(model.cdr_std_err_)[..., 1:]).all()
    assert np.isfinite(model.total_cdr_std_err_.values).all()
    np.testing.assert_array_equal(model.summary_.set_backend("numpy").values, summary)
    np.testing.assert_array_equal(model.total_mack_std_err_.values, total)
