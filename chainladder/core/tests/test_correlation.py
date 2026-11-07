from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scipy.stats import norm

import chainladder as cl

if TYPE_CHECKING:
    from chainladder import Triangle


def test_val_corr_total_true(raa: Triangle) -> None:
    assert raa.valuation_correlation(p_critical=0.5, total=True)


def test_val_corr_total_matches_mack(raa: Triangle) -> None:
    # Appendix H of Mack, "Measuring the Variability of Chain Ladder Reserve
    # Estimates" (CAS Forum, Spring 1994), applies the test to this triangle,
    # the example of the paper's Chapter 6: Z = 14, E(Z) = 12.875 and
    # Var(Z) = 3.9785, inside the 95% range E(Z) +/- 2 * sqrt(Var(Z)) =
    # (8.886, 16.864), so the test does not reject. A band of two standard
    # deviations corresponds to p_critical = 2 * norm.sf(2).
    vc = raa.valuation_correlation(p_critical=2 * norm.sf(2), total=True)
    assert vc.z.iloc[0, 0] == 14
    assert vc.z_expectation.iloc[0, 0] == pytest.approx(12.875)
    assert vc.z_variance.iloc[0, 0] == pytest.approx(3.9785, abs=5e-5)
    assert vc.range[0].item() == pytest.approx(8.886, abs=5e-4)
    assert vc.range[1].item() == pytest.approx(16.864, abs=5e-4)
    assert not vc.z_critical.iloc[0, 0]


def test_val_corr_total_flags_calendar_effect() -> None:
    # Question 4 of the CAS Spring 2015 Exam 7 gives these paid age-to-age
    # factors. Sample Answer 1, published with the examiners' report, finds
    # Z = 2, E(Z) = 4.625 and Var(Z) = 1.117, outside the 95% range
    # (2.554, 6.696), so it rejects the hypothesis of no calendar year
    # effects: https://www.casact.org/sites/default/files/2021-03/15-7.pdf
    # Only the factors matter to the test, so each accident year starts at 1.
    factors = [
        [7.00, 2.55, 1.60, 1.30, 1.20, 1.05],
        [6.00, 2.60, 1.40, 1.50, 1.10],
        [5.00, 2.50, 1.80, 1.20],
        [5.50, 2.80, 1.50],
        [8.00, 2.40],
        [7.00],
    ]
    origin, valuation, paid = [], [], []
    for i, row in enumerate(factors):
        amount = 1.0
        for k, factor in enumerate([1.0, *row]):
            amount *= factor
            origin.append(2008 + i)
            valuation.append(2008 + i + k)
            paid.append(amount)
    triangle = cl.Triangle(
        data={"origin": origin, "valuation": valuation, "paid": paid},
        origin="origin",
        valuation="valuation",
        columns=["paid"],
        cumulative=True,
    )
    vc = triangle.valuation_correlation(p_critical=0.05, total=True)
    assert vc.z.iloc[0, 0] == 2
    assert vc.z_expectation.iloc[0, 0] == pytest.approx(4.625)
    assert vc.z_variance.iloc[0, 0] == pytest.approx(1.117, abs=5e-4)
    assert vc.range[0].item() == pytest.approx(2.554, abs=1e-3)
    assert vc.range[1].item() == pytest.approx(6.696, abs=1e-3)
    assert vc.z_critical.iloc[0, 0]


def test_val_corr_total_false(raa: Triangle) -> None:
    assert raa.valuation_correlation(p_critical=0.5, total=False)


def test_dev_corr(raa: Triangle) -> None:
    assert raa.development_correlation(p_critical=0.5)


def test_validate_critical(raa: Triangle) -> None:
    with pytest.raises(ValueError):
        raa.valuation_correlation(p_critical=1.5, total=True)


def test_val_corr_incomplete_triangle(xyz: Triangle) -> None:
    # GH #320: a triangle missing its earliest diagonals raised
    # "Shape of passed values is (1, 10), indices imply (1, 9)" on repr,
    # because z_critical dropped all-NaN diagonals while its values kept
    # one entry per link-ratio diagonal.
    z_critical = (
        xyz["Paid"].valuation_correlation(p_critical=0.1, total=False).z_critical
    )
    assert z_critical.values.shape[-1] == len(z_critical.ddims)
    assert repr(z_critical)
