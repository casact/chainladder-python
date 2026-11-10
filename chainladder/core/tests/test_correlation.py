from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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


def test_dev_corr_matches_mack(raa: Triangle) -> None:
    # Appendix G of Mack, "Measuring the Variability of Chain Ladder Reserve
    # Estimates" (CAS Forum, Spring 1994), applies the test to this triangle,
    # the example of the paper's Chapter 6. Spearman's coefficients
    # T_k = 4/21, -9/28, 3/7, -1/5, 2/5, -1/2 and 1 for k = 2, ..., 8, weighted
    # by I - k - 1 = 7, ..., 1, average to T = .070 with Var(T) = 1/28. T lies
    # inside the 50% limits +/- .67 / sqrt(28) = +/- .127, so the test does
    # not reject. The library uses norm.ppf(0.75) = 0.6745 for Mack's .67, and
    # its limits also round to +/- .127.
    dc = raa.development_correlation(p_critical=0.5)
    t_k = [4 / 21, -9 / 28, 3 / 7, -1 / 5, 2 / 5, -1 / 2, 1]
    assert dc.t.iloc[0].tolist() == pytest.approx(t_k)
    assert dc.weights.iloc[0].tolist() == [7, 6, 5, 4, 3, 2, 1]
    assert dc.t_expectation.iloc[0, 0] == pytest.approx(0.070, abs=5e-4)
    assert dc.t_variance.iloc[0, 0] == 1 / 28
    assert dc.confidence_interval[0].iloc[0, 0] == pytest.approx(-0.127, abs=5e-4)
    assert dc.confidence_interval[1].iloc[0, 0] == pytest.approx(0.127, abs=5e-4)
    assert not dc.t_critical.iloc[0, 0]


def test_dev_corr_complete_triangle_counts(genins: Triangle) -> None:
    """
    On a complete triangle the pair counts are Mack's I - k.

    genins has ten origins and ten development periods, so the weights are
    I - k - 1 = 7, ..., 1 and Var(T) = 2 / ((I - 2)(I - 3)) = 1/28, as in
    formula G6 of Mack (1994).
    """
    dc = genins.development_correlation(p_critical=0.5)
    assert dc.weights.iloc[0].tolist() == [7, 6, 5, 4, 3, 2, 1]
    assert dc.t_variance.iloc[0, 0] == 1 / 28


def test_dev_corr_no_data(raa: Triangle) -> None:
    """
    A triangle without link ratios gets NaN and is not flagged.

    GH #1577: with no pairs to rank there is no T_k to average, so T and its
    variance are NaN, and t_critical is False, as ValuationCorrelation does
    for empty diagonals.
    """
    dc = (raa * np.nan).development_correlation(p_critical=0.5)
    assert np.isnan(dc.t_expectation.iloc[0, 0])
    assert np.isnan(dc.t_variance.iloc[0, 0])
    assert not dc.t_critical.iloc[0, 0]


def test_dev_corr_empty_member(auto: Triangle) -> None:
    """
    An empty triangle among several leaves the others unchanged.

    GH #1577: the auto sample has no incurred losses for CommercialAuto. That
    triangle gets NaN and is not flagged, and each of the other three gets
    the result it gets on its own.
    """
    dc = auto.development_correlation(p_critical=0.5)
    assert np.isnan(dc.t_expectation.loc["CommercialAuto", "incurred"])
    assert not dc.t_critical.loc["CommercialAuto", "incurred"]
    others = [
        ("CommercialAuto", "paid"),
        ("PersonalAuto", "incurred"),
        ("PersonalAuto", "paid"),
    ]
    for lob, column in others:
        alone = auto.loc[lob][column].development_correlation(p_critical=0.5)
        expected = alone.t_expectation.iloc[0, 0]
        assert dc.t_expectation.loc[lob, column] == pytest.approx(expected)
        assert dc.t_variance.loc[lob, column] == alone.t_variance.iloc[0, 0]


def test_dev_corr_counts_origins_with_data(raa: Triangle) -> None:
    """
    I is the number of origins with data, not of development columns.

    GH #1577: the last six origins of raa, kept on the full ten-column grid,
    give the same T and variance as the same data cut to a 6 x 6 triangle,
    where Var(T) = 2 / ((6 - 2)(6 - 3)) = 1/6.
    """
    recent = raa[raa.origin >= "1985"]
    square = recent[recent.development <= 72]
    on_grid = recent.development_correlation(p_critical=0.5)
    cut = square.development_correlation(p_critical=0.5)
    expected = cut.t_expectation.iloc[0, 0]
    assert on_grid.t_expectation.iloc[0, 0] == pytest.approx(expected)
    assert on_grid.t_variance.iloc[0, 0] == pytest.approx(1 / 6)
    assert cut.t_variance.iloc[0, 0] == pytest.approx(1 / 6)


@pytest.mark.parametrize("size", [2, 3])
def test_dev_corr_too_small(raa: Triangle, size: int) -> None:
    """
    Triangles with fewer than four origins give NaN instead of raising.

    GH #1577: T_k needs at least two pairs, which a triangle first has at
    I = 4. A 3 x 3 triangle raised AttributeError and a 2 x 2 triangle
    IndexError.
    """
    small = raa[raa.origin >= str(1991 - size)]
    small = small[small.development <= 12 * size]
    dc = small.development_correlation(p_critical=0.5)
    assert np.isnan(dc.t_expectation.iloc[0, 0])
    assert not dc.t_critical.iloc[0, 0]


def test_dev_corr_smallest_triangle(raa: Triangle) -> None:
    """
    A 4 x 4 triangle has one T_k, from two pairs, with Var(T) = 1.

    Formula G6 gives 2 / ((4 - 2)(4 - 3)) = 1, and k = 2 is the only column
    with two pairs.
    """
    small = raa[raa.origin >= "1987"]
    small = small[small.development <= 48]
    dc = small.development_correlation(p_critical=0.5)
    assert dc.weights.iloc[0].tolist() == [1]
    assert dc.t_variance.iloc[0, 0] == 1


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
