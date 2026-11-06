from __future__ import annotations

import chainladder as cl
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_val_corr_total_true(raa: Triangle) -> None:
    assert raa.valuation_correlation(p_critical=0.5, total=True)


def test_val_corr_total_matches_mack(raa: Triangle) -> None:
    # Mack (1994), "Measuring the Variability of Chain Ladder Reserve
    # Estimates", works the test on RAA: Z = 14, E(Z) = 12.875 and
    # Var(Z) = 3.9785. R's ChainLadder cyEffTest(RAA) at ci = 0.95 prints the
    # band (8.965613, 16.784387), which is E(Z) +/- 1.96 * sqrt(Var(Z)).
    vc = raa.valuation_correlation(p_critical=0.05, total=True)
    assert vc.z.iloc[0, 0] == 14
    assert vc.z_expectation.iloc[0, 0] == pytest.approx(12.875)
    assert vc.z_variance.iloc[0, 0] == pytest.approx(3.978516, abs=1e-6)
    assert vc.range[0].item() == pytest.approx(8.965613, abs=1e-6)
    assert vc.range[1].item() == pytest.approx(16.784387, abs=1e-6)
    assert not vc.z_critical.iloc[0, 0]


def test_val_corr_total_flags_calendar_effect() -> None:
    # On the abc sample Z = 8, far below E(Z) = 16.30, so at
    # p_critical = 0.1 (band about 12.88 to 19.72) the test rejects.
    abc = cl.load_sample("abc")
    vc = abc.valuation_correlation(p_critical=0.1, total=True)
    assert vc.z.iloc[0, 0] == 8
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
