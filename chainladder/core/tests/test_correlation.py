from __future__ import annotations

import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_val_corr_total_true(raa: Triangle) -> None:
    assert raa.valuation_correlation(p_critical=0.5, total=True)


def test_val_corr_total_false(raa: Triangle) -> None:
    assert raa.valuation_correlation(p_critical=0.5, total=False)


def test_dev_corr(raa: Triangle) -> None:
    assert raa.development_correlation(p_critical=0.5)


def test_validate_critical(raa: Triangle) -> None:
    with pytest.raises(ValueError):
        raa.valuation_correlation(p_critical=1.5, total=True)
