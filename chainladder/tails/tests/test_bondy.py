from __future__ import annotations

import pytest

import chainladder as cl
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_bondy_rejects_non_finite_development_factor(clrd: Triangle) -> None:
    """
    ``TailBondy`` names the fit age with no finite development factor (#1707).

    Aegis Grp / comauto has a NaN last development factor, so the default
    fit (from the second-to-last age) used to fail inside scipy with
    ``ValueError: Initial guess is outside of provided bounds``.
    """
    tri = clrd["CumPaidLoss"].iloc[2]
    with pytest.raises(
        ValueError, match=r"no finite development factor at age\(s\) 108"
    ):
        cl.TailBondy().fit(tri)


def test_bondy_names_each_non_finite_fit_age(clrd: Triangle) -> None:
    """Every non-finite fit age is listed, not just the first one (#1707)."""
    tri = clrd["CumPaidLoss"].iloc[2]
    with pytest.raises(
        ValueError, match=r"no finite development factor at age\(s\) 84, 96, 108"
    ):
        cl.TailBondy(earliest_age=12).fit(tri)


def test_bondy1(tail_sample: Triangle) -> None:
    tri = tail_sample["paid"]
    dev = cl.Development(average="simple").fit_transform(tri)
    assert (
        round(float(cl.TailBondy(earliest_age=12).fit(dev).cdf_.values[0, 0, 0, -2]), 3)
        == 1.028
    )


def test_bondy_fits_grouped_development_pattern(clrd: Triangle) -> None:
    """
    ``TailBondy`` fits a pattern from ``Development(groupby=...)``.

    The grouped pattern has one row per line while the triangle keeps one row
    per company, so ``TailBondy`` sizes its starting values and labels ``b_``
    from ``ldf_``. The result matches summing by line before the fit. See
    #1599.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    triangle = clrd["CumPaidLoss"]
    grouped = cl.TailBondy().fit(cl.Development(groupby="LOB").fit_transform(triangle))
    summed = cl.TailBondy().fit(
        cl.Development().fit_transform(triangle.groupby("LOB").sum())
    )

    assert grouped.ldf_.shape == summed.ldf_.shape
    assert list(grouped.b_.index) == list(summed.b_.index)
    assert np.allclose(grouped.b_.values, summed.b_.values, equal_nan=True)
    assert np.allclose(grouped.ldf_.values, summed.ldf_.values, equal_nan=True)
