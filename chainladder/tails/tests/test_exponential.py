from __future__ import annotations

import warnings

import numpy as np

import chainladder as cl
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_fit_period(tail_sample: Triangle) -> None:
    dev = cl.Development(average="simple").fit_transform(tail_sample)
    assert (
        round(
            cl
            .TailCurve(fit_period=(tail_sample.ddims[-7], None), extrap_periods=10)
            .fit(dev)
            .cdf_["paid"]
            .set_backend("numpy", inplace=True)
            .values[0, 0, 0, -2],
            3,
        )
        == 1.044
    )


def test_curve_validation(tail_sample: Triangle) -> None:
    """
    Test validation of the curve parameter. Should raise a value error if an incorrect argument is supplied.
    """

    with pytest.raises(ValueError):
        cl.TailCurve(curve="Exponential").fit_transform(tail_sample)


def test_errors_validation(tail_sample: Triangle) -> None:
    """
    Test validation of the errors parameter. Should raise a value error if an incorrect argument is supplied.
    """
    with pytest.raises(ValueError):
        cl.TailCurve(errors="Ignore").fit_transform(tail_sample)


def test_no_log_warning_when_only_some_tails_exceed_one() -> None:
    """
    A tail at or below 1.0 must not reach ``log(tail - 1)``.

    The nominal 1.001 used to be substituted only when the *largest* tail was
    at or below 1, so an estimator producing a mix of tails skipped the
    substitution entirely and evaluated the logarithm of a negative number.
    ``TailBondy`` on the grouped ``clrd`` sample gives 6 tails below 1 out of
    12, with a maximum of 1.018. See #1414.

    Returns
    -------
    None
    """
    clrd = cl.load_sample("clrd")
    triangle = clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss"]]
    triangle["CaseIncurredLoss"] = triangle["IncurLoss"] - triangle["CumPaidLoss"]
    development = cl.Development().fit_transform(
        triangle[["CumPaidLoss", "CaseIncurredLoss"]]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cl.TailBondy().fit(development)

    offending = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "log" in str(w.message)
        and "tails/base.py" in str(w.filename).replace("\\", "/")
    ]
    assert not offending, [str(w.message) for w in offending]


def test_nominal_tail_option_is_honoured() -> None:
    """
    ``NOMINAL_TAIL`` sets the tail substituted for a tail at or below 1.0.

    The full ``clrd`` sample has one index entry whose tail is exactly 1.0 while
    its regression coefficients are finite, so the substituted value reaches
    ``sigma_``. See #1414.

    Returns
    -------
    None
    """
    development = cl.Development().fit_transform(cl.load_sample("clrd")["CumPaidLoss"])

    assert cl.options.NOMINAL_TAIL == 1.001
    baseline = cl.TailCurve().fit(development).sigma_.values.copy()

    try:
        cl.options.set_option("NOMINAL_TAIL", 1.5)
        widened = cl.TailCurve().fit(development).sigma_.values
    finally:
        cl.options.set_option("NOMINAL_TAIL", 1.001)

    assert not np.allclose(np.nan_to_num(baseline), np.nan_to_num(widened))
    assert cl.options.NOMINAL_TAIL == 1.001
