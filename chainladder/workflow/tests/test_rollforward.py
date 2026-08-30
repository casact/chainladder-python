import pandas as pd
import pytest

import chainladder as cl


def test_reserve_rollforward_compares_expected_and_actual_payments(raa):
    result = cl.ReserveRollforward().fit(raa)

    summary = result.summary_.iloc[0]
    assert summary["prior_valuation"] < summary["current_valuation"]
    assert summary["actual_payment"] > 0
    assert {"expected_payment", "actual_payment", "payment_variance"}.issubset(
        result.detail_
    )


def test_reserve_rollforward_accepts_a_prior_valuation_period(raa):
    result = cl.ReserveRollforward(prior_valuation_period="1988").fit(raa)

    period = pd.Period(result.prior_valuation_, freq="Y")
    assert str(period) == "1988"


def test_reserve_rollforward_rejects_current_valuation_as_prior(raa):
    with pytest.raises(ValueError, match="before the current valuation"):
        cl.ReserveRollforward(prior_valuation_period="1990").fit(raa)


def test_reserve_rollforward_rejects_non_triangle_input():
    with pytest.raises(TypeError, match="Triangle"):
        cl.ReserveRollforward().fit(pd.DataFrame())
