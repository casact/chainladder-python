import pandas as pd
import pytest

import chainladder as cl


def test_discounted_reserve_calculates_present_value(raa):
    model = cl.Chainladder().fit(raa)
    result = cl.DiscountedReserve(
        annual_rates=0.03, valuation_date=raa.valuation_date
    ).fit(model.full_triangle_)

    assert result.cashflows_["payment_date"].gt(raa.valuation_date).all()
    assert result.cashflows_["discount_factor"].lt(1).all()
    assert result.summary_["present_value"].sum() < result.summary_["undiscounted"].sum()


def test_discounted_reserve_accepts_payment_period_curve(raa):
    model = cl.Chainladder().fit(raa)
    projected = model.full_triangle_
    payment_dates = projected.valuation[projected.valuation > raa.valuation_date]
    curve = {
        int(period): 0.03
        for period in pd.PeriodIndex(payment_dates, freq=projected.development_grain)
        .astype(str)
        .unique()
    }

    result = cl.DiscountedReserve(curve, raa.valuation_date).fit(projected)

    assert result.cashflows_["annual_rate"].eq(0.03).all()


def test_risk_adjustment_supports_margin_and_confidence_level():
    reserve = pd.DataFrame({"portfolio": ["A"], "present_value": [100.0], "se": [20.0]})

    margin = cl.RiskAdjustment(margin=0.1).fit(reserve)
    confidence = cl.RiskAdjustment(
        method="confidence_level", confidence_level=0.75, standard_error="se"
    ).fit(reserve)

    assert margin.summary_.loc[0, "risk_adjustment"] == 10.0
    assert margin.summary_.loc[0, "reserve_including_risk_adjustment"] == 110.0
    assert confidence.summary_.loc[0, "risk_adjustment"] > 0


def test_risk_adjustment_requires_standard_error():
    reserve = pd.DataFrame({"present_value": [100.0]})

    with pytest.raises(ValueError, match="standard_error"):
        cl.RiskAdjustment(method="confidence_level").fit(reserve)


def test_discounted_reserve_rejects_non_triangle_input():
    with pytest.raises(TypeError, match="Triangle"):
        cl.DiscountedReserve(0.03, "2024-12-31").fit(pd.DataFrame())
