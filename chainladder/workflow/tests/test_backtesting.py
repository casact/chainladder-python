import pandas as pd
import pytest
import chainladder as cl


def test_backtest_uses_requested_valuation_periods(raa):
    result = cl.Backtest(valuation_periods=["1987", "1988"]).fit(raa)

    periods = pd.PeriodIndex(result.valuation_periods_, freq="Y").astype(str)
    targets = pd.PeriodIndex(result.summary_["target_valuation"], freq="Y").astype(str)

    assert periods.tolist() == ["1987", "1988"]
    assert targets.tolist() == ["1988", "1989"]
    assert {"actual", "predicted", "error"}.issubset(result.results_.columns)


def test_backtest_accepts_quarterly_valuation_periods():
    quarterly = cl.load_sample("quarterly")

    result = cl.Backtest(valuation_periods="2005Q3").fit(quarterly)

    period = pd.PeriodIndex(result.valuation_periods_, freq="Q").astype(str)
    assert period.tolist() == ["2005Q3"]


def test_backtest_uses_most_recent_periods(raa):
    result = cl.Backtest(n_periods=2).fit(raa)

    periods = pd.PeriodIndex(result.valuation_periods_, freq="Y").astype(str)

    assert periods.tolist() == ["1988", "1989"]
    assert result.summary_["observations"].gt(0).all()


def test_backtest_compares_the_next_observed_diagonal(raa):
    result = cl.Backtest(n_periods=1).fit(raa)
    target = result.summary_.loc[0, "target_valuation"]
    next_diagonal = raa[raa.valuation == target].to_frame(keepdims=True)

    expected_actual = next_diagonal.loc[
        next_diagonal["origin"].isin(result.results_["origin"]), "values"
    ]
    assert result.results_["actual"].sum() == expected_actual.sum()


def test_backtest_accepts_a_pipeline_estimator(raa):
    estimator = cl.Pipeline(
        [("development", cl.Development()), ("model", cl.Chainladder())]
    )

    result = cl.Backtest(estimator=estimator, n_periods=1).fit(raa)

    assert result.summary_["observations"].tolist() == [9]
    assert len(result.models_) == 1


def test_backtest_rejects_period_without_following_valuation(raa):
    with pytest.raises(ValueError, match="following valuation period"):
        cl.Backtest(valuation_periods="1990").fit(raa)
