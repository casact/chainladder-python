import pytest

import chainladder as cl


def test_sensitivity_analysis_compares_scenarios(raa):
    estimator = cl.Pipeline(
        [("development", cl.Development(average="simple")), ("model", cl.Chainladder())]
    )
    result = cl.SensitivityAnalysis(
        estimator,
        scenarios={"base": {}, "volume": {"development__average": "volume"}},
    ).fit(raa)

    assert result.base_scenario_ == "base"
    assert result.results_["scenario"].tolist() == ["base", "volume"]
    assert result.results_.loc[0, "ibnr_change"] == 0
    assert result.results_["ibnr"].notna().all()


def test_sensitivity_analysis_validates_scenarios(raa):
    with pytest.raises(ValueError, match="scenarios"):
        cl.SensitivityAnalysis(cl.Chainladder(), {}).fit(raa)
