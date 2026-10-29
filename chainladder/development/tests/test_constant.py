import numpy as np
import pandas as pd
import pytest

import chainladder as cl


def test_constant_cdf(raa):
    dev = cl.Development().fit(raa)
    xp = dev.ldf_.get_array_module()
    link_ratios = {
        (num + 1) * 12: item for num, item in enumerate(dev.ldf_.values[0, 0, 0, :])
    }
    dev_c = cl.DevelopmentConstant(patterns=link_ratios, style="ldf").fit(raa)
    assert xp.allclose(dev.cdf_.values, dev_c.cdf_.values, atol=1e-5)


def test_constant_ldf(raa):
    dev = cl.Development().fit(raa)
    xp = dev.ldf_.get_array_module()
    link_ratios = {
        (num + 1) * 12: item for num, item in enumerate(dev.ldf_.values[0, 0, 0, :])
    }
    dev_c = cl.DevelopmentConstant(patterns=link_ratios, style="ldf").fit(raa)
    assert xp.allclose(dev.ldf_.values, dev_c.ldf_.values, atol=1e-5)


def test_constant_callable_axis0(clrd, atol):
    agway = clrd.loc["Agway Ins Co", "CumPaidLoss"]

    def paid_cdfs(x):
        """A function that returns different CDFs depending on a specified LOB"""
        cdfs = {
            "comauto": [
                3.832,
                1.874,
                1.386,
                1.181,
                1.085,
                1.043,
                1.022,
                1.013,
                1.007,
                1,
            ],
            "medmal": [
                24.168,
                4.127,
                2.103,
                1.528,
                1.275,
                1.161,
                1.088,
                1.047,
                1.018,
                1,
            ],
            "othliab": [
                10.887,
                3.416,
                1.957,
                1.433,
                1.231,
                1.119,
                1.06,
                1.031,
                1.011,
                1,
            ],
            "ppauto": [2.559, 1.417, 1.181, 1.084, 1.04, 1.019, 1.009, 1.004, 1.001, 1],
            "prodliab": [
                13.703,
                5.613,
                2.92,
                1.765,
                1.385,
                1.177,
                1.072,
                1.034,
                1.008,
                1,
            ],
            "wkcomp": [4.106, 1.865, 1.418, 1.234, 1.141, 1.09, 1.056, 1.03, 1.01, 1],
        }
        patterns = pd.DataFrame(cdfs, index=range(12, 132, 12)).T
        return patterns.loc[x.loc["LOB"]].to_dict()

    model = cl.DevelopmentConstant(patterns=paid_cdfs, callable_axis=0, style="cdf")
    assert (
        abs(model.fit_transform(agway).cdf_.loc["comauto"].iloc[..., 0].sum() - 3.832)
        < atol
    )


def test_constant_callable_axis1(clrd, atol):
    agway = clrd.loc["Agway Ins Co", "comauto"]
    cdfs = {
        "IncurLoss": [3.832, 1.874, 1.386, 1.181, 1.085, 1.043, 1.022, 1.013, 1.007, 1],
        "CumPaidLoss": [
            24.168,
            4.127,
            2.103,
            1.528,
            1.275,
            1.161,
            1.088,
            1.047,
            1.018,
            1,
        ],
        "BulkLoss": [10.887, 3.416, 1.957, 1.433, 1.231, 1.119, 1.06, 1.031, 1.011, 1],
        "EarnedPremDIR": [
            2.559,
            1.417,
            1.181,
            1.084,
            1.04,
            1.019,
            1.009,
            1.004,
            1.001,
            1,
        ],
        "EarnedPremCeded": [
            13.703,
            5.613,
            2.92,
            1.765,
            1.385,
            1.177,
            1.072,
            1.034,
            1.008,
            1,
        ],
        "EarnedPremNet": [
            4.106,
            1.865,
            1.418,
            1.234,
            1.141,
            1.09,
            1.056,
            1.03,
            1.01,
            1,
        ],
    }
    patterns = pd.DataFrame(cdfs, index=range(12, 132, 12)).T

    def paid_cdfs(x):
        """A function that returns different CDFs depending on a specified column"""
        return patterns.loc[x.loc["columns"]].to_dict()

    with pytest.raises(ValueError):
        cl.DevelopmentConstant(patterns=paid_cdfs, callable_axis=2, style="cdf").fit(
            agway
        )
    lhs = (
        cl
        .DevelopmentConstant(patterns=paid_cdfs, callable_axis=1, style="cdf")
        .fit(agway)
        .cdf_
    )
    assert np.all(abs(lhs.values[0, :, 0, :] - patterns.values) < atol)


def test_constant_pattern_no_tail():
    reported_patterns = {
        12: 4.0,
        24: 2.9,
        36: 1.8,
        48: 1.4,
        60: 1.2,
        72: 1.1,
        84: 1.03,
        96: 1.02,
        # 108: 1.005,
    }
    auto_bi = cl.load_sample("friedland_auto_bi_insurer")
    reported_BI_claim = cl.DevelopmentConstant(
        patterns=reported_patterns, style="cdf"
    ).fit_transform(auto_bi["Reported Claims"])

    assert np.all(
        np.round(reported_BI_claim.cdf_.to_frame().values.flatten(), 6)
        == np.array([4.0, 2.9, 1.8, 1.4, 1.2, 1.1, 1.03, 1.02])
    )
    assert np.all(
        np.round(reported_BI_claim.ldf_.to_frame().values.flatten(), 6)
        == np.round(
            np.array([
                4.0 / 2.9,
                2.9 / 1.8,
                1.8 / 1.4,
                1.4 / 1.2,
                1.2 / 1.1,
                1.1 / 1.03,
                1.03 / 1.02,
                1.02,
            ]),
            6,
        )
    )


def test_constant_pattern_has_tail():
    reported_patterns = {
        12: 4.0,
        24: 2.9,
        36: 1.8,
        48: 1.4,
        60: 1.2,
        72: 1.1,
        84: 1.03,
        96: 1.02,
        108: 1.005,
    }
    auto_bi = cl.load_sample("friedland_auto_bi_insurer")
    reported_BI_claim = cl.DevelopmentConstant(
        patterns=reported_patterns, style="cdf"
    ).fit_transform(auto_bi["Reported Claims"])

    assert np.all(
        np.round(reported_BI_claim.cdf_.to_frame().values.flatten(), 6)
        == np.array([4.0, 2.9, 1.8, 1.4, 1.2, 1.1, 1.03, 1.02, 1.005])
    )
    assert np.all(
        np.round(reported_BI_claim.ldf_.to_frame().values.flatten(), 6)
        == np.round(
            np.array([
                4.0 / 2.9,
                2.9 / 1.8,
                1.8 / 1.4,
                1.4 / 1.2,
                1.2 / 1.1,
                1.1 / 1.03,
                1.03 / 1.02,
                1.02 / 1.005,
                1.005,
            ]),
            6,
        )
    )


def test_constant_pattern_exact_cdf(raa):
    reported_patterns = {
        12: 1.1,
        24: 1.1,
        36: 1.1,
        48: 1.1,
        60: 1.1,
        72: 1.1,
        84: 1.1,
        96: 1.1,
        108: 1.1,
        120: 1.1,
    }

    result = cl.DevelopmentConstant(
        patterns=reported_patterns, style="cdf"
    ).fit_transform(raa)

    assert np.all(
        result.cdf_.to_frame().values.flatten()
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    )
    assert np.all(
        result.ldf_.to_frame().values.flatten()
        == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1])
    )


def test_constant_pattern_exact_ldf(raa):
    reported_patterns = {
        12: 1.1,
        24: 1.1,
        36: 1.1,
        48: 1.1,
        60: 1.1,
        72: 1.1,
        84: 1.1,
        96: 1.1,
        108: 1.1,
        120: 1.1,
    }

    result = cl.DevelopmentConstant(
        patterns=reported_patterns, style="ldf"
    ).fit_transform(raa)

    assert np.all(
        np.round(result.cdf_.to_frame().values.flatten(), 6)
        == np.round(
            np.array([
                1.1**10,
                1.1**9,
                1.1**8,
                1.1**7,
                1.1**6,
                1.1**5,
                1.1**4,
                1.1**3,
                1.1**2,
                1.1,
            ]),
            6,
        )
    )
    assert np.all(
        result.ldf_.to_frame().values.flatten()
        == np.array([
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
            1.1,
        ])
    )


def test_constant_pattern_short_cdf(raa):
    reported_patterns = {
        12: 1.1,
        24: 1.1,
        36: 1.1,
        48: 1.1,
        60: 1.1,
        72: 1.1,
        # 84: 1.1,
        # 96: 1.1,
        # 108: 1.1,
        # 120: 1.1,
    }

    result = cl.DevelopmentConstant(
        patterns=reported_patterns, style="cdf"
    ).fit_transform(raa)

    assert np.all(
        result.cdf_.to_frame().values.flatten()
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0])
    )
    assert np.all(
        result.ldf_.to_frame().values.flatten()
        == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0])
    )


def test_constant_pattern_short_ldf(raa):
    reported_patterns = {
        12: 1.1,
        24: 1.1,
        36: 1.1,
        48: 1.1,
        60: 1.1,
        72: 1.1,
        # 84: 1.1,
        # 96: 1.1,
        # 108: 1.1,
        # 120: 1.1,
    }

    result = cl.DevelopmentConstant(
        patterns=reported_patterns, style="ldf"
    ).fit_transform(raa)

    assert np.all(
        np.round(result.cdf_.to_frame().values.flatten(), 6)
        == np.round(
            np.array([1.1**6, 1.1**5, 1.1**4, 1.1**3, 1.1**2, 1.1, 1.0, 1.0, 1.0]), 6
        )
    )
    assert np.all(
        result.ldf_.to_frame().values.flatten()
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0])
    )


def test_constant_pattern_long_cdf(raa):
    reported_patterns = {
        12: 1.1,
        24: 1.1,
        36: 1.1,
        48: 1.1,
        60: 1.1,
        72: 1.1,
        84: 1.1,
        96: 1.1,
        108: 1.1,
        120: 1.1,
        132: 1.1,
    }

    result = cl.DevelopmentConstant(
        patterns=reported_patterns, style="cdf"
    ).fit_transform(raa)
    assert np.all(
        np.round(result.cdf_.to_frame().values.flatten(), 6)
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    )
    assert np.all(
        np.round(result.ldf_.to_frame().values.flatten(), 6)
        == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1])
    )


def test_constant_pattern_long_ldf(raa):
    reported_patterns = {
        12: 1.1,
        24: 1.1,
        36: 1.1,
        48: 1.1,
        60: 1.1,
        72: 1.1,
        84: 1.1,
        96: 1.1,
        108: 1.1,
        120: 1.1,
        132: 1.1,
    }

    result = cl.DevelopmentConstant(
        patterns=reported_patterns, style="ldf"
    ).fit_transform(raa)

    assert np.all(
        np.round(result.cdf_.to_frame().values.flatten(), 6)
        == np.round(
            np.array([
                1.1**11,
                1.1**10,
                1.1**9,
                1.1**8,
                1.1**7,
                1.1**6,
                1.1**5,
                1.1**4,
                1.1**3,
                1.1**2,
                1.1,
            ]),
            6,
        )
    )
    assert np.all(
        result.ldf_.to_frame().values.flatten()
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
    )


def test_constant_incr():
    raa_incr = cl.load_sample("raa").cum_to_incr()
    reported_patterns = {
        12: 4.0,
        24: 2.9,
        36: 1.8,
        48: 1.4,
        60: 1.2,
        72: 1.1,
        84: 1.03,
        96: 1.02,
        108: 1.005,
    }

    result = cl.DevelopmentConstant(
        patterns=reported_patterns, style="cdf"
    ).fit_transform(raa_incr)

    assert np.all(
        np.round(result.cdf_.to_frame().values.flatten(), 6)
        == np.array([4.0, 2.9, 1.8, 1.4, 1.2, 1.1, 1.03, 1.02, 1.005])
    )


def test_constant_series_pattern(raa):
    dev = cl.Development().fit(raa)
    # Series with integer index
    s_int = pd.Series(dev.ldf_.values[0, 0, 0, :], index=dev.ldf_.ddims)
    model_int = cl.DevelopmentConstant(patterns=s_int, style="ldf").fit(raa)
    assert np.allclose(dev.ldf_.values, model_int.ldf_.values)

    # Series with interval string index (e.g. '12-24')
    s_str = pd.Series(
        dev.ldf_.values[0, 0, 0, :], index=[f"{d}-{d + 12}" for d in dev.ldf_.ddims]
    )
    model_str = cl.DevelopmentConstant(patterns=s_str, style="ldf").fit(raa)
    assert np.allclose(dev.ldf_.values, model_str.ldf_.values)

    # Series with non-numeric index labels (fallback to original keys)
    s_non_numeric = pd.Series([1.5, 1.2], index=["dev_12", "dev_24"])
    normalized = cl.DevelopmentConstant._normalize_patterns(s_non_numeric)
    assert normalized == {"dev_12": 1.5, "dev_24": 1.2}


def test_constant_dataframe_pattern(raa):
    dev = cl.Development().fit(raa)
    # 1-row DataFrame
    df_row = pd.DataFrame([dev.ldf_.values[0, 0, 0, :]], columns=dev.ldf_.ddims)
    model_row = cl.DevelopmentConstant(patterns=df_row, style="ldf").fit(raa)
    assert np.allclose(dev.ldf_.values, model_row.ldf_.values)

    # 1-column DataFrame
    df_col = pd.DataFrame(dev.ldf_.values[0, 0, 0, :], index=dev.ldf_.ddims)
    model_col = cl.DevelopmentConstant(patterns=df_col, style="ldf").fit(raa)
    assert np.allclose(dev.ldf_.values, model_col.ldf_.values)


def test_constant_triangle_pattern(raa):
    dev = cl.Development().fit(raa)
    # Triangle pattern directly from dev.ldf_
    model_ldf = cl.DevelopmentConstant(patterns=dev.ldf_, style="ldf").fit(raa)
    assert np.allclose(dev.ldf_.values, model_ldf.ldf_.values)

    # Triangle pattern directly from dev.cdf_
    model_cdf = cl.DevelopmentConstant(patterns=dev.cdf_, style="cdf").fit(raa)
    assert np.allclose(dev.cdf_.values, model_cdf.cdf_.values)


def test_constant_estimator_pattern(raa):
    dev = cl.Development().fit(raa)
    # Fitted Development estimator with style='ldf'
    model_est_ldf = cl.DevelopmentConstant(patterns=dev, style="ldf").fit(raa)
    assert np.allclose(dev.ldf_.values, model_est_ldf.ldf_.values)

    # Fitted Development estimator with style='cdf'
    model_est_cdf = cl.DevelopmentConstant(patterns=dev, style="cdf").fit(raa)
    assert np.allclose(dev.cdf_.values, model_est_cdf.cdf_.values)

    # Object with only cdf_ (no ldf_) with default/ldf style fallback
    class OnlyCDF:
        def __init__(self, cdf):
            self.cdf_ = cdf

    with pytest.warns(UserWarning, match="Estimator provides only 'cdf_'"):
        model_only_cdf = cl.DevelopmentConstant(patterns=OnlyCDF(dev.cdf_)).fit(raa)
    assert np.allclose(dev.cdf_.values, model_only_cdf.ldf_.values)

    # Object with only ldf_ (no cdf_) with cdf style fallback
    class OnlyLDF:
        def __init__(self, ldf):
            self.ldf_ = ldf

    with pytest.warns(UserWarning, match="Estimator provides only 'ldf_'"):
        model_only_ldf = cl.DevelopmentConstant(
            patterns=OnlyLDF(dev.ldf_), style="cdf"
        ).fit(raa)
    assert np.allclose(dev.ldf_.values, model_only_ldf.cdf_.values)


def test_constant_invalid_pattern_types(raa):
    # Invalid DataFrame shape (>1 row and >1 col)
    df_invalid = pd.DataFrame([[1.2, 1.1], [1.0, 1.0]], columns=[12, 24])
    with pytest.raises(ValueError, match="must have 1 row or 1 column"):
        cl.DevelopmentConstant(patterns=df_invalid).fit(raa)

    # Multi-origin Triangle pattern
    with pytest.raises(ValueError, match="must have a single origin row"):
        cl.DevelopmentConstant(patterns=raa).fit(raa)

    # Transformed multi-origin Triangle pattern (has ldf_ attribute attached)
    transformed_multi = cl.Development().fit_transform(raa)
    with pytest.raises(ValueError, match="must have a single origin row"):
        cl.DevelopmentConstant(patterns=transformed_multi).fit(raa)

    # Transformed 1-origin Triangle pattern (uses triangle values directly)
    transformed_1orig = transformed_multi.iloc[..., 0:1, :]
    model_1orig = cl.DevelopmentConstant(patterns=transformed_1orig, style="ldf").fit(
        raa
    )
    assert np.allclose(
        transformed_1orig.to_frame().values, model_1orig.ldf_.to_frame().values
    )

    # Unsupported pattern type
    with pytest.raises(TypeError, match="Unsupported pattern type"):
        cl.DevelopmentConstant(patterns=[1.2, 1.1]).fit(raa)
