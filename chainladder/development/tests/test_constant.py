import chainladder as cl
import numpy as np
import pandas as pd
import pytest


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
        xerror = cl.DevelopmentConstant(
            patterns=paid_cdfs, callable_axis=2, style="cdf"
        ).fit(agway)
    lhs = (
        cl.DevelopmentConstant(patterns=paid_cdfs, callable_axis=1, style="cdf")
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
        np.round(result.cdf_.to_frame().values.flatten(), 6)
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
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
        == np.array(
            [
                2.593742,
                2.357948,
                2.143589,
                1.948717,
                1.771561,
                1.61051,
                1.4641,
                1.331,
                1.21,
                1.1,
            ]
        )
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
        np.round(result.cdf_.to_frame().values.flatten(), 6)
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0])
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
        == np.array([1.771561, 1.61051, 1.4641, 1.331, 1.21, 1.1, 1.0, 1.0, 1.0])
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
        == np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
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
        == np.array(
            [
                2.853117,
                2.593742,
                2.357948,
                2.143589,
                1.948717,
                1.771561,
                1.61051,
                1.4641,
                1.331,
                1.21,
            ]
        )
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
