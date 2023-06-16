import chainladder as cl
import pandas as pd
import numpy as np
import copy
import pytest


def test_grain(qtr):
    actual = qtr.iloc[0, 0].grain("OYDY")
    xp = actual.get_array_module()
    nan = xp.nan
    expected = xp.array(
        [
            [44, 621, 950, 1020, 1070, 1069, 1089, 1094, 1097, 1099, 1100, 1100],
            [42, 541, 1052, 1169, 1238, 1249, 1266, 1269, 1296, 1300, 1300, nan],
            [17, 530, 966, 1064, 1100, 1128, 1155, 1196, 1201, 1200, nan, nan],
            [10, 393, 935, 1062, 1126, 1209, 1243, 1286, 1298, nan, nan, nan],
            [13, 481, 1021, 1267, 1400, 1476, 1550, 1583, nan, nan, nan, nan],
            [2, 380, 788, 953, 1001, 1030, 1066, nan, nan, nan, nan, nan],
            [4, 777, 1063, 1307, 1362, 1411, nan, nan, nan, nan, nan, nan],
            [2, 472, 1617, 1818, 1820, nan, nan, nan, nan, nan, nan, nan],
            [3, 597, 1092, 1221, nan, nan, nan, nan, nan, nan, nan, nan],
            [4, 583, 1212, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [21, 422, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [13, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
        ]
    )
    xp.testing.assert_array_equal(actual.values[0, 0, :, :], expected)


def test_grain_returns_valid_tri(qtr):
    assert qtr.grain("OYDY").latest_diagonal == qtr.latest_diagonal


def test_grain_increm_arg(qtr):
    clrd_i = qtr["incurred"].cum_to_incr()
    a = clrd_i.grain("OYDY").incr_to_cum()
    assert a == qtr["incurred"].grain("OYDY")


def test_commutative(qtr, atol):
    xp = qtr.get_array_module()
    full = cl.Chainladder().fit(qtr).full_expectation_
    assert qtr.grain("OYDY").val_to_dev() == qtr.val_to_dev().grain("OYDY")
    assert qtr.cum_to_incr().grain(
        "OYDY"
    ).val_to_dev() == qtr.val_to_dev().cum_to_incr().grain("OYDY")
    assert qtr.grain(
        "OYDY"
    ).cum_to_incr().val_to_dev().incr_to_cum() == qtr.val_to_dev().grain("OYDY")
    assert full.grain("OYDY").val_to_dev() == full.val_to_dev().grain("OYDY")
    assert full.cum_to_incr().grain(
        "OYDY"
    ).val_to_dev() == full.val_to_dev().cum_to_incr().grain("OYDY")
    a = full.grain("OYDY").cum_to_incr().val_to_dev().incr_to_cum()
    b = full.val_to_dev().grain("OYDY")
    assert abs(a - b).max().max().max() < atol


@pytest.mark.parametrize(
    "grain", ["OYDY", "OYDQ", "OYDM", "OSDS", "OSDQ", "OSDM", "OQDQ", "OQDM"]
)
@pytest.mark.parametrize("alt", [0, 1, 2])
@pytest.mark.parametrize("trailing", [False, True])
def test_different_forms_of_grain(prism_dense, grain, trailing, alt, atol):
    t = prism_dense["Paid"]
    if alt == 1:
        t = t.dev_to_val().copy()
    if alt == 2:
        t = t.val_to_dev().copy()
        t = t[t.valuation < "2017-09"]

    a = t.incr_to_cum().grain(grain, trailing=trailing)
    b = t.grain(grain, trailing=trailing).incr_to_cum()
    assert abs(a - b).sum().sum() < atol


def test_asymmetric_origin_grain(prism_dense):
    x = prism_dense.iloc[..., 8:, :].incr_to_cum()
    x = x[x.valuation < x.valuation_date]
    assert x.grain("OYDM").development[0] == 1

    x = x[x.valuation < x.valuation_date]
    assert x.grain("OYDM").development[0] == 1


def test_vector_triangle_grain_mismatch(prism):
    tri = prism["Paid"].sum().incr_to_cum().grain("OQDM")
    exposure = tri.latest_diagonal
    tri = tri.grain("OQDQ")
    assert (tri / exposure).development_grain == "Q"


def test_annual_trailing(prism):
    tri = prism["Paid"].sum().incr_to_cum()
    # (limit data to November)
    tri = tri[tri.valuation < tri.valuation_date].incr_to_cum()
    tri = tri.grain("OQDQ", trailing=True).grain("OYDY")
    assert np.all(tri.ddims[:4] == np.array([12, 24, 36, 48]))


def test_development_age():
    assert (
        cl.load_sample("raa").ddims == [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
    ).all()


def test_development_age_quarterly():
    assert (
        cl.load_sample("quarterly").ddims
        == [
            3,
            6,
            9,
            12,
            15,
            18,
            21,
            24,
            27,
            30,
            33,
            36,
            39,
            42,
            45,
            48,
            51,
            54,
            57,
            60,
            63,
            66,
            69,
            72,
            75,
            78,
            81,
            84,
            87,
            90,
            93,
            96,
            99,
            102,
            105,
            108,
            111,
            114,
            117,
            120,
            123,
            126,
            129,
            132,
            135,
        ]
    ).all()
