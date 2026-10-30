from typing import Literal

import pytest
import chainladder as cl
import pandas as pd
import numpy as np


def test_parallelogram(clrd):
    lob = ["wkcomp"] * 3 + ["comauto"] * 3 + ["wkcomp"] * 2
    values = [0.05, 0.02, -0.1, 0.05, 0.05, 0.05, 0.2, 1 / 1.1 - 1]
    date = [
        "1/1/1989",
        "2/14/1990",
        "10/1/1992",
        "7/1/1988",
        "1/1/1990",
        "10/1/1993",
        "1/1/1996",
        "10/1/1992",
    ]
    rates = pd.DataFrame({"LOB": lob, "effdate": date, "change": values})

    olf = cl.ParallelogramOLF(
        rate_history=rates, change_col="change", date_col="effdate"
    )

    X = clrd["EarnedPremNet"].latest_diagonal
    X = X[X["LOB"].isin(["wkcomp", "comauto"])]
    X = olf.fit_transform(X)
    assert X.get_array_module().all(
        X.olf_.loc["comauto", "EarnedPremNet", "1994"].values
        - (9 / 12 * 9 / 12 / 2 * 0.05 + 1)
        < 0.005
    )
    assert X.get_array_module().all(
        X.olf_.loc["wkcomp", "EarnedPremNet", "1996"].values - 1.1 < 0.005
    )


def test_non_vertical_line():
    true_olf = (
        1.20
        / (
            (1 - 0.5 * ((31 + 31 + 30 + 31 + 30 + 31) / 365) ** 2) * 1.0
            + (0.5 * ((31 + 31 + 30 + 31 + 30 + 31) / 365) ** 2) * 1.2
        )
        - 1
    )

    result = (
        cl
        .parallelogram_olf([0.20], ["7/1/2017"], approximation_grain="D")
        .loc["2017"]
        .iloc[0]
        - 1
    )

    assert true_olf == result

    # Monthly approximation
    rate_history = pd.DataFrame({
        "EffDate": ["2010-07-01", "2011-01-01", "2012-07-01", "2013-04-01"],
        "RateChange": [0.035, 0.05, 0.10, -0.01],
    })

    data = pd.DataFrame({
        "Year": list(range(2006, 2016)),
        "EarnedPremium": [10_000] * 10,
    })

    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )
    prem_tri = cl.ParallelogramOLF(
        rate_history,
        change_col="RateChange",
        date_col="EffDate",
        approximation_grain="M",
        vertical_line=False,
    ).fit_transform(prem_tri)
    assert (
        np.round(prem_tri.olf_.to_frame().values, 6).flatten()
        == [
            1.183471,
            1.183471,
            1.183471,
            1.183471,
            1.178316,
            1.120181,
            1.075556,
            1.004236,
            0.999684,
            1.000000,
        ]
    ).all()

    # Daily approximation
    rate_history = pd.DataFrame({
        "EffDate": ["2010-07-01", "2011-01-01", "2012-07-01", "2013-04-01"],
        "RateChange": [0.035, 0.05, 0.10, -0.01],
    })

    data = pd.DataFrame({
        "Year": list(range(2006, 2016)),
        "EarnedPremium": [10_000] * 10,
    })

    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )
    prem_tri = cl.ParallelogramOLF(
        rate_history,
        change_col="RateChange",
        date_col="EffDate",
        approximation_grain="D",
        vertical_line=False,
    ).fit_transform(prem_tri)
    assert (
        np.round(prem_tri.olf_.to_frame().values, 6).flatten()
        == [
            1.183471,
            1.183471,
            1.183471,
            1.183471,
            1.178231,
            1.120105,
            1.075410,
            1.004073,
            0.999693,
            1.000000,
        ]
    ).all()


def test_vertical_line():
    olf = cl.parallelogram_olf(
        [0.20], ["7/1/2017"], approximation_grain="D", vertical_line=True
    )
    true_olf = 1.2 / ((1 - 184 / 365) * 1.0 + (184 / 365) * 1.2)
    assert abs(olf.loc["2017"].iloc[0] - true_olf) < 0.00001


def test_policy_length():
    rate_history = pd.DataFrame({
        "EffDate": ["2010-07-01", "2011-01-01", "2012-04-01"],
        "RateChange": [0.05, 0.1, -0.01],
    })
    data = pd.DataFrame({
        "Year": [2010, 2011, 2012, 2013, 2014],
        "EarnedPremium": [10_000] * 5,
    })
    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )

    prem_tri = cl.ParallelogramOLF(
        rate_history, change_col="RateChange", date_col="EffDate", policy_length=12
    ).fit_transform(prem_tri)
    assert (
        np.round(prem_tri.olf_.values.flatten(), 6)
        == [1.136348, 1.043056, 0.992792, 0.999684, 1]
    ).all()

    prem_tri = cl.ParallelogramOLF(
        rate_history, change_col="RateChange", date_col="EffDate", policy_length=6
    ).fit_transform(prem_tri)
    assert (
        np.round(prem_tri.olf_.values.flatten(), 6)
        == [1.129333, 1.013023, 0.994975, 1, 1]
    ).all()

    rate_history = pd.DataFrame({
        "EffDate": ["2010-07-01", "2011-10-01", "2012-04-01"],
        "RateChange": [0.35, 0.149, -0.095],
    })
    data = pd.DataFrame({
        "Year": [2010, 2011, 2012, 2013, 2014],
        "EarnedPremium": [10_000] * 5,
    })
    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )

    prem_tri = cl.ParallelogramOLF(
        rate_history,
        change_col="RateChange",
        date_col="EffDate",
        policy_length=12,
        approximation_grain="M",
    ).fit_transform(prem_tri)
    assert (
        np.round(prem_tri.olf_.values.flatten(), 6)
        == [1.344949, 1.069526, 0.966045, 0.996730, 1]
    ).all()

    prem_tri = cl.ParallelogramOLF(
        rate_history,
        change_col="RateChange",
        date_col="EffDate",
        policy_length=6,
        approximation_grain="M",
    ).fit_transform(prem_tri)
    assert (
        np.round(prem_tri.olf_.values.flatten(), 6)
        == [1.290842, 1.030251, 0.958285, 1, 1]
    ).all()

    rate_history = pd.DataFrame({
        "EffDate": ["2010-07-01"],
        "RateChange": [0.20],
    })
    data = pd.DataFrame({
        "Year": [2010, 2011, 2012, 2013, 2014],
        "EarnedPremium": [10_000] * 5,
    })
    prem_tri = cl.Triangle(
        data,
        origin="Year",
        columns="EarnedPremium",
        cumulative=True,
    )

    lhs = np.round(
        cl
        .ParallelogramOLF(
            rate_history,
            change_col="RateChange",
            date_col="EffDate",
            policy_length=24,
            approximation_grain="M",
        )
        .fit_transform(prem_tri)
        .olf_.to_frame()
        .values.flatten(),
        6,
    )
    rhs = [1.185185, 1.090909, 1.010526, 1, 1]
    assert np.all(lhs == rhs)

    data = [
        [2002, 61183, 0],
        [2003, 69175, 0.05],
        [2004, 99322, 0.075],
        [2005, 138151, 0.15],
        [2006, 107578, 0.1],
        [2007, 62438, -0.2],
        [2008, 47797, -0.2],
    ]
    columns = ["Calendar Year", "Earned Premiums", "Rate Changes"]
    df_prem = pd.DataFrame(data, columns=columns)
    df_prem["Date"] = pd.to_datetime(
        df_prem["Calendar Year"].astype(int).astype(str) + "-01-01"
    )

    assert (
        cl
        .parallelogram_olf(df_prem["Rate Changes"], df_prem["Date"])
        .reset_index()["OLF"]
        .notna()
        .all()
    )
    assert (
        cl
        .parallelogram_olf(df_prem["Rate Changes"], df_prem["Date"], policy_length=12)
        .reset_index()["OLF"]
        .notna()
        .all()
    )
    assert (
        cl
        .parallelogram_olf(df_prem["Rate Changes"], df_prem["Date"], policy_length=6)
        .reset_index()["OLF"]
        .notna()
        .all()
    )
    assert (
        cl
        .parallelogram_olf(
            df_prem["Rate Changes"],
            df_prem["Date"],
            policy_length=6,
            approximation_grain="D",
        )
        .reset_index()["OLF"]
        .notna()
        .all()
    )


def test_rate_impact_middle_of_year():
    rate_history = pd.DataFrame({
        "EffDate": ["2010-01-01"],
        "RateChange": [0.20],
    })
    data = pd.DataFrame({
        "Year": [2010, 2011, 2012, 2013, 2014],
        "EarnedPremium": [10_000] * 5,
    })
    prem_tri = cl.Triangle(
        data,
        origin="Year",
        columns="EarnedPremium",
        cumulative=True,
    )

    monthly = np.round(
        cl
        .ParallelogramOLF(
            rate_history,
            change_col="RateChange",
            date_col="EffDate",
            policy_length=24,
            approximation_grain="M",
        )
        .fit_transform(prem_tri)
        .olf_.to_frame()
        .values.flatten(),
        6,
    )
    # print(monthly)
    daily = np.round(
        cl
        .ParallelogramOLF(
            rate_history,
            change_col="RateChange",
            date_col="EffDate",
            policy_length=24,
            approximation_grain="D",
        )
        .fit_transform(prem_tri)
        .olf_.to_frame()
        .values.flatten(),
        6,
    )
    assert np.all(
        monthly == daily
    )  # when rate change is effective on 1/1, there's no difference in daily or monthly approximatation


def test_rate_impact_beginning_of_year():
    rate_history = pd.DataFrame({
        "EffDate": ["2010-07-01"],
        "RateChange": [0.20],
    })
    data = pd.DataFrame({
        "Year": [2010, 2011, 2012, 2013, 2014],
        "EarnedPremium": [10_000] * 5,
    })
    prem_tri = cl.Triangle(
        data,
        origin="Year",
        columns="EarnedPremium",
        cumulative=True,
    )

    monthly = np.round(
        cl
        .ParallelogramOLF(
            rate_history,
            change_col="RateChange",
            date_col="EffDate",
            policy_length=24,
            approximation_grain="M",
        )
        .fit_transform(prem_tri)
        .olf_.to_frame()
        .values.flatten(),
        6,
    )
    # print(monthly)
    daily = np.round(
        cl
        .ParallelogramOLF(
            rate_history,
            change_col="RateChange",
            date_col="EffDate",
            policy_length=24,
            approximation_grain="D",
        )
        .fit_transform(prem_tri)
        .olf_.to_frame()
        .values.flatten(),
        6,
    )
    assert np.array_equal(
        np.where(monthly > daily, ">", np.where(monthly == daily, "=", "<")),
        np.array([">", ">", ">", "=", "="]),
    )  # this is true becuase there are less "days" in the first half of the year (from Jan - Jun) compared to (Jul - Dec), and only the first three origins would need to be brought to current rate level


def test_cumulative_tort_reform():
    """Cumulative on-level factors can be supplied directly. See GH #922."""
    tort = pd.DataFrame({
        "EffDate": ["1998-01-01", "2003-01-01", "2004-01-01"],
        "Factor": [0.67, 0.75, 1.00],
    })
    olf = (
        cl
        .ParallelogramOLF(
            rate_history=tort,
            change_col="Factor",
            date_col="EffDate",
            policy_length=12,
            approximation_grain="M",
            vertical_line=True,
            cumulative=True,
        )
        .fit_transform(cl.load_sample("friedland_gl_self_insurer")["Reported Claims"])
        .olf_
    )
    assert np.all(
        np.round(olf.to_frame().values.flatten(), 6)
        == [0.67, 0.67, 0.67, 0.67, 0.67, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_cumulative_matches_incremental():
    """Cumulative factors and their incremental equivalent agree."""
    data = pd.DataFrame({
        "Year": list(range(2006, 2016)),
        "EarnedPremium": [10_000] * 10,
    })
    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )
    dates = ["2006-01-01", "2010-07-01", "2011-01-01", "2012-07-01", "2013-04-01"]
    changes = [0.0, 0.035, 0.05, 0.10, -0.01]
    levels = np.cumprod(np.array(changes) + 1)
    factors = levels[-1] / levels

    for grain in ["M", "D"]:
        for vertical_line in [True, False]:
            kw = dict(
                date_col="EffDate",
                approximation_grain=grain,
                vertical_line=vertical_line,
            )
            incremental = cl.ParallelogramOLF(
                pd.DataFrame({"EffDate": dates, "RateChange": changes}),
                change_col="RateChange",
                **kw,
            ).fit_transform(prem_tri)
            cumulative = cl.ParallelogramOLF(
                pd.DataFrame({"EffDate": dates, "Factor": factors}),
                change_col="Factor",
                cumulative=True,
                **kw,
            ).fit_transform(prem_tri)
            assert np.allclose(incremental.olf_.values, cumulative.olf_.values), (
                grain,
                vertical_line,
            )


def test_cumulative_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        cl.parallelogram_olf([1.0, 0.0], ["2010-01-01", "2011-01-01"], cumulative=True)


def test_cumulative_factor_predates_window():
    """
    A factor in force before the triangle window is honored, not dropped.

    The earliest factor's effective date (2000) predates the triangle's
    lookback window (origins start 2003), so it must still apply to the first
    origins rather than being backfilled with a later factor. See GH #922.
    """
    data = pd.DataFrame({"Year": list(range(2003, 2009)), "EarnedPremium": [1000] * 6})
    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )
    factors = pd.DataFrame({
        "EffDate": ["2000-01-01", "2005-01-01"],
        "Factor": [0.5, 1.0],
    })
    olf = (
        cl
        .ParallelogramOLF(
            rate_history=factors,
            change_col="Factor",
            date_col="EffDate",
            approximation_grain="M",
            vertical_line=True,
            cumulative=True,
        )
        .fit_transform(prem_tri)
        .olf_
    )
    assert np.all(
        np.round(olf.to_frame().values.flatten(), 6) == [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    )


def test_cumulative_duplicate_date_last_wins():
    """Duplicate effective dates keep the last cumulative factor, not a product."""
    dup = pd.DataFrame({
        "EffDate": ["1998-01-01", "2003-01-01", "2003-01-01", "2004-01-01"],
        "Factor": [0.67, 0.67, 0.75, 1.00],
    })
    olf = (
        cl
        .ParallelogramOLF(
            rate_history=dup,
            change_col="Factor",
            date_col="EffDate",
            approximation_grain="M",
            vertical_line=True,
            cumulative=True,
        )
        .fit_transform(cl.load_sample("friedland_gl_self_insurer")["Reported Claims"])
        .olf_
    )
    assert np.all(
        np.round(olf.to_frame().values.flatten(), 6)
        == [0.67, 0.67, 0.67, 0.67, 0.67, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
    )


def _olf_for_freq(freq, rates):
    """Fit ParallelogramOLF against a premium triangle of the given origin grain."""
    origin = pd.date_range("2016-01-01", "2018-12-31", freq=freq)
    triangle = cl.Triangle(
        pd.DataFrame({"origin": origin, "values": 1.0}),
        origin="origin",
        columns="values",
        cumulative=True,
    )
    olf = cl.ParallelogramOLF(
        rate_history=rates,
        change_col="RateChange",
        date_col="EffDate",
        vertical_line=False,
    ).fit(triangle)
    return triangle.origin_grain, np.asarray(olf.olf_.values).flatten()


RATE_HISTORY = pd.DataFrame({
    "EffDate": [pd.Timestamp("2016-07-01"), pd.Timestamp("2017-04-01")],
    "RateChange": [0.08, 0.05],
})


@pytest.mark.parametrize(
    "freq, grain, origins",
    [("YS", "Y", 3), ("2QS", "S", 6), ("QS", "Q", 12), ("MS", "M", 36)],
)
def test_olf_supports_every_origin_grain(freq, grain, origins):
    # Quarterly and semiannual premium triangles used to raise ValueError, so
    # only two of the four grains Triangle.origin_grain can return were usable.
    # The row count is asserted because parallelogram_olf is broadcast against
    # the triangle positionally: a semiannual result of length 12 would align
    # against the wrong origins rather than fail.
    observed_grain, olf = _olf_for_freq(freq, RATE_HISTORY)
    assert observed_grain == grain
    assert len(olf) == origins


@pytest.mark.parametrize(
    "grain, freq, months", [("Y", "YS", 12), ("S", "2QS", 6), ("Q", "QS", 3)]
)
def test_olf_aggregates_consistently_with_monthly(grain, freq, months):
    # An on-level factor is a ratio of rate levels, so a coarser grain is the
    # reciprocal of the mean reciprocal of the months it spans. The "Y" case
    # holds on unfixed code too and is kept as a control on the identity.
    _, monthly = _olf_for_freq("MS", RATE_HISTORY)
    _, coarse = _olf_for_freq(freq, RATE_HISTORY)
    expected = np.array([
        1 / np.mean(1 / monthly[i : i + months]) for i in range(0, len(monthly), months)
    ])
    np.testing.assert_allclose(coarse, expected, rtol=1e-9)


def test_daily_grain_after_leap_year():
    # A daily approximation on a triangle whose origins start right after a leap
    # year used to trim the lookback window by row count, which dropped one more
    # row in the leap pass than the non-leap pass. When that row was the only one
    # of its origin period the two passes produced a different number of origins
    # and .fit() failed with a broadcast error.
    rates = pd.DataFrame({
        "EffDate": [pd.Timestamp("2016-07-01")],
        "RateChange": [0.05],
    })
    origin = pd.date_range("2017-01-01", "2019-12-31", freq="YS")
    triangle = cl.Triangle(
        pd.DataFrame({"origin": origin, "values": 1.0}),
        origin="origin",
        columns="values",
        cumulative=True,
    )
    olf = cl.ParallelogramOLF(
        rate_history=rates,
        change_col="RateChange",
        date_col="EffDate",
        approximation_grain="D",
    ).fit(triangle)
    assert len(olf.olf_.origin) == len(triangle.origin)
    assert not olf.olf_.to_frame(origin_as_datetime=False).isna().any().any()


def test_off_grid_incremental_change_is_not_dropped():
    """A rate change dated mid-month must not vanish in monthly mode."""
    kw = dict(start_date="2000-01-01", end_date="2003-12-31", grain="Y")
    off = cl.parallelogram_olf(values=[0.20], dates=["2001-07-15"], **kw)
    # In monthly mode the change takes effect at the next grid point.
    on = cl.parallelogram_olf(values=[0.20], dates=["2001-08-01"], **kw)
    assert np.isclose(off.loc["2000", "OLF"], 1.2)
    assert np.allclose(off["OLF"].values, on["OLF"].values)


def test_off_grid_incremental_matches_cumulative():
    """Incremental and cumulative inputs agree for off-grid effective dates."""
    data = pd.DataFrame({"Year": list(range(2006, 2016)), "EarnedPremium": [1e4] * 10})
    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )
    dates = ["2006-01-01", "2010-07-15", "2011-02-14", "2012-06-30", "2013-04-10"]
    changes = [0.0, 0.035, 0.05, 0.10, -0.01]
    levels = np.cumprod(np.array(changes) + 1)
    factors = levels[-1] / levels
    for grain in ["M", "D"]:
        for vertical_line in [True, False]:
            kw = dict(
                date_col="EffDate",
                approximation_grain=grain,
                vertical_line=vertical_line,
            )
            incremental = cl.ParallelogramOLF(
                pd.DataFrame({"EffDate": dates, "RateChange": changes}),
                change_col="RateChange",
                **kw,
            ).fit_transform(prem_tri)
            cumulative = cl.ParallelogramOLF(
                pd.DataFrame({"EffDate": dates, "Factor": factors}),
                change_col="Factor",
                cumulative=True,
                **kw,
            ).fit_transform(prem_tri)
            assert np.allclose(incremental.olf_.values, cumulative.olf_.values), (
                grain,
                vertical_line,
            )


# Rate levels of the documented rate-history example below: +2% on 2016-07-15,
# +5% on 2017-03-01, -3% on 2018-01-01 and +10% on 2019-10-31.
L0 = 1.0
L1 = L0 * 1.02
L2 = L1 * 1.05
L3 = L2 * 0.97
L4 = L3 * 1.10  # current rate level, 1.142757


def _tri(n, term):
    """
    Earned-exposure share of a corner triangle with legs n / term.

    Policies written uniformly over the year and earned uniformly over a
    policy term of ``term`` periods: the part of a calendar year's earned
    exposure that comes from policies written during the first (or last)
    ``n`` periods is the triangle (n / term) ** 2 / 2.
    """
    return (n / term) ** 2 / 2


@pytest.mark.parametrize(
    "approximation_grain, expected",
    [
        (
            # Monthly convention: 12 equal months, policy term 12 months. An
            # off-grid date changes the rate from the next month start, so
            # 2016-07-15 acts from Aug 1, 2016 and 2019-10-31 from Nov 1, 2019.
            "M",
            [
                # 2016: policies written Aug-Dec 2016 (5 months) are at L1.
                # = 10284813 / 9015625
                L4 / (L0 * (1 - _tri(5, 12)) + L1 * _tri(5, 12)),
                # 2017: L0 policies written Jan-Jul 2016 still earn during
                # Jan-Jul 2017 (7 months); policies written Mar-Dec 2017
                # (10 months) are at L2.
                # = 934983 / 846250
                L4
                / (
                    L0 * _tri(7, 12)
                    + L1 * (1 - _tri(7, 12) - _tri(10, 12))
                    + L2 * _tri(10, 12)
                ),
                # 2018: L1 policies written Jan-Feb 2017 still earn during
                # Jan-Feb 2018 (2 months); policies written in 2018 are at L3
                # and earn half of the year's exposure.
                # = 201663 / 186040
                L4 / (L1 * _tri(2, 12) + L2 * (1 / 2 - _tri(2, 12)) + L3 / 2),
                # 2019: policies written Nov-Dec 2019 (2 months) are at L4,
                # i.e. 1.1 / (1 + 0.1 * 1 / 72).
                # = 792 / 721
                L4 / (L3 * (1 - _tri(2, 12)) + L4 * _tri(2, 12)),
                # 2020: L3 policies written Jan-Oct 2019 still earn during
                # Jan-Oct 2020 (10 months), i.e. 1.1 / (1 + 0.1 * 47 / 72).
                # = 792 / 767
                L4 / (L3 * _tri(10, 12) + L4 * (1 - _tri(10, 12))),
            ],
        ),
        (
            # Daily convention: actual days, effective dates honored exactly,
            # policy term of 365 days for non-leap origins and 366 days for
            # leap origins (2016, 2020), as in the current algorithm.
            "D",
            [
                # 2016: 170 days from 2016-07-15 to 2016-12-31 are at L1.
                # = 38269789173 / 33561250000
                L4 / (L0 * (1 - _tri(170, 366)) + L1 * _tri(170, 366)),
                # 2017: L0 policies still earn from 2017-01-01 to 2017-07-14
                # (195 days); policies written from 2017-03-01 to 2017-12-31
                # (306 days) are at L2.
                # = 184537941 / 167147840
                L4
                / (
                    L0 * _tri(195, 365)
                    + L1 * (1 - _tri(195, 365) - _tri(306, 365))
                    + L2 * _tri(306, 365)
                ),
                # 2018: L1 policies written in Jan-Feb 2017 still earn from
                # 2018-01-01 to 2018-02-28 (59 days); 2018 writings are at L3.
                # = 119406903 / 110160745
                L4 / (L1 * _tri(59, 365) + L2 * (1 / 2 - _tri(59, 365)) + L3 / 2),
                # 2019: 62 days from 2019-10-31 to 2019-12-31 are at L4.
                # = 1465475 / 1334172
                L4 / (L3 * (1 - _tri(62, 365)) + L4 * _tri(62, 365)),
                # 2020: L3 policies still earn from 2020-01-01 to 2020-10-30
                # (304 days).
                # = 368379 / 356827
                L4 / (L3 * _tri(304, 366) + L4 * (1 - _tri(304, 366))),
            ],
        ),
    ],
)
def test_documented_rate_history_independent_geometry(
    approximation_grain: Literal["M", "D"], expected: list[float]
) -> None:
    """
    Pin the documented rate-history example to independent geometry.

    Uniform issuance of equal exposure, uniform earning during the policy term;
    OLF = current rate level / average earned rate level, not the average of
    individual OLFs. The expected values are the elementary parallelogram
    areas written out above; the exact fractions in the comments were derived
    separately with rational arithmetic by integrating each policy cohort's
    earned overlap with the origin year. This does not assert exact
    calendar-anniversary earning across leap years.
    """
    rate_history = pd.DataFrame({
        "EffDate": ["2016-07-15", "2017-03-01", "2018-01-01", "2019-10-31"],
        "RateChange": [0.02, 0.05, -0.03, 0.1],
    })
    data = pd.DataFrame({"Year": range(2016, 2021), "EarnedPremium": [10000] * 5})
    premium = cl.Triangle(data, origin="Year", columns="EarnedPremium", cumulative=True)
    result = (
        cl
        .ParallelogramOLF(
            rate_history,
            change_col="RateChange",
            date_col="EffDate",
            approximation_grain=approximation_grain,
            policy_length=12,
        )
        .fit_transform(premium)
        .olf_.values.flatten()
    )
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-14)
