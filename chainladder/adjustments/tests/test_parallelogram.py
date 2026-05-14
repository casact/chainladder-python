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
        cl.parallelogram_olf([0.20], ["7/1/2017"], approximation_grain="D")
        .loc["2017"]
        .iloc[0]
        - 1
    )

    assert true_olf == result

    # Monthly approximation
    rate_history = pd.DataFrame(
        {
            "EffDate": ["2010-07-01", "2011-01-01", "2012-07-01", "2013-04-01"],
            "RateChange": [0.035, 0.05, 0.10, -0.01],
        }
    )

    data = pd.DataFrame(
        {"Year": list(range(2006, 2016)), "EarnedPremium": [10_000] * 10}
    )

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
    rate_history = pd.DataFrame(
        {
            "EffDate": ["2010-07-01", "2011-01-01", "2012-07-01", "2013-04-01"],
            "RateChange": [0.035, 0.05, 0.10, -0.01],
        }
    )

    data = pd.DataFrame(
        {"Year": list(range(2006, 2016)), "EarnedPremium": [10_000] * 10}
    )

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
    rate_history = pd.DataFrame(
        {
            "EffDate": ["2010-07-01", "2011-01-01", "2012-04-01"],
            "RateChange": [0.05, 0.1, -0.01],
        }
    )
    data = pd.DataFrame(
        {"Year": [2010, 2011, 2012, 2013, 2014], "EarnedPremium": [10_000] * 5}
    )
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

    rate_history = pd.DataFrame(
        {
            "EffDate": ["2010-07-01", "2011-10-01", "2012-04-01"],
            "RateChange": [0.35, 0.149, -0.095],
        }
    )
    data = pd.DataFrame(
        {"Year": [2010, 2011, 2012, 2013, 2014], "EarnedPremium": [10_000] * 5}
    )
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

    rate_history = pd.DataFrame(
        {
            "EffDate": ["2010-07-01"],
            "RateChange": [0.20],
        }
    )
    data = pd.DataFrame(
        {"Year": [2010, 2011, 2012, 2013, 2014], "EarnedPremium": [10_000] * 5}
    )
    prem_tri = cl.Triangle(
        data,
        origin="Year",
        columns="EarnedPremium",
        cumulative=True,
    )

    lhs = np.round(
        cl.ParallelogramOLF(
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
        cl.parallelogram_olf(df_prem["Rate Changes"], df_prem["Date"])
        .reset_index()["OLF"]
        .notna()
        .all()
    )
    assert (
        cl.parallelogram_olf(df_prem["Rate Changes"], df_prem["Date"], policy_length=12)
        .reset_index()["OLF"]
        .notna()
        .all()
    )
    assert (
        cl.parallelogram_olf(df_prem["Rate Changes"], df_prem["Date"], policy_length=6)
        .reset_index()["OLF"]
        .notna()
        .all()
    )
    assert (
        cl.parallelogram_olf(
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
    rate_history = pd.DataFrame(
        {
            "EffDate": ["2010-01-01"],
            "RateChange": [0.20],
        }
    )
    data = pd.DataFrame(
        {"Year": [2010, 2011, 2012, 2013, 2014], "EarnedPremium": [10_000] * 5}
    )
    prem_tri = cl.Triangle(
        data,
        origin="Year",
        columns="EarnedPremium",
        cumulative=True,
    )

    monthly = np.round(
        cl.ParallelogramOLF(
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
        cl.ParallelogramOLF(
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
    rate_history = pd.DataFrame(
        {
            "EffDate": ["2010-07-01"],
            "RateChange": [0.20],
        }
    )
    data = pd.DataFrame(
        {"Year": [2010, 2011, 2012, 2013, 2014], "EarnedPremium": [10_000] * 5}
    )
    prem_tri = cl.Triangle(
        data,
        origin="Year",
        columns="EarnedPremium",
        cumulative=True,
    )

    monthly = np.round(
        cl.ParallelogramOLF(
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
        cl.ParallelogramOLF(
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
