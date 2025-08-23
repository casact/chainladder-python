import pytest

import chainladder as cl
from chainladder.utils.cupy import cp
import numpy as np
import copy
import pandas as pd


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


def test_triangle_json_io(clrd):
    xp = clrd.get_array_module()
    clrd2 = cl.read_json(clrd.to_json(), array_backend=clrd.array_backend)
    xp.testing.assert_array_equal(clrd.values, clrd2.values)
    xp.testing.assert_array_equal(clrd.kdims, clrd2.kdims)
    xp.testing.assert_array_equal(clrd.vdims, clrd2.vdims)
    xp.testing.assert_array_equal(clrd.odims, clrd2.odims)
    xp.testing.assert_array_equal(clrd.ddims, clrd2.ddims)
    assert np.all(clrd.valuation == clrd2.valuation)


def test_json_for_val(raa):
    x = raa.dev_to_val().to_json()
    assert cl.read_json(x) == raa.dev_to_val()


def test_estimator_json_io():
    assert (
        cl.read_json(cl.Development().to_json()).get_params()
        == cl.Development().get_params()
    )


def test_pipeline_json_io():
    pipe = cl.Pipeline(
        steps=[("dev", cl.Development()), ("model", cl.BornhuetterFerguson())]
    )
    pipe2 = cl.read_json(pipe.to_json())
    assert {item[0]: item[1].get_params() for item in pipe.get_params()["steps"]} == {
        item[0]: item[1].get_params() for item in pipe2.get_params()["steps"]
    }


def test_json_subtri(raa):
    a = cl.read_json(cl.Chainladder().fit_predict(raa).to_json()).full_triangle_
    b = cl.Chainladder().fit_predict(raa).full_triangle_
    assert abs(a - b).max().max() < 1e-4


def test_json_df():
    x = cl.MunichAdjustment(paid_to_incurred=("paid", "incurred")).fit_transform(
        cl.load_sample("mcl")
    )
    assert abs(cl.read_json(x.to_json()).lambda_ - x.lambda_).sum() < 1e-5

def test_read_csv_single(raa):
    # Test the read_csv function for a single dimensional input.
    
    # Read in the csv file.
    from pathlib import Path
    raa_csv_path = Path(__file__).parent.parent / "data" / "raa.csv"

    assert raa == cl.read_csv(
        filepath_or_buffer=raa_csv_path,
        origin = "origin",
        development = "development",
        columns = ["values"],
        index = None,
        cumulative = True)

def test_read_csv_multi(clrd):
    # Test the read_csv function for multidimensional input.

    # Read in the csv file.
    from pathlib import Path
    clrd_csv_path = Path(__file__).parent.parent / "data" / "clrd.csv"

    assert clrd == cl.read_csv(
        filepath_or_buffer=clrd_csv_path,
        origin = "AccidentYear",
        development = "DevelopmentYear",
        columns = [
            "IncurLoss",
            "CumPaidLoss",
            "BulkLoss",
            "EarnedPremDIR",
            "EarnedPremCeded",
            "EarnedPremNet",
        ],
        index = ["GRNAME","LOB"],
        cumulative = True
    ) 

def test_concat(clrd):
    tri = clrd.groupby("LOB").sum()
    assert (
        cl.concat([tri.loc["wkcomp"], tri.loc["comauto"]], axis=0)
        == tri.loc[["wkcomp", "comauto"]]
    )


def test_model_diagnostics(qtr):
    cl.model_diagnostics(cl.Chainladder().fit(qtr))


def test_concat_immutability(raa):
    u = cl.Chainladder().fit(raa).ultimate_
    l = raa.latest_diagonal
    u.columns = l.columns
    u_new = copy.deepcopy(u)
    cl.concat((l, u), axis=3)
    assert u == u_new


def test_invalid_sample() -> None:
    """
    Test that an invalid sample name provided to cl.load_sample() raises an error.
    """
    with pytest.raises(ValueError):
        cl.load_sample(key="not_a_real_sample_38473743")
