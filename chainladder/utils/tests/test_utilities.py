import chainladder as cl
from chainladder.utils.cupy import cp
import numpy as np
import copy


def test_non_vertical_line():
    true_olf = (1 - 0.5 * (184 / 365) ** 2) * 0.2
    olf_low = (
        cl.parallelogram_olf([0.20], ["7/1/2017"], grain="Y").loc["2017"].iloc[0] - 1
    )
    olf_high = (
        cl.parallelogram_olf([0.20], ["7/2/2017"], grain="Y").loc["2017"].iloc[0] - 1
    )
    assert olf_low < true_olf < olf_high


def test_vertical_line():
    olf = cl.parallelogram_olf([0.20], ["7/1/2017"], grain="Y", vertical_line=True)
    assert abs(olf.loc["2017"].iloc[0] - ((1 - 184 / 365) * 0.2 + 1)) < 0.00001


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
    a = cl.read_json(
        cl.Chainladder().fit_predict(raa).to_json()
    ).full_triangle_
    b = cl.Chainladder().fit_predict(raa).full_triangle_
    abs(a - b).max().max() < 1e-4


def test_json_df():
    x = cl.MunichAdjustment(paid_to_incurred=("paid", "incurred")).fit_transform(
        cl.load_sample("mcl")
    )
    assert abs(cl.read_json(x.to_json()).lambda_ - x.lambda_).sum() < 1e-5


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