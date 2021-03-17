import numpy as np
import chainladder as cl
import pytest


@pytest.fixture
def triangle_data():
    clrd = cl.load_sample("clrd")[["CumPaidLoss", "EarnedPremDIR"]]
    clrd = clrd[clrd["LOB"] == "wkcomp"]
    return clrd


@pytest.fixture
def estimators():
    bcl = cl.Chainladder()
    bf = cl.BornhuetterFerguson()
    cc = cl.CapeCod()

    estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]

    return estimators


array_weight = np.array([[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3)

list_weight = [[[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3]

callable_weight = lambda origin: np.where(
        origin.year < 1992,
        (1, 2, 3),
        np.where(origin.year > 1994, (0, 0, 1), (0, 0.5, 0.5))
    )

dict_weight = {
    '1992': (0, 0.5, 0.5),
    '1993': (0, 0.5, 0.5),
    '1994': (0, 0.5, 0.5),
    '1995': (0, 0, 1),
    '1996': (0, 0, 1),
    '1997': (0, 0, 1),
    }


@pytest.fixture(params=[array_weight, list_weight, callable_weight, dict_weight])
def weights(request):
    return request.param


def test_voting_ultimate(triangle_data, estimators, weights):
    bcl_ult = cl.Chainladder().fit(
        triangle_data["CumPaidLoss"].sum(),
        ).ultimate_
    bf_ult = cl.BornhuetterFerguson().fit(
        triangle_data["CumPaidLoss"].sum(),
        sample_weight=triangle_data["EarnedPremDIR"].sum().latest_diagonal
        ).ultimate_
    cc_ult = cl.CapeCod().fit(
        triangle_data["CumPaidLoss"].sum(),
        sample_weight=triangle_data["EarnedPremDIR"].sum().latest_diagonal
        ).ultimate_

    vot_ult = cl.VotingChainladder(
        estimators=estimators,
        weights=weights,
        default_weighting=(1, 2, 3)
    ).fit(
        triangle_data["CumPaidLoss"].sum(),
        sample_weight=triangle_data["EarnedPremDIR"].sum().latest_diagonal,
    ).ultimate_

    direct_weight = np.array([[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3)
    direct_weight = direct_weight[..., np.newaxis]

    assert abs(
        (
            (
                bcl_ult * direct_weight[..., 0, :] +
                bf_ult * direct_weight[..., 1, :] +
                cc_ult * direct_weight[..., 2, :]
            ) / direct_weight.sum(axis=-2)
        ).sum() - vot_ult.sum()
    ) < 1


def test_different_backends(triangle_data, estimators, weights):
    model = cl.VotingChainladder(
        estimators=estimators,
        weights=weights,
        default_weighting=(1, 2, 3)
    ).fit(
        triangle_data["CumPaidLoss"].sum().set_backend("numpy"),
        sample_weight=triangle_data["EarnedPremDIR"].sum().latest_diagonal.set_backend("numpy"),
    )
    assert (
        abs(
            (
                model.predict(
                    triangle_data["CumPaidLoss"].sum().set_backend("sparse"),
                    sample_weight=triangle_data["EarnedPremDIR"].sum().latest_diagonal.set_backend(
                        "sparse"
                    )
                ).ultimate_.sum()
                - model.ultimate_.sum()
            )
        )
        < 1
    )


def test_weight_broadcasting(triangle_data, estimators, weights):
    mid_dim_weights = np.array([[[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3] * 1)
    max_dim_weights = np.array(mid_dim_weights * 132)

    min_dim_ult = cl.VotingChainladder(estimators=estimators, weights=weights).fit(
        triangle_data['CumPaidLoss'],
        sample_weight=triangle_data["EarnedPremDIR"].latest_diagonal,
    ).ultimate_.sum()
    mid_dim_ult = cl.VotingChainladder(estimators=estimators, weights=mid_dim_weights).fit(
        triangle_data['CumPaidLoss'],
        sample_weight=triangle_data["EarnedPremDIR"].latest_diagonal,
    ).ultimate_.sum()
    max_dim_ult = cl.VotingChainladder(estimators=estimators, weights=max_dim_weights).fit(
        triangle_data['CumPaidLoss'],
        sample_weight=triangle_data["EarnedPremDIR"].latest_diagonal,
    ).ultimate_.sum()
    assert (abs(min_dim_ult - mid_dim_ult - max_dim_ult) < 1)
