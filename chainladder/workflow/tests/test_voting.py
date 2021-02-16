import numpy as np
import chainladder as cl


def test_voting_ultimate():
    clrd = cl.load_sample("clrd")[["CumPaidLoss", "EarnedPremDIR"]]
    clrd = clrd[clrd["LOB"] == "wkcomp"]

    bcl_ult = cl.Chainladder().fit(
        clrd["CumPaidLoss"].sum(),
        ).ultimate_
    bf_ult = cl.BornhuetterFerguson().fit(
        clrd["CumPaidLoss"].sum(),
        sample_weight=clrd["EarnedPremDIR"].sum().latest_diagonal
        ).ultimate_
    cc_ult = cl.CapeCod().fit(
        clrd["CumPaidLoss"].sum(),
        sample_weight=clrd["EarnedPremDIR"].sum().latest_diagonal
        ).ultimate_

    bcl = cl.Chainladder()
    bf = cl.BornhuetterFerguson()
    cc = cl.CapeCod()

    estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]
    weights = np.array([[0.25, 0.25, 0.5]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3)

    vot_ult = cl.VotingChainladder(estimators=estimators, weights=weights).fit(
        clrd["CumPaidLoss"].sum(),
        sample_weight=clrd["EarnedPremDIR"].sum().latest_diagonal,
    ).ultimate_

    weights = weights[..., np.newaxis]

    assert abs(
        (
            bcl_ult * weights[..., 0, :] +
            bf_ult * weights[..., 1, :] +
            cc_ult * weights[..., 2, :]
        ).sum() - vot_ult.sum()
    ) < 1


def test_different_backends():
    clrd = cl.load_sample("clrd")[["CumPaidLoss", "EarnedPremDIR"]]
    clrd = clrd[clrd["LOB"] == "wkcomp"]

    bcl = cl.Chainladder()
    bf = cl.BornhuetterFerguson()
    cc = cl.CapeCod()

    estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]
    weights = np.array([[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3)

    model = cl.VotingChainladder(estimators=estimators, weights=weights).fit(
        clrd["CumPaidLoss"].sum().set_backend("numpy"),
        sample_weight=clrd["EarnedPremDIR"].sum().latest_diagonal.set_backend("numpy"),
    )
    assert (
        abs(
            (
                model.predict(
                    clrd["CumPaidLoss"].sum().set_backend("sparse"),
                    sample_weight=clrd["EarnedPremDIR"].sum().latest_diagonal.set_backend(
                        "sparse"
                    )
                ).ultimate_.sum()
                - model.ultimate_.sum()
            )
        )
        < 1
    )


def test_weight_broadcasting():
    clrd = cl.load_sample("clrd")[["CumPaidLoss", "EarnedPremDIR"]]
    clrd = clrd[clrd["LOB"] == "wkcomp"]

    bcl = cl.Chainladder()
    bf = cl.BornhuetterFerguson()
    cc = cl.CapeCod()

    estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]
    min_dim_weights = np.array([[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3)
    mid_dim_weights = np.array([[[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3] * 1)
    max_dim_weights = np.array([[[[1, 2, 3]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3] * 1] * 132)

    min_dim_ult = cl.VotingChainladder(estimators=estimators, weights=min_dim_weights).fit(
        clrd['CumPaidLoss'],
        sample_weight=clrd["EarnedPremDIR"].latest_diagonal,
    ).ultimate_.sum()
    mid_dim_ult = cl.VotingChainladder(estimators=estimators, weights=mid_dim_weights).fit(
        clrd['CumPaidLoss'],
        sample_weight=clrd["EarnedPremDIR"].latest_diagonal,
    ).ultimate_.sum()
    max_dim_ult = cl.VotingChainladder(estimators=estimators, weights=max_dim_weights).fit(
        clrd['CumPaidLoss'],
        sample_weight=clrd["EarnedPremDIR"].latest_diagonal,
    ).ultimate_.sum()
    assert (abs(min_dim_ult - mid_dim_ult - max_dim_ult) < 1)
