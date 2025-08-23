import chainladder as cl
import numpy as np


def test_trend1(clrd):
    tri = clrd[["CumPaidLoss", "EarnedPremDIR"]].sum()
    lhs = (
        cl.CapeCod(0.05)
        .fit(tri["CumPaidLoss"], sample_weight=tri["EarnedPremDIR"].latest_diagonal)
        .ibnr_
    )
    rhs = (
        cl.CapeCod()
        .fit(
            cl.Trend(0.05).fit_transform(tri["CumPaidLoss"]),
            sample_weight=tri["EarnedPremDIR"].latest_diagonal,
        )
        .ibnr_
    )
    assert np.round(lhs, 0) == np.round(rhs, 0)


def test_trend2(raa):
    tri = raa
    assert (
        abs(
            cl.Trend(
                trends=[0.05, 0.05],
                dates=[(None, "1985"), ("1985", None)],
                axis="origin",
            )
            .fit(tri)
            .trend_
            * tri
            - tri.trend(0.05, axis="origin")
        )
        .sum()
        .sum()
        < 1e-6
    )
