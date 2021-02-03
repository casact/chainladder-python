import chainladder as cl
import pandas as pd


def test_parallelogram():
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

    X = cl.load_sample("clrd")["EarnedPremNet"].latest_diagonal
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
