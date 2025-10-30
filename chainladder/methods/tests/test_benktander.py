import pytest
import numpy as np
import chainladder as cl
import pandas as pd

@pytest.fixture
def atol():
    return 1e-5


data = ["RAA", "ABC", "GenIns", "MW2008", "MW2014"]


def test_bk_fit_weight():
    """
    Test validation of sample_weight requirement. Should raise a value error if no weight is supplied.
    """
    raa = cl.load_sample("RAA")
    with pytest.raises(ValueError):
        cl.Benktander().fit(raa)

@pytest.mark.parametrize("data", data)
def test_benktander_to_chainladder(data, atol):
    tri = cl.load_sample(data)
    a = cl.Chainladder().fit(tri).ibnr_
    b = cl.Benktander(apriori=0.8, n_iters=255).fit(tri, sample_weight=a).ibnr_
    xp = tri.get_array_module()
    assert xp.allclose(xp.nan_to_num(a.values), xp.nan_to_num(b.values), atol=atol)


def test_bf_eq_cl_when_using_cl_apriori(qtr):
    cl_ult = cl.Chainladder().fit(qtr).ultimate_
    bf_ult = cl.BornhuetterFerguson().fit(qtr, sample_weight=cl_ult).ultimate_
    xp = cl_ult.get_array_module()
    assert xp.allclose(cl_ult.values, bf_ult.values, atol=1e-5)


def test_different_backends(clrd):
    clrd = clrd[["CumPaidLoss", "EarnedPremDIR"]]
    clrd = clrd[clrd["LOB"] == "wkcomp"]
    model = cl.BornhuetterFerguson().fit(
        clrd["CumPaidLoss"].sum().set_backend("numpy"),
        sample_weight=clrd["EarnedPremDIR"].sum().latest_diagonal.set_backend("numpy"),
    )
    assert (
        abs(
            (
                model.predict(
                    clrd["CumPaidLoss"].set_backend("sparse"),
                    sample_weight=clrd["EarnedPremDIR"].latest_diagonal.set_backend(
                        "sparse"
                    ),
                ).ibnr_.sum()
                - model.ibnr_
            ).sum()
        )
        < 1
    )

def test_odd_shaped_triangle():
    df = pd.DataFrame({
        "claim_year": 2000 + pd.Series([0] * 8 + [1] * 4),
        "claim_month": [1, 4, 7, 10] * 3,
        "dev_year": 2000 + pd.Series([0] * 4 + [1] * 8),
        "dev_month": [1, 4, 7, 10] * 3,
        "payment": [1] * 12,
    })
    tr = cl.Triangle(
        df,
        origin=["claim_year", "claim_month"],
        development=["dev_year", "dev_month"],
        columns="payment",
        cumulative=False,
    )
    atr = tr.grain("OYDQ")
    ult1 = cl.Benktander(apriori = 1,n_iters=10000).fit(cl.Development(average="volume").fit_transform(atr),sample_weight = atr.latest_diagonal).ultimate_.sum()
    ult2 = cl.Benktander(apriori = 1,n_iters=10000).fit(cl.Development(average="volume").fit_transform(tr),sample_weight = tr.latest_diagonal).ultimate_.grain("OYDQ").sum()
    assert abs(ult1 - ult2) < 1e-5