
import numpy as np
import pytest


def test_slice_by_boolean(clrd):
    assert (
        clrd[clrd["LOB"] == "ppauto"].loc["Wolverine Mut Ins Co"]["CumPaidLoss"]
        == clrd.loc["Wolverine Mut Ins Co"].loc["ppauto"]["CumPaidLoss"])


def test_slice_by_loc(clrd):
    assert clrd.loc["Aegis Grp"].loc["comauto"].index.iloc[0, 0] == "comauto"


def test_slice_origin(raa):
    assert raa[raa.origin > "1985"].shape == (1, 1, 5, 10)
    raa.loc[..., raa.origin<='1994', :]


def test_slice_development(raa):
    assert raa[raa.development < 72].shape == (1, 1, 10, 5)
    raa.loc[..., 24:]


def test_slice_by_loc_iloc(clrd):
    assert clrd.groupby("LOB").sum().loc["comauto"].index.iloc[0, 0] == "comauto"
    assert len(clrd.loc[clrd.index.iloc[150]]) == 1


def test_slicers_honor_order(clrd):
    clrd = clrd.groupby("LOB").sum()
    assert clrd.iloc[[1, 0], :].iloc[0, 1] == clrd.iloc[1, 1]  # row
    assert clrd.iloc[[1, 0], [1, 0]].iloc[0, 0] == clrd.iloc[1, 1]  # col
    assert clrd.loc[:, ["CumPaidLoss", "IncurLoss"]].iloc[0, 0] == clrd.iloc[0, 1]
    assert (
        clrd.loc[["ppauto", "medmal"], ["CumPaidLoss", "IncurLoss"]].iloc[0, 0]
        == clrd.iloc[3]["CumPaidLoss"]
    )
    assert (
        clrd.loc[clrd["LOB"] == "comauto", ["CumPaidLoss", "IncurLoss"]]
        == clrd[clrd["LOB"] == "comauto"].iloc[:, [1, 0]]
    )
    assert clrd.groupby("LOB").sum() == clrd


def test_backends(clrd):
    clrd = clrd[["CumPaidLoss", "EarnedPremDIR"]]
    a = clrd.iloc[1, 0].set_backend("sparse").dropna()
    b = clrd.iloc[1, 0].dropna()
    assert a == b


def test_union_columns(clrd):
    assert clrd.iloc[:, :3] + clrd.iloc[:, 3:] == clrd


def test_4loc(clrd):
    clrd = clrd.groupby("LOB").sum()
    assert (
        clrd.iloc[:3, :2, 0, 0]
        == clrd[clrd.origin == clrd.origin.min()][
            clrd.development == clrd.development.min()
        ].loc["comauto":"othliab", :"CumPaidLoss", :, :]
    )
    assert (
        clrd.iloc[:3, :2, 0:1, -1]
        == clrd[clrd.development == clrd.development.max()].loc[
            "comauto":"othliab", :"CumPaidLoss", "1988", :
        ]
    )


def test_loc_ellipsis(clrd):
    assert (
        clrd.loc["Aegis Grp"] == clrd.loc["Adriatic Ins Co":"Aegis Grp"].loc["Aegis Grp"]
    )
    assert clrd.loc["Aegis Grp", ..., :] == clrd.loc["Aegis Grp"]
    assert clrd.loc[..., 24:] == clrd.loc[..., :, 24:]
    assert clrd.loc[:, ..., 24:] == clrd.loc[..., :, 24:]
    assert clrd.loc[:, "CumPaidLoss"] == clrd.loc[:, "CumPaidLoss", ...]
    assert clrd.loc[..., "CumPaidLoss", :, :] == clrd.loc[:, "CumPaidLoss", :, :]


def test_missing_first_lag(raa):
    x = raa.copy()
    x.values[:, :, :, 0] = 0
    x = x.sum(0)
    x.link_ratio.shape == (1, 1, 9, 9)


def test_reverse_slice_integrity(clrd):
    assert clrd.iloc[::-1, ::-1].shape == clrd.shape
    assert np.all(clrd.iloc[:, ::-1].columns.values == clrd.columns[::-1])
    assert clrd.iloc[clrd.index.index[::-1]] + clrd == 2 * clrd


def test_loc_tuple(clrd):
    assert len(clrd.loc[("Adriatic Ins Co", "othliab")]) == 1
    assert clrd.loc[clrd.index] == clrd


def test_at_iat(raa):
    raa1 = raa.copy()
    raa2 = raa.copy()
    raa1.at['Total','values', '1985', 120]
    raa1.at['Total','values', '1985', 120] = 5
    raa1.at['Total','values', '1985', 12] = 5
    raa2.iat[0, 0, 4, -1] = 5
    raa2.iat[0, 0, 4, -1]
    raa2.iat[-1, -1, 4, 0] = 5
    assert raa1 == raa2


@pytest.mark.xfail
def test_sparse_at_iat(prism):
    prism.iloc[0, 0, 0, 0] = 1.0


def test_sparse_at_iat1(prism):
    t = prism.copy()
    t.iat[0, 0, 0, 0] = 5.0
    t.at[12138, 'reportedCount', '2008-01', 1] = 0
    assert t == prism


def test_sparse_column_assignment(prism):
    t = prism.copy()
    out = t['Paid']
    t['Paid2'] = t['Paid'] # New from physical
    t['Paid2'] = lambda x: x['Paid'] # Existing from virtual
    t['Paid'] = t['Paid2'] # Existing from physical
    t['Paid3'] = lambda t: t['Paid'] # New from virtual
    assert out == t['Paid']
    assert t.shape == (34244, 6, 120, 120)
