import chainladder as cl
import pandas as pd
import numpy as np
import copy

tri = cl.load_sample("clrd")
qtr = cl.load_sample("quarterly")
raa = cl.load_sample("raa")
tri_gt = copy.deepcopy(tri)
qtr_gt = copy.deepcopy(qtr)
raa_gt = copy.deepcopy(raa)
# Test Triangle slicing
def test_slice_by_boolean():
    assert tri == tri_gt
    assert (
        tri[tri["LOB"] == "ppauto"].loc["Wolverine Mut Ins Co"]["CumPaidLoss"]
        == tri.loc["Wolverine Mut Ins Co"].loc["ppauto"]["CumPaidLoss"]
    )


def test_slice_by_loc():
    assert tri == tri_gt
    assert tri.loc["Aegis Grp"].loc["comauto"].index.iloc[0, 0] == "comauto"


def test_slice_origin():
    assert raa == raa_gt
    assert raa[raa.origin > "1985"].shape == (1, 1, 5, 10)


def test_slice_development():
    assert raa == raa_gt
    assert raa[raa.development < 72].shape == (1, 1, 10, 5)


def test_slice_by_loc_iloc():
    assert tri == tri_gt
    assert tri.groupby("LOB").sum().loc["comauto"].index.iloc[0, 0] == "comauto"


def test_repr():
    assert raa == raa_gt
    np.testing.assert_array_equal(
        pd.read_html(raa._repr_html_())[0].set_index("Unnamed: 0").values,
        raa.to_frame().values,
    )


def test_arithmetic_union():
    assert raa == raa_gt
    assert raa.shape == (raa - raa[raa.valuation < "1987"]).shape


def test_to_frame_unusual():
    assert tri == tri_gt
    a = tri.groupby(["LOB"]).sum().latest_diagonal["CumPaidLoss"].to_frame().values
    b = tri.latest_diagonal["CumPaidLoss"].groupby(["LOB"]).sum().to_frame().values
    np.testing.assert_array_equal(a, b)


def test_link_ratio():
    assert raa == raa_gt
    xp = raa.get_array_module()
    assert (
        xp.sum(
            xp.nan_to_num(raa.link_ratio.values * raa.values[:, :, :-1, :-1])
            - xp.nan_to_num(raa.values[:, :, :-1, 1:])
        )
        < 1e-5
    )


def test_incr_to_cum():
    assert tri == tri_gt
    xp = tri.get_array_module()
    xp.testing.assert_array_equal(tri.cum_to_incr().incr_to_cum().values, tri.values)


def test_create_new_value():
    assert tri == tri_gt
    tri2 = tri.copy()
    tri2["lr"] = tri2["CumPaidLoss"] / tri2["EarnedPremDIR"]
    assert (tri.shape[0], tri.shape[1] + 1, tri.shape[2], tri.shape[3]) == tri2.shape


def test_multilevel_index_groupby_sum1():
    assert tri == tri_gt
    assert tri.groupby("LOB").sum().sum() == tri.sum()


def test_multilevel_index_groupby_sum2():
    assert tri == tri_gt
    a = tri.groupby("GRNAME").sum().sum()
    b = tri.groupby("LOB").sum().sum()
    assert a == b


def test_boolean_groupby_eq_groupby_loc():
    assert tri == tri_gt
    xp = tri.get_array_module()
    xp.testing.assert_array_equal(
        tri[tri["LOB"] == "ppauto"].sum().values,
        tri.groupby("LOB").sum().loc["ppauto"].values,
    )


def test_latest_diagonal_two_routes():
    assert tri == tri_gt
    assert (
        tri.latest_diagonal.sum()["BulkLoss"] == tri.sum().latest_diagonal["BulkLoss"]
    )


def test_sum_of_diff_eq_diff_of_sum():
    assert tri == tri_gt
    assert (tri["BulkLoss"] - tri["CumPaidLoss"]).latest_diagonal == (
        tri.latest_diagonal["BulkLoss"] - tri.latest_diagonal["CumPaidLoss"]
    )


def test_append():
    assert raa == raa_gt
    raa2 = raa.copy()
    raa2.kdims = np.array([["P2"]])
    raa.append(raa2).sum() == raa * 2
    assert raa.append(raa2).sum() == 2 * raa


def test_assign_existing_col():
    qtr = cl.load_sample("quarterly")
    before = qtr.shape
    qtr["paid"] = 1 / qtr["paid"]
    assert qtr.shape == before


def test_arithmetic_across_keys():
    x = cl.load_sample("auto")
    xp = x.get_array_module()
    xp.testing.assert_array_equal((x.sum() - x.iloc[0]).values, x.iloc[1].values)


def test_grain():
    assert qtr == qtr_gt
    actual = qtr.iloc[0, 0].grain("OYDY")
    xp = actual.get_array_module()
    expected = xp.array(
        [
            [
                44.0,
                621.0,
                950.0,
                1020.0,
                1070.0,
                1069.0,
                1089.0,
                1094.0,
                1097.0,
                1099.0,
                1100.0,
                1100.0,
            ],
            [
                42.0,
                541.0,
                1052.0,
                1169.0,
                1238.0,
                1249.0,
                1266.0,
                1269.0,
                1296.0,
                1300.0,
                1300.0,
                xp.nan,
            ],
            [
                17.0,
                530.0,
                966.0,
                1064.0,
                1100.0,
                1128.0,
                1155.0,
                1196.0,
                1201.0,
                1200.0,
                xp.nan,
                xp.nan,
            ],
            [
                10.0,
                393.0,
                935.0,
                1062.0,
                1126.0,
                1209.0,
                1243.0,
                1286.0,
                1298.0,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                13.0,
                481.0,
                1021.0,
                1267.0,
                1400.0,
                1476.0,
                1550.0,
                1583.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                2.0,
                380.0,
                788.0,
                953.0,
                1001.0,
                1030.0,
                1066.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                4.0,
                777.0,
                1063.0,
                1307.0,
                1362.0,
                1411.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                2.0,
                472.0,
                1617.0,
                1818.0,
                1820.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                3.0,
                597.0,
                1092.0,
                1221.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                4.0,
                583.0,
                1212.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                21.0,
                422.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
            [
                13.0,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
                xp.nan,
            ],
        ]
    )
    xp.testing.assert_array_equal(actual.values[0, 0, :, :], expected)


def test_off_cycle_val_date():
    assert qtr == qtr_gt
    assert qtr.valuation_date.strftime("%Y-%m-%d") == "2006-03-31"


def test_printer():
    print(cl.load_sample("abc"))


def test_value_order():
    assert tri == tri_gt
    a = tri[["CumPaidLoss", "BulkLoss"]]
    b = tri[["BulkLoss", "CumPaidLoss"]]
    xp = a.get_array_module()
    xp.testing.assert_array_equal(a.values[:, -1], b.values[:, 0])


def test_trend():
    assert (
        abs(
            (
                cl.load_sample("abc").trend(0.05).trend((1 / 1.05) - 1)
                - cl.load_sample("abc")
            )
            .sum()
            .sum()
        )
        < 1e-5
    )


def test_arithmetic_1():
    x = cl.load_sample("mortgage")
    np.testing.assert_array_equal(-(((x / x) + 0) * x), -(+x))


def test_arithmetic_2():
    x = cl.load_sample("mortgage")
    np.testing.assert_array_equal(1 - (x / x), 0 * x * 0)


def test_rtruediv():
    assert raa == raa_gt
    xp = raa.get_array_module()
    assert xp.nansum(abs(((1 / raa) * raa).values[0, 0] - raa.nan_triangle)) < 0.00001


def test_shift():
    assert qtr == qtr_gt
    x = qtr.iloc[0, 0]
    xp = x.get_array_module()
    xp.testing.assert_array_equal(x[x.valuation <= x.valuation_date].values, x.values)


def test_quantile_vs_median():
    assert tri == tri_gt
    xp = tri.get_array_module()
    xp.testing.assert_array_equal(
        tri.quantile(q=0.5)["CumPaidLoss"].values, tri.median()["CumPaidLoss"].values
    )


def test_grain_returns_valid_tri():
    assert qtr == qtr_gt
    assert qtr.grain("OYDY").latest_diagonal == qtr.latest_diagonal


def test_base_minimum_exposure_triangle():
    assert raa == raa_gt
    d = (raa.latest_diagonal * 0 + 50000).to_frame().reset_index()
    d["index"] = d["index"].astype(str)
    cl.Triangle(d, origin="index", columns=d.columns[-1])


def test_origin_and_value_setters():
    assert raa == raa_gt
    raa2 = raa.copy()
    raa.columns = list(raa.columns)
    raa.origin = list(raa.origin)
    assert np.all(
        (
            np.all(raa2.origin == raa.origin),
            np.all(raa2.development == raa.development),
            np.all(raa2.odims == raa.odims),
            np.all(raa2.vdims == raa.vdims),
        )
    )


def test_grain_increm_arg():
    assert qtr == qtr_gt
    tri_i = qtr["incurred"].cum_to_incr()
    a = tri_i.grain("OYDY").incr_to_cum()
    assert a == qtr["incurred"].grain("OYDY").set_backend(a.array_backend)


def test_valdev1():
    assert qtr == qtr_gt
    a = qtr.dev_to_val().val_to_dev()
    b = qtr
    assert a == b


def test_valdev2():
    assert qtr == qtr_gt
    a = qtr.dev_to_val().grain("OYDY").val_to_dev()
    b = qtr.grain("OYDY")
    assert a == b


def test_valdev3():
    assert qtr == qtr_gt
    a = qtr.grain("OYDY").dev_to_val().val_to_dev()
    b = qtr.grain("OYDY")
    assert a == b


# def test_valdev4():
#    # Does not work with pandas 0.23, consider requiring only pandas>=0.24
#    raa = raa
#    np.testing.assert_array_equal(raa.dev_to_val()[raa.dev_to_val().development>='1989'].values,
#        raa[raa.valuation>='1989'].dev_to_val().values)


def test_valdev5():
    assert raa == raa_gt
    xp = raa.get_array_module()
    xp.testing.assert_array_equal(
        raa[raa.valuation >= "1989"].latest_diagonal.values, raa.latest_diagonal.values
    )


def test_valdev6():
    assert raa == raa_gt
    xp = raa.get_array_module()
    xp.testing.assert_array_equal(
        raa.grain("OYDY").latest_diagonal.values,
        raa.latest_diagonal.grain("OYDY").values,
    )


def test_valdev7():
    assert qtr == qtr_gt
    xp = qtr.get_array_module()
    x = cl.Chainladder().fit(qtr).full_expectation_
    assert xp.sum(x.dev_to_val().val_to_dev().values - x.values) < 1e-5


def test_reassignment():
    assert tri == tri_gt
    clrd = tri.copy()
    clrd["values"] = clrd["CumPaidLoss"]
    clrd["values"] = clrd["values"] + clrd["CumPaidLoss"]


def test_dropna():
    assert tri == tri_gt
    assert tri.shape == tri.dropna().shape
    a = tri[tri["LOB"] == "wkcomp"].iloc[-5]["CumPaidLoss"].dropna().shape
    assert a == (1, 1, 2, 2)


def test_commutative():
    assert qtr == qtr_gt
    xp = qtr.get_array_module()
    full = cl.Chainladder().fit(qtr).full_expectation_
    assert qtr.grain("OYDY").val_to_dev() == qtr.val_to_dev().grain("OYDY")
    assert qtr.cum_to_incr().grain(
        "OYDY"
    ).val_to_dev() == qtr.val_to_dev().cum_to_incr().grain("OYDY")
    assert qtr.grain(
        "OYDY"
    ).cum_to_incr().val_to_dev().incr_to_cum() == qtr.val_to_dev().grain("OYDY")
    assert full.grain("OYDY").val_to_dev() == full.val_to_dev().grain("OYDY")
    assert full.cum_to_incr().grain(
        "OYDY"
    ).val_to_dev() == full.val_to_dev().cum_to_incr().grain("OYDY")
    a = full.grain("OYDY").cum_to_incr().val_to_dev().incr_to_cum()
    b = full.val_to_dev().grain("OYDY")
    assert abs(a - b).max().max().max() < 1e-5


def test_broadcasting():
    assert tri == tri_gt
    assert raa == raa_gt
    t1 = raa
    t2 = tri
    assert t1.broadcast_axis("columns", t2.columns).shape[1] == t2.shape[1]
    assert t1.broadcast_axis("index", t2.index).shape[0] == t2.shape[0]
    raa.latest_diagonal.to_frame()


def test_slicers_honor_order():
    assert tri == tri_gt
    clrd = tri.groupby("LOB").sum()
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


def test_exposure_tri():
    x = cl.load_sample("auto")
    x = x[x.development == 12]
    x = x["paid"].to_frame().T.unstack().reset_index()
    x.columns = ["LOB", "origin", "paid"]
    x.origin = x.origin.astype(str)
    y = cl.Triangle(x, origin="origin", index="LOB", columns="paid")
    x = cl.load_sample("auto")["paid"]
    x = x[x.development == 12]
    assert x == y


def test_jagged_1_add():
    assert raa == raa_gt
    raa1 = raa[raa.origin <= "1984"]
    raa2 = raa[raa.origin > "1984"]
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa


def test_jagged_2_add():
    assert raa == raa_gt
    raa1 = raa[raa.development <= 48]
    raa2 = raa[raa.development > 48]
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa


def test_df_period_input():
    assert raa == raa_gt
    d = raa.latest_diagonal
    df = d.to_frame().reset_index()
    assert cl.Triangle(df, origin="index", columns=df.columns[-1]) == d


def test_trend_on_vector():
    assert raa == raa_gt
    d = raa.latest_diagonal
    assert d.trend(0.05, axis=2).to_frame().astype(int).iloc[0, 0] == 29217


def test_latest_diagonal_val_to_dev():
    assert raa == raa_gt
    assert raa.latest_diagonal.val_to_dev() == raa[raa.valuation == raa.valuation_date]


def test_vector_division():
    assert raa == raa_gt
    raa.latest_diagonal / raa


def test_sumdiff_to_diffsum():
    assert tri == tri_gt
    out = tri["CumPaidLoss"]
    assert out.cum_to_incr().incr_to_cum().sum() == out.sum()


def test_multiindex_broadcast():
    assert tri == tri_gt
    clrd = tri["CumPaidLoss"]
    clrd / clrd.groupby("LOB").sum()


def test_backends():
    assert tri == tri_gt
    clrd = tri[["CumPaidLoss", "EarnedPremDIR"]]
    a = clrd.iloc[1, 0].set_backend("sparse").dropna()
    b = clrd.iloc[1, 0].dropna()
    assert a == b


def test_union_columns():
    assert tri == tri_gt
    assert tri.iloc[:, :3] + tri.iloc[:, 3:] == tri


def test_4loc():
    assert tri == tri_gt
    clrd = tri.groupby("LOB").sum()
    assert (
        clrd.iloc[:3, :2, 0, 0]
        == clrd[clrd.origin == tri.origin.min()][
            clrd.development == clrd.development.min()
        ].loc["comauto":"othliab", :"CumPaidLoss", :, :]
    )
    assert (
        clrd.iloc[:3, :2, 0:1, -1]
        == clrd[clrd.development == tri.development.max()].loc[
            "comauto":"othliab", :"CumPaidLoss", "1988", :
        ]
    )


def test_init_vector():
    assert raa == raa_gt
    a = raa[raa.development == 12]
    b = pd.DataFrame(
        {"AccYear": [item for item in range(1981, 1991)], "premium": [3000000] * 10}
    )
    b = cl.Triangle(b, origin="AccYear", columns="premium")
    assert np.all(a.valuation == b.valuation)
    assert a.valuation_date == b.valuation_date


def test_loc_ellipsis():
    assert tri == tri_gt
    assert (
        tri.loc["Aegis Grp"] == tri.loc["Adriatic Ins Co":"Aegis Grp"].loc["Aegis Grp"]
    )
    assert tri.loc["Aegis Grp", ..., :] == tri.loc["Aegis Grp"]
    assert tri.loc[..., 24:] == tri.loc[..., :, 24:]
    assert tri.loc[:, ..., 24:] == tri.loc[..., :, 24:]
    assert tri.loc[:, "CumPaidLoss"] == tri.loc[:, "CumPaidLoss", ...]
    assert tri.loc[..., "CumPaidLoss", :, :] == tri.loc[:, "CumPaidLoss", :, :]


def missing_first_lag():
    x = raa.copy()
    x.values[:, :, :, 0] = 0
    x = x.sum(0)
    x.link_ratio.shape == (1, 1, 9, 9)


def test_reverse_slice_integrity():
    assert tri == tri_gt
    assert tri.iloc[::-1, ::-1].shape == tri.shape
    assert np.all(tri.iloc[:, ::-1].columns.values == tri.columns[::-1])
    assert tri.iloc[tri.index.index[::-1]] + tri == 2 * tri


def test_loc_tuple():
    assert tri == tri_gt
    assert len(tri.loc[("Adriatic Ins Co", "othliab")]) == 1
    assert tri.loc[tri.index] == tri


def test_index_broadcasting():
    assert tri == tri_gt
    assert ((tri / tri.sum()) - ((1 / tri.sum()) * tri)).sum().sum().sum() < 1e-4


def test_groupby_axis1():
    assert tri == tri_gt
    clrd = tri.sum("origin").sum("development")
    groups = [i.find("Loss") >= 0 for i in clrd.columns]
    assert np.all(
        clrd.to_frame().groupby(groups, axis=1).sum()
        == clrd.groupby(groups, axis=1).sum().to_frame()
    )
    assert np.all(
        clrd.to_frame().groupby("LOB").sum() == clrd.groupby("LOB").sum().to_frame()
    )
    cl.load_sample("prism").sum().grain("OYDY")


def test_different_forms_of_grain():
    t = cl.load_sample("prism").sum()["Paid"]
    assert (
        abs(t.grain("OYDY") - t.incr_to_cum().grain("OYDY").cum_to_incr()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.grain("OYDY", trailing=True)
            - t.incr_to_cum().grain("OYDY", trailing=True).cum_to_incr()
        )
        .sum()
        .sum()
        < 1e-4
    )
    assert (
        abs(t.incr_to_cum().grain("OYDY") - t.grain("OYDY").incr_to_cum()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.incr_to_cum().grain("OYDY", trailing=True)
            - t.grain("OYDY", trailing=True).incr_to_cum()
        )
        .sum()
        .sum()
        < 1e-4
    )
    assert (
        abs(t.grain("OYDQ") - t.incr_to_cum().grain("OYDQ").cum_to_incr()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.grain("OYDQ", trailing=True)
            - t.incr_to_cum().grain("OYDQ", trailing=True).cum_to_incr()
        )
        .sum()
        .sum()
        < 1e-4
    )
    assert (
        abs(t.incr_to_cum().grain("OQDM") - t.grain("OQDM").incr_to_cum()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.incr_to_cum().grain("OQDM", trailing=True)
            - t.grain("OQDM", trailing=True).incr_to_cum()
        )
        .sum()
        .sum()
        < 1e-4
    )
    t = t.dev_to_val()
    assert (
        abs(t.grain("OYDY") - t.incr_to_cum().grain("OYDY").cum_to_incr()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.grain("OYDY", trailing=True)
            - t.incr_to_cum().grain("OYDY", trailing=True).cum_to_incr()
        )
        .sum()
        .sum()
        < 1e-4
    )
    assert (
        abs(t.incr_to_cum().grain("OYDY") - t.grain("OYDY").incr_to_cum()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.incr_to_cum().grain("OYDY", trailing=True)
            - t.grain("OYDY", trailing=True).incr_to_cum()
        )
        .sum()
        .sum()
        < 1e-4
    )
    t = t.val_to_dev()
    t = t[t.valuation < "2017-09"]
    assert (
        abs(t.grain("OYDY") - t.incr_to_cum().grain("OYDY").cum_to_incr()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.grain("OYDY", trailing=True)
            - t.incr_to_cum().grain("OYDY", trailing=True).cum_to_incr()
        )
        .sum()
        .sum()
        < 1e-4
    )
    assert (
        abs(t.incr_to_cum().grain("OYDY") - t.grain("OYDY").incr_to_cum()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.incr_to_cum().grain("OYDY", trailing=True)
            - t.grain("OYDY", trailing=True).incr_to_cum()
        )
        .sum()
        .sum()
        < 1e-4
    )
    t = t.dev_to_val()
    assert (
        abs(t.grain("OYDY") - t.incr_to_cum().grain("OYDY").cum_to_incr()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.grain("OYDY", trailing=True)
            - t.incr_to_cum().grain("OYDY", trailing=True).cum_to_incr()
        )
        .sum()
        .sum()
        < 1e-4
    )
    assert (
        abs(t.incr_to_cum().grain("OYDY") - t.grain("OYDY").incr_to_cum()).sum().sum()
        < 1e-4
    )
    assert (
        abs(
            t.incr_to_cum().grain("OYDY", trailing=True)
            - t.grain("OYDY", trailing=True).incr_to_cum()
        )
        .sum()
        .sum()
        < 1e-4
    )


def test_partial_year():
    before = cl.load_sample('prism')['Paid'].sum().incr_to_cum()
    before=before[before.valuation<='2017-08'].latest_diagonal

    after = cl.Triangle(
        before.to_frame(keepdims=True).reset_index(),
        origin='origin', development='valuation', columns='Paid', index=before.key_labels)

    assert after.valuation_date == before.valuation_date


def test_at_iat():
    raa1 = cl.load_sample('raa')
    raa2 = cl.load_sample('raa')
    raa1.at['Total','values', '1985', 120] = 5
    raa1.at['Total','values', '1985', 12] = 5
    raa2.iat[0, 0, 4, -1] = 5
    raa2.iat[-1, -1, 4, 0] = 5
    assert raa1 == raa2


def test_at_iat_sparse():
    raa1 = cl.load_sample('raa').set_backend('sparse')
    raa2 = cl.load_sample('raa').set_backend('sparse')
    raa1.at['Total','values', '1985', 120] = 5
    raa1.at['Total','values', '1985', 12] = 5
    raa2.iat[0, 0, 4, -1] = 5
    raa2.iat[-1, -1, 4, 0] = 5
    assert raa1 == raa2

def test_array_protocol():
    assert np.sqrt(raa) == raa.sqrt()
    assert np.concatenate((tri.iloc[:200], tri.iloc[200:]),0) == cl.concat((tri.iloc[:200], tri.iloc[200:]),0)

def test_dask_backend():
    raa1 = cl.Chainladder().fit(cl.load_sample('raa').set_backend('dask')).ultimate_
    raa2 = cl.Chainladder().fit(cl.load_sample('raa')).ultimate_
    assert (raa1 == raa2).compute()
