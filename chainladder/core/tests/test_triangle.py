import chainladder as cl
import pandas as pd
import numpy as np
import pytest
import io
from datetime import datetime

try:
    from IPython.core.display import HTML
except:
    HTML = None


def test_repr(raa):
    np.testing.assert_array_equal(
        pd.read_html(raa._repr_html_())[0].set_index("Unnamed: 0").values,
        raa.to_frame(origin_as_datetime=False).values,
    )


def test_to_frame_unusual(clrd):
    a = (
        clrd.groupby(["LOB"])
        .sum()
        .latest_diagonal["CumPaidLoss"]
        .to_frame(origin_as_datetime=False)
    )
    b = (
        clrd.latest_diagonal["CumPaidLoss"]
        .groupby(["LOB"])
        .sum()
        .to_frame(origin_as_datetime=False)
    )
    assert (a == b).all().all()


def test_link_ratio(raa, atol):
    assert (
        raa.link_ratio * raa.iloc[:, :, :-1, :-1].values - raa.values[:, :, :-1, 1:]
    ).sum().sum() < atol


def test_incr_to_cum(clrd):
    clrd.cum_to_incr().incr_to_cum() == clrd


def test_create_new_value(clrd):
    clrd2 = clrd.copy()
    clrd2["lr"] = clrd2["CumPaidLoss"] / clrd2["EarnedPremDIR"]
    assert (
        clrd.shape[0],
        clrd.shape[1] + 1,
        clrd.shape[2],
        clrd.shape[3],
    ) == clrd2.shape


def test_multilevel_index_groupby_sum1(clrd):
    assert clrd.groupby("LOB").sum().sum() == clrd.sum()


def test_multilevel_index_groupby_sum2(clrd):
    a = clrd.groupby("GRNAME").sum().sum()
    b = clrd.groupby("LOB").sum().sum()
    assert a == b


def test_boolean_groupby_eq_groupby_loc(clrd):
    assert (
        clrd[clrd["LOB"] == "ppauto"].sum() == clrd.groupby("LOB").sum().loc["ppauto"]
    )


def test_latest_diagonal_two_routes(clrd):
    assert (
        clrd.latest_diagonal.sum()["BulkLoss"] == clrd.sum().latest_diagonal["BulkLoss"]
    )


def test_sum_of_diff_eq_diff_of_sum(clrd):
    assert (clrd["BulkLoss"] - clrd["CumPaidLoss"]).latest_diagonal == (
        clrd.latest_diagonal["BulkLoss"] - clrd.latest_diagonal["CumPaidLoss"]
    )


def test_append(raa):
    raa2 = raa.copy()
    raa2.kdims = np.array([["P2"]])
    raa.append(raa2).sum() == raa * 2
    assert raa.append(raa2).sum() == 2 * raa

def test_rename_columns(genins, clrd) -> None:
    """
    Test the renaming of triangle columns.
    """
    # Scalar case - single column triangle.
    genins.rename('columns', 'foo')

    assert genins.columns.to_list() == ['foo']

    # Test the cascading of rename to triangle.columns_label.
    assert genins.columns_label == ['foo']

def test_rename_exception(genins, clrd) -> None:
    # Test incorrect axis argument - misspelling of string.
    with pytest.raises(ValueError):
        genins.rename('colunms', 'foo')

    # Test incorrect axis integer.
    with pytest.raises(ValueError):
        genins.rename(4, 'foo')

    # Test incorrect number of columns.
    with pytest.raises(ValueError):
        clrd.rename('columns', ['foo'])

def test_assign_existing_col(qtr):
    out = qtr.copy()
    before = out.shape
    out["paid"] = 1 / out["paid"]
    assert out.shape == before


def test_off_cycle_val_date(qtr):
    assert qtr.valuation_date.strftime("%Y-%m-%d") == "2006-03-31"


def test_printer(raa):
    print(raa)


def test_value_order(clrd):
    a = clrd[["CumPaidLoss", "BulkLoss"]]
    b = clrd[["BulkLoss", "CumPaidLoss"]]
    xp = a.get_array_module()
    xp.testing.assert_array_equal(a.values[:, -1], b.values[:, 0])


def test_trend(raa, atol):
    assert abs((raa.trend(0.05).trend((1 / 1.05) - 1) - raa).sum().sum()) < 1e-5


def test_shift(qtr):
    x = qtr.iloc[0, 0]
    xp = x.get_array_module()
    xp.testing.assert_array_equal(x[x.valuation <= x.valuation_date].values, x.values)


def test_quantile_vs_median(clrd):
    xp = clrd.get_array_module()
    xp.testing.assert_array_equal(
        clrd.quantile(q=0.5)["CumPaidLoss"].values, clrd.median()["CumPaidLoss"].values
    )


def test_base_minimum_exposure_triangle(raa):
    d = (
        (raa.latest_diagonal * 0 + 50000)
        .to_frame(origin_as_datetime=False)
        .reset_index()
    )
    d["index"] = d["index"].astype(str)
    cl.Triangle(d, origin="index", columns=d.columns[-1])


def test_origin_and_value_setters(raa):
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


def test_valdev1(qtr):
    assert qtr.dev_to_val().val_to_dev() == qtr


def test_valdev2(qtr):
    a = qtr.dev_to_val().grain("OYDY").val_to_dev()
    b = qtr.grain("OYDY")
    assert a == b


def test_valdev3(qtr):
    a = qtr.grain("OYDY").dev_to_val().val_to_dev()
    b = qtr.grain("OYDY")
    assert a == b


# def test_valdev4():
#    # Does not work with pandas 0.23, consider requiring only pandas>=0.24
#    raa = raa
#    np.testing.assert_array_equal(raa.dev_to_val()[raa.dev_to_val().development>='1989'].values,
#        raa[raa.valuation>='1989'].dev_to_val().values)


def test_valdev5(raa):
    xp = raa.get_array_module()
    xp.testing.assert_array_equal(
        raa[raa.valuation >= "1989"].latest_diagonal.values, raa.latest_diagonal.values
    )


def test_valdev6(raa):
    xp = raa.get_array_module()
    xp.testing.assert_array_equal(
        raa.grain("OYDY").latest_diagonal.set_backend("numpy").values,
        raa.latest_diagonal.grain("OYDY").set_backend("numpy").values,
    )


def test_valdev7(qtr, atol):
    xp = qtr.get_array_module()
    x = cl.Chainladder().fit(qtr).full_expectation_
    assert xp.sum(x.dev_to_val().val_to_dev().values - x.values) < atol


def test_reassignment(clrd):
    clrd = clrd.copy()
    clrd["values"] = clrd["CumPaidLoss"]
    clrd["values"] = clrd["values"] + clrd["CumPaidLoss"]


def test_dropna(clrd):
    assert clrd.shape == clrd.dropna().shape
    a = clrd[clrd["LOB"] == "wkcomp"].iloc[-5]["CumPaidLoss"].dropna().shape
    assert a == (1, 1, 2, 2)


def test_exposure_tri():
    x = cl.load_sample("auto")
    x = x[x.development == 12]
    x = x["paid"].to_frame(origin_as_datetime=False).T.unstack().reset_index()
    x.columns = ["LOB", "origin", "paid"]
    x.origin = x.origin.astype(str)
    y = cl.Triangle(x, origin="origin", index="LOB", columns="paid")
    x = cl.load_sample("auto")["paid"]
    x = x[x.development == 12]
    assert x == y


def test_jagged_1_add(raa):
    raa1 = raa[raa.origin <= "1984"]
    raa2 = raa[raa.origin > "1984"]
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa


def test_jagged_2_add(raa):
    raa1 = raa[raa.development <= 48]
    raa2 = raa[raa.development > 48]
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa


def test_df_period_input(raa):
    d = raa.latest_diagonal
    df = d.to_frame(origin_as_datetime=False).reset_index()
    assert cl.Triangle(df, origin="index", columns=df.columns[-1]) == d


def test_trend_on_vector(raa):
    d = raa.latest_diagonal
    assert (
        d.trend(0.05, axis=2).to_frame(origin_as_datetime=False).astype(int).iloc[0, 0]
        == 29217
    )


def test_latest_diagonal_val_to_dev(raa):
    assert raa.latest_diagonal.val_to_dev() == raa[raa.valuation == raa.valuation_date]


def test_sumdiff_to_diffsum(clrd):
    out = clrd["CumPaidLoss"]
    assert out.cum_to_incr().incr_to_cum().sum() == out.sum()


def test_init_vector(raa):
    a = raa.latest_diagonal
    b = pd.DataFrame(
        {"AccYear": [item for item in range(1981, 1991)], "premium": [3000000] * 10}
    )
    b = cl.Triangle(b, origin="AccYear", columns="premium")
    assert np.all(a.valuation == b.valuation)
    assert a.valuation_date == b.valuation_date


def test_groupby_axis1(clrd, prism):
    clrd = clrd.sum("origin").sum("development")
    groups = [i.find("Loss") >= 0 for i in clrd.columns]
    assert np.all(
        clrd.to_frame(origin_as_datetime=False).groupby(groups, axis=1).sum()
        == clrd.groupby(groups, axis=1).sum().to_frame(origin_as_datetime=False)
    )
    assert np.all(
        clrd.to_frame(origin_as_datetime=False).groupby("LOB").sum()
        == clrd.groupby("LOB").sum().to_frame(origin_as_datetime=False)
    )
    prism.sum().grain("OYDY")


def test_partial_year(prism):
    before = prism["Paid"].sum().incr_to_cum()
    before = before[before.valuation <= "2017-08"].latest_diagonal

    after = cl.Triangle(
        before.to_frame(keepdims=True, origin_as_datetime=True).reset_index(),
        origin="origin",
        development="valuation",
        columns="Paid",
        index=before.key_labels,
    )

    assert after.valuation_date == before.valuation_date


def test_array_protocol(raa, clrd):
    assert np.sqrt(raa) == raa.sqrt()
    assert np.concatenate((clrd.iloc[:200], clrd.iloc[200:]), 0) == cl.concat(
        (clrd.iloc[:200], clrd.iloc[200:]), 0
    )


# def test_dask_backend(raa):
#     """ Dask backend not fully implemented """
#    raa1 = cl.Chainladder().fit(raa.set_backend('dask')).ultimate_
#    raa2 = cl.Chainladder().fit(raa).ultimate_
#    assert (raa1 == raa2).compute()


def test_partial_val_dev(raa):
    raa = raa.latest_diagonal
    raa.iloc[..., -3:, :] = np.nan
    raa.val_to_dev().iloc[0, 0, 0, -1] == raa.iloc[0, 0, 0, -1]


def test_sort_axis(clrd):
    assert clrd.iloc[::-1, ::-1, ::-1, ::-1].sort_axis(0).sort_axis(1).sort_axis(
        2
    ).sort_axis(3) == clrd.sort_axis(1)


def test_shift(raa):
    assert (
        raa.iloc[..., 1:-1, 1:-1]
        - raa.shift(-1, axis=2)
        .shift(-1, axis=3)
        .shift(2, axis=2)
        .shift(2, axis=3)
        .dropna()
        .values
    ).to_frame(origin_as_datetime=False).fillna(0).sum().sum() == 0


def test_array_protocol2(raa):
    import numpy as np

    assert raa.log().exp() == np.exp(np.log(raa))


def test_create_full_triangle(raa):
    a = cl.Chainladder().fit(raa).full_triangle_
    b = cl.Triangle(
        a.to_frame(keepdims=True, implicit_axis=True, origin_as_datetime=True),
        origin="origin",
        development="valuation",
        columns="values",
    )
    assert a == b


def test_groupby_getitem(clrd):
    assert (
        clrd.groupby("LOB")["CumPaidLoss"].sum()
        == clrd["CumPaidLoss"].groupby("LOB").sum()
    )


def test_virtual_column(prism):
    prism["P"] = prism["Paid"]
    prism["Paid"] = lambda x: x["P"]
    assert prism["Paid"] == prism["P"]


def test_correct_valutaion(raa):
    new = cl.Triangle(
        raa.iloc[..., :-3, :].latest_diagonal.to_frame(
            keepdims=True, implicit_axis=True, origin_as_datetime=True
        ),
        origin="origin",
        development="valuation",
        columns="values",
    )
    assert new.valuation_date == raa.valuation_date


@pytest.mark.xfail
@pytest.mark.parametrize(
    "prop", ["cdf_", "ibnr_", "full_expectation_", "full_triangle_"]
)
def test_no_fitted(raa, prop):
    getattr(raa, prop)


def test_pipe(raa):
    def f(x):
        return x.loc[..., 48:]

    assert raa.loc[..., 48:] == raa.pipe(f)


def test_repr_html(raa, clrd):
    assert type(raa._repr_html_()) == str
    assert type(clrd._repr_html_()) == str


@pytest.mark.xfail(HTML is None, reason="ipython needed for test")
def test_heatmap(raa):
    raa.link_ratio.heatmap()


def test_agg_sparse():
    a = cl.load_sample("raa")
    b = cl.load_sample("raa").set_backend("sparse")
    assert a.mean().mean() == b.mean().mean()


def test_inplace(raa):
    t = raa.copy()
    t.dev_to_val(inplace=True)
    t.val_to_dev(inplace=True)
    t.grain("OYDY", inplace=True)


def test_malformed_init():
    assert (
        cl.Triangle(
            data=pd.DataFrame(
                {
                    "Accident Date": [
                        "2020-07-23",
                        "2019-07-23",
                        "2018-07-23",
                        "2016-07-23",
                        "2020-08-23",
                        "2019-09-23",
                        "2018-10-23",
                    ],
                    "Valuation Date": [
                        "2021-01-01",
                        "2021-01-01",
                        "2021-01-01",
                        "2021-01-01",
                        "2021-01-01",
                        "2021-01-01",
                        "2021-01-01",
                    ],
                    "Loss": [10000, 10000, 10000, 10000, 0, 0, 0],
                }
            ),
            origin="Accident Date",
            development="Valuation Date",
            columns="Loss",
        ).origin_grain
        == "M"
    )


def test_sparse_reassignment_no_mutate(prism):
    x = prism["Paid"].incr_to_cum()
    x["Capped 100k Paid"] = cl.minimum(x["Paid"], 100000)
    x["Excess 100k Paid"] = x["Paid"] - x["Capped 100k Paid"]
    a = x["Excess 100k Paid"].sum().grain("OYDY")
    x["Capped 100k Paid"] = cl.minimum(x["Paid"], 100000)
    x["Excess 100k Paid"] = x["Paid"] - x["Capped 100k Paid"]
    b = x["Excess 100k Paid"].sum().grain("OYDY")
    assert a == b


def test_trailing_origin():
    raa = (
        cl.load_sample("raa")
        .dev_to_val()
        .to_frame(keepdims=True, origin_as_datetime=True)
        .reset_index()
    )
    # adjust valuations to mid-year
    raa["valuation"] = raa["valuation"] - pd.DateOffset(months=6)
    tri = cl.Triangle(
        raa, origin="origin", development="valuation", columns="values", cumulative=True
    )
    assert tri.development.to_list() == [6, 18, 30, 42, 54, 66, 78, 90, 102, 114]
    assert tri.origin_close == "DEC"
    raa["origin2"] = raa["origin"] - pd.DateOffset(months=6)
    tri = cl.Triangle(
        raa,
        origin="origin2",
        development="valuation",
        columns="values",
        cumulative=True,
    )
    assert tri.development.to_list() == [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
    assert tri.origin_close == "JUN"


def test_trailing_valuation():
    data = (
        cl.load_sample("raa")
        .dev_to_val()
        .to_frame(keepdims=True, origin_as_datetime=True)
    )
    data.valuation = (data.valuation.dt.year + 1) * 100 + 3
    tri = cl.Triangle(data, origin="origin", development="valuation", columns="values")
    assert tri.development.to_list() == [3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123]
    tri2 = cl.Triangle(
        data, origin="origin", development="valuation", columns="values", trailing=True
    )
    assert tri == tri2


def test_edgecase_236():
    assert (
        cl.Triangle(
            pd.DataFrame(
                {
                    "origin": [201906, 201907],
                    "development": [201911, 201911],
                    "amount": [1, 0],
                }
            ),
            origin="origin",
            development="development",
            columns=["amount"],
            cumulative=True,
        )
        .val_to_dev()
        .iloc[..., 0, -1]
        .sum()
        == 1
    )


def test_to_frame_on_zero(clrd):
    assert len((clrd * 0).latest_diagonal.to_frame(origin_as_datetime=False)) == 0


def test_valuation_vector():
    df = pd.DataFrame(
        {
            "Accident Date": [201508, 201608, 201708],
            "Valuation Date": [202111, 202111, 202111],
            "Loss": [110, 594, 696],
        }
    )

    tri = cl.Triangle(
        df,
        origin="Accident Date",
        development="Valuation Date",
        columns="Loss",
        cumulative=True,
        trailing=True,
    )

    assert int(tri.valuation_date.strftime("%Y%m")) == 202111


def test_single_entry():
    # triangle with one entry
    data = pd.DataFrame(
        {"origin": 2014, "valuation_date": "01.01.2017", "amount": 100}, index=[1]
    )
    cl_tri = cl.Triangle(
        data,
        origin="origin",
        development="valuation_date",
        columns="amount",
        cumulative=True,
    )

    # create a development constant
    dev_periods = cl_tri.val_to_dev().development.to_list()
    kwargs = {"patterns": {k: 1.5 for k in dev_periods}, "style": "ldf_"}
    cl_dev_constant = cl.DevelopmentConstant(**kwargs)

    # fit - this now works
    cl_dev_constant_fit = cl_dev_constant.fit(cl_tri.val_to_dev())

    # aim
    cl.Chainladder().fit(cl_dev_constant_fit.transform(cl_tri)).ultimate_


def test_origin_as_datetime_arg(clrd):
    from pandas.api.types import is_datetime64_any_dtype

    assert is_datetime64_any_dtype(clrd.to_frame(origin_as_datetime=True)["origin"])
    assert not is_datetime64_any_dtype(
        clrd.to_frame(origin_as_datetime=False)["origin"]
    )


def test_full_triangle_and_full_expectation(raa):
    raa_cum = raa
    assert raa_cum.is_cumulative == True

    raa_incr = raa_cum.cum_to_incr()
    assert raa_incr.is_cumulative == False
    assert raa_incr.incr_to_cum().is_cumulative == True
    assert raa_incr.incr_to_cum() == raa_cum

    cl_fit_incr = cl.Chainladder().fit(X=raa_incr)
    assert cl_fit_incr.X_.is_cumulative == False

    cl_fit_cum = cl.Chainladder().fit(X=raa_cum)
    assert cl_fit_cum.X_.is_cumulative == True

    assert cl_fit_incr.cdf_ == cl_fit_cum.cdf_
    assert cl_fit_incr.ultimate_ == cl_fit_cum.ultimate_

    assert (
        cl_fit_cum.full_expectation_ - cl_fit_incr.full_expectation_.incr_to_cum()
        < 0.00001
    )
    assert (
        cl_fit_cum.full_triangle_ - cl_fit_incr.full_triangle_.incr_to_cum() < 0.00001
    )
    assert (cl_fit_cum.full_triangle_ - raa_cum) - (
        cl_fit_incr.full_triangle_.incr_to_cum() - raa_incr.incr_to_cum()
    ) < 0.00001

    bf_fit_incr = cl.BornhuetterFerguson(apriori=1).fit(
        X=raa_incr, sample_weight=raa_incr.incr_to_cum().latest_diagonal * 0
    )
    assert bf_fit_incr.X_.is_cumulative == False

    bf_fit_cum = cl.BornhuetterFerguson(apriori=1).fit(
        X=raa_cum, sample_weight=raa_cum.latest_diagonal * 0
    )
    assert bf_fit_cum.X_.is_cumulative == True

    assert bf_fit_incr.cdf_ == bf_fit_cum.cdf_
    assert bf_fit_incr.ultimate_ == bf_fit_cum.ultimate_

    assert (
        bf_fit_cum.full_expectation_ - bf_fit_incr.full_expectation_.incr_to_cum()
        < 0.00001
    )
    assert (
        bf_fit_cum.full_triangle_ - bf_fit_incr.full_triangle_.incr_to_cum() < 0.00001
    )
    assert (bf_fit_cum.full_triangle_ - raa_cum) - (
        bf_fit_incr.full_triangle_.incr_to_cum() - raa_incr.incr_to_cum()
    ) < 0.00001

    assert (
        cl.Chainladder().fit(raa_incr).full_triangle_
        - cl.Chainladder().fit(raa_cum).full_triangle_.cum_to_incr()
        <= 0.0001
    )
    bk_fit_incr = cl.Benktander(apriori=1.00, n_iters=2).fit(
        X=raa_incr, sample_weight=raa_incr.incr_to_cum().latest_diagonal * 0
    )
    bk_fit_cum = cl.Benktander(apriori=1.00, n_iters=2).fit(
        X=raa_cum, sample_weight=raa_cum.latest_diagonal * 0
    )

    assert (
        bk_fit_cum.full_expectation_ - bk_fit_incr.full_expectation_.incr_to_cum()
        < 0.00001
    )
    assert (
        bk_fit_cum.full_triangle_ - bk_fit_incr.full_triangle_.incr_to_cum() < 0.00001
    )
    assert (bk_fit_cum.full_triangle_ - raa_cum) - (
        bk_fit_incr.full_triangle_.incr_to_cum() - raa_incr.incr_to_cum()
    ) < 0.00001


def test_halfyear_grain():
    data = pd.DataFrame(
        {"AccMo": [201409, 201503, 201603], "ValMo": [202203] * 3, "value": [100] * 3}
    )
    assert cl.Triangle(
        data=data, origin="AccMo", development="ValMo", columns="value", cumulative=True
    ).shape == (1, 1, 16, 1)


def test_predict(raa):
    raa_cum = raa
    assert cl.Chainladder().fit(raa_cum).X_.is_cumulative == True
    assert (
        cl.BornhuetterFerguson()
        .fit(raa_cum, sample_weight=raa_cum.latest_diagonal * 0 + 40000)
        .X_.is_cumulative
        == True
    )
    assert cl.Chainladder().fit_predict(raa_cum).is_cumulative == True
    assert (
        cl.BornhuetterFerguson()
        .fit_predict(raa_cum, sample_weight=raa_cum.latest_diagonal * 0 + 40000)
        .is_cumulative
        == True
    )

    raa_incr = raa.cum_to_incr()
    assert cl.Chainladder().fit(raa_incr).X_.is_cumulative == False
    assert (
        cl.BornhuetterFerguson()
        .fit(raa_incr, sample_weight=raa_incr.latest_diagonal * 0 + 40000)
        .X_.is_cumulative
        == False
    )
    assert cl.Chainladder().fit_predict(raa_incr).is_cumulative == False
    assert (
        cl.BornhuetterFerguson()
        .fit_predict(raa_incr, sample_weight=raa_incr.latest_diagonal * 0 + 40000)
        .is_cumulative
        == False
    )


def test_halfyear_development():
    df_sub = pd.read_csv(
        io.StringIO(
            """
        2011-01-01, 2011-01-01, 179.74
        2011-01-01, 2011-07-01, 664.94
        2011-01-01, 2012-01-01, 7471.75
        2011-01-01, 2012-07-01, 820.99
        2011-01-01, 2013-01-01, 908.77
        """
        ),
        names=["origin", "development", "paid"],
        parse_dates=["origin", "development"],
    )
    assert (
        type(
            cl.Triangle(
                data=df_sub,
                origin="origin",
                origin_format="%Y-%m-%d",
                development="development",
                development_format="%Y-%m-%d",
                columns="paid",
                cumulative=True,
            )
        )
        == cl.Triangle
    )

    data = [
        ["2010-01-01", "2011-06-30", "premium", 100.0],
        ["2010-01-01", "2011-12-31", "incurred", 100.0],
        ["2010-01-01", "2012-06-30", "premium", 200.0],
        ["2010-01-01", "2012-12-31", "incurred", 100.0],
        ["2010-01-01", "2013-12-31", "incurred", 200.0],
        ["2011-01-01", "2011-06-30", "premium", 100.0],
        ["2011-01-01", "2012-06-30", "premium", 200.0],
        ["2011-01-01", "2012-12-31", "incurred", 100.0],
        ["2011-01-01", "2013-12-31", "incurred", 200.0],
        ["2012-01-01", "2012-06-30", "premium", 200.0],
        ["2012-01-01", "2013-12-31", "incurred", 200.0],
    ]

    assert (
        type(
            cl.Triangle(
                data=pd.DataFrame(data, columns=["origin", "val_date", "idx", "value"]),
                index="idx",
                columns="value",
                origin="origin",
                development="val_date",
                cumulative=True,
            )
        )
    ) == cl.Triangle


def test_latest_diagonal_vs_full_tri_raa(raa):
    model = cl.Chainladder().fit(raa)
    assert model.ultimate_.latest_diagonal == model.full_triangle_.latest_diagonal


def test_latest_diagonal_vs_full_tri_clrd(clrd):
    model = cl.Chainladder().fit(clrd)
    ult = model.ultimate_
    full_tri = model.full_triangle_

    assert np.round(full_tri.latest_diagonal, 0) == np.round(ult.latest_diagonal, 0)

def test_semi_annual_grain():
    Sdata = {
        'origin': ["2007-01-01", "2007-01-01", "2007-01-01", "2007-01-01", "2007-01-01", "2007-01-01", "2007-01-01",
                "2007-07-01", "2007-07-01", "2007-07-01", "2007-07-01", "2007-07-01", "2007-07-01",
                "2008-01-01", "2008-01-01", "2008-01-01", "2008-01-01", "2008-01-01",
                "2008-07-01", "2008-07-01", "2008-07-01", "2008-07-01",
                "2009-01-01", "2009-01-01", "2009-01-01",
                "2009-07-01", "2009-07-01",
                "2010-01-01"],
        'development': ["2007-01-01", "2007-07-01", "2008-01-01", "2008-07-01", "2009-01-01", "2009-07-01", "2010-01-01",
                        "2007-07-01", "2008-01-01", "2008-07-01", "2009-01-01", "2009-07-01", "2010-01-01",
                        "2008-01-01", "2008-07-01", "2009-01-01", "2009-07-01", "2010-01-01",
                        "2008-07-01", "2009-01-01", "2009-07-01", "2010-01-01",
                        "2009-01-01", "2009-07-01", "2010-01-01",
                        "2009-07-01", "2010-01-01",
                        "2010-01-01"],
        'loss': [100, 200, 300, 400, 500, 600, 700, 
                150, 300, 450, 500, 550, 600, 
                200, 250, 350, 400, 450, 
                50, 100, 150, 200,
                100, 200, 300,
                50, 150, 
                100]
    }

    Stri = cl.Triangle(
        pd.DataFrame(Sdata), 
        origin='origin',
        development='development',
        columns='loss',
        cumulative=True
    )

    Adata = {
        'origin': ["2007-01-01", "2007-01-01", "2007-01-01", "2007-01-01",
                "2008-01-01", "2008-01-01", "2008-01-01",
                "2009-01-01", "2009-01-01",
                "2010-01-01"],
        'development': ["2007-01-01", "2008-01-01", "2009-01-01", "2010-01-01",
                        "2008-01-01", "2009-01-01", "2010-01-01",
                        "2009-01-01", "2010-01-01",
                        "2010-01-01"],
        'loss': [100, 600, 1000, 1300, 
                200, 450, 650, 
                100, 450,
                100]
    }

    Atri = cl.Triangle(
        pd.DataFrame(Adata), 
        origin='origin',
        development='development',
        columns='loss',
        cumulative=True
    )
    assert Atri == Stri.grain('OYDY')

def test_odd_quarter_end():
    data= pd.DataFrame([
        ["5/1/2023", 12, '4/30/2024', 100],
        ["8/1/2023", 9, "4/30/2024", 130],
        ["11/1/2023", 6, "4/30/2024", 160],
        ["2/1/2024", 3, "4/30/2024", 140]], 
        columns = ['origin', 'development', 'valuation', 'EarnedPremium'])
    triangle = cl.Triangle(
        data, origin='origin', origin_format='%Y-%m-%d', development='valuation', columns='EarnedPremium', trailing=True, cumulative=True
    )
    data_from_tri = triangle.to_frame(origin_as_datetime=True)
    assert np.all(data_from_tri['2024Q2'].values == [100.,130.,160.,140.])
    assert np.all(data_from_tri.index == pd.DatetimeIndex(data=["5/1/2023","8/1/2023","11/1/2023","2/1/2024"],freq = 'QS-NOV'))


def test_single_valuation_date_preserves_exact_date():
    # Test that a single development date is preserved exactly and not converted to fiscal year
    # Regression test for issue where 202510 was incorrectly converted to 2026-09 instead of 2025-10
    data = pd.DataFrame({
        'Accident Year Month': [202002, 202003, 202105, 202201, 202301, 202401, 202501],
        'Calendar Year Month': [202510] * 7,  # Single valuation date
        'Loss': [100, 200, 150, 300, 250, 400, 350]
    })

    triangle = cl.Triangle(
        data=data,
        origin='Accident Year Month',
        development='Calendar Year Month',
        columns='Loss',
        cumulative=True,
        development_format='%Y%m',
        origin_format='%Y%m'
    )

    # Valuation date should be end of October 2025, not converted to a fiscal year
    assert triangle.valuation_date == pd.Timestamp('2025-10-31 23:59:59.999999999')
    assert triangle.development_grain == 'M'
    assert int(triangle.valuation_date.strftime('%Y%m')) == 202510
def test_OXDX_triangle():
    
    for x in [12,6,3,1]:
        for y in [i for i in [12,6,3,1] if i <= x]:
            first_orig = '2020-01-01'
            width = int(x / y) + 1
            dev_series = (pd.date_range(start=first_orig,periods = width, freq = str(y) + 'ME') + pd.DateOffset(months=y-1)).to_series()
            tri_df = pd.DataFrame({
                'origin_date': pd.concat([pd.to_datetime([first_orig] * (width)).to_series(), (pd.to_datetime([first_orig]) + pd.DateOffset(months=x)).to_series()]).to_list(),
                'development_date': pd.concat([dev_series,dev_series.iloc[[0]] + pd.DateOffset(months=x)]).to_list(),
                'value': list(range(1,width + 2))
            })
            for i in range(12):
                for j in range(y):
                    test_data = tri_df.copy()
                    test_data['origin_date'] += pd.DateOffset(months=i)        
                    test_data['development_date'] += pd.DateOffset(months=i-j)
                    tri = cl.Triangle(
                        test_data, 
                        origin='origin_date', 
                        development='development_date', 
                        columns='value', 
                        cumulative=True
                    )
                    assert tri.shape == (1,1,2,width)
                    assert tri.sum().sum() == tri_df['value'].sum()
                    assert np.all(tri.development == [y-j + x * y for x in range(width)])
                    #there's a known bug with origin that displays incorrect year when origin doesn't start on 1/1
                    #if x == 12:
                        #assert np.all(tri.origin == ['2020','2021'])
                    #elif x in [6,3]:
                        #assert np.all(tri.origin.strftime('%Y') == pd.to_datetime(tri.odims).strftime('%Y'))
                        #assert np.all(tri.origin.strftime('%q').values.astype(float) == np.ceil((pd.to_datetime(tri.odims).strftime('%m').values.astype(int) - 0.5) / 3))

def test_fillzero():
    raa = cl.load_sample('raa')
    zero = raa - raa[raa.origin=='1982']
    filled = zero.fillzero()
    assert (filled[filled.origin == '1982'][filled.development == 24].values.flatten()[0]) == 0