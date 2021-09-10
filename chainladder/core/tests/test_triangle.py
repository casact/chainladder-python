import chainladder as cl
import pandas as pd
import numpy as np
import copy
import pytest

try:
    from IPython.core.display import HTML
except:
    HTML = None


def test_repr(raa):
    np.testing.assert_array_equal(
        pd.read_html(raa._repr_html_())[0].set_index("Unnamed: 0").values,
        raa.to_frame().values,
    )


def test_to_frame_unusual(clrd):
    a = clrd.groupby(["LOB"]).sum().latest_diagonal["CumPaidLoss"].to_frame()
    b = clrd.latest_diagonal["CumPaidLoss"].groupby(["LOB"]).sum().to_frame()
    assert (a == b).all().all()


def test_link_ratio(raa, atol):
    assert (raa.link_ratio * raa.iloc[:, :, :-1, :-1].values -
            raa.values[:, :, :-1, 1:]).sum().sum() < atol


def test_incr_to_cum(clrd):
    clrd.cum_to_incr().incr_to_cum() == clrd


def test_create_new_value(clrd):
    clrd2 = clrd.copy()
    clrd2["lr"] = clrd2["CumPaidLoss"] / clrd2["EarnedPremDIR"]
    assert (clrd.shape[0], clrd.shape[1] + 1, clrd.shape[2], clrd.shape[3]) == clrd2.shape


def test_multilevel_index_groupby_sum1(clrd):
    assert clrd.groupby("LOB").sum().sum() == clrd.sum()


def test_multilevel_index_groupby_sum2(clrd):
    a = clrd.groupby("GRNAME").sum().sum()
    b = clrd.groupby("LOB").sum().sum()
    assert a == b


def test_boolean_groupby_eq_groupby_loc(clrd):
    assert (clrd[clrd["LOB"] == "ppauto"].sum() ==
            clrd.groupby("LOB").sum().loc["ppauto"])


def test_latest_diagonal_two_routes(clrd):
    assert (clrd.latest_diagonal.sum()["BulkLoss"] ==
            clrd.sum().latest_diagonal["BulkLoss"])


def test_sum_of_diff_eq_diff_of_sum(clrd):
    assert (clrd["BulkLoss"] - clrd["CumPaidLoss"]).latest_diagonal == (
        clrd.latest_diagonal["BulkLoss"] - clrd.latest_diagonal["CumPaidLoss"]
    )


def test_append(raa):
    raa2 = raa.copy()
    raa2.kdims = np.array([["P2"]])
    raa.append(raa2).sum() == raa * 2
    assert raa.append(raa2).sum() == 2 * raa


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
    assert (
        abs((raa.trend(0.05).trend((1 / 1.05) - 1) - raa).sum().sum()) < 1e-5)


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
    d = (raa.latest_diagonal * 0 + 50000).to_frame().reset_index()
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
        raa.grain("OYDY").latest_diagonal.set_backend('numpy').values,
        raa.latest_diagonal.grain("OYDY").set_backend('numpy').values,
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
    x = x["paid"].to_frame().T.unstack().reset_index()
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
    df = d.to_frame().reset_index()
    assert cl.Triangle(df, origin="index", columns=df.columns[-1]) == d


def test_trend_on_vector(raa):
    d = raa.latest_diagonal
    assert d.trend(0.05, axis=2).to_frame().astype(int).iloc[0, 0] == 29217


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
        clrd.to_frame().groupby(groups, axis=1).sum()
        == clrd.groupby(groups, axis=1).sum().to_frame()
    )
    assert np.all(
        clrd.to_frame().groupby("LOB").sum() == clrd.groupby("LOB").sum().to_frame()
    )
    prism.sum().grain("OYDY")


def test_partial_year(prism):
    before = prism['Paid'].sum().incr_to_cum()
    before=before[before.valuation<='2017-08'].latest_diagonal

    after = cl.Triangle(
        before.to_frame(keepdims=True).reset_index(),
        origin='origin', development='valuation', columns='Paid', index=before.key_labels)

    assert after.valuation_date == before.valuation_date


def test_array_protocol(raa, clrd):
    assert np.sqrt(raa) == raa.sqrt()
    assert np.concatenate((clrd.iloc[:200], clrd.iloc[200:]),0) == cl.concat((clrd.iloc[:200], clrd.iloc[200:]),0)


def test_dask_backend(raa):
    raa1 = cl.Chainladder().fit(raa.set_backend('dask')).ultimate_
    raa2 = cl.Chainladder().fit(raa).ultimate_
    assert (raa1 == raa2).compute()


def test_partial_val_dev(raa):
    raa = raa.latest_diagonal
    raa.iloc[..., -3:, :] = np.nan
    raa.val_to_dev().iloc[0, 0, 0, -1] == raa.iloc[0, 0, 0, -1]


def test_sort_axis(clrd):
    assert clrd.iloc[::-1, ::-1, ::-1, ::-1].sort_axis(0).sort_axis(1).sort_axis(2).sort_axis(3) == clrd.sort_axis(1)


def test_shift(raa):
    assert (
        raa.iloc[..., 1:-1, 1:-1] -
        raa.shift(-1, axis=2).shift(-1, axis=3).shift(2, axis=2).shift(2, axis=3).dropna().values
    ).to_frame().fillna(0).sum().sum() == 0


def test_array_protocol2(raa):
    import numpy as np
    assert raa.log().exp() == np.exp(np.log(raa))


def test_create_full_triangle(raa):
    a = cl.Chainladder().fit(raa).full_triangle_
    b = cl.Triangle(
        a.to_frame(keepdims=True, implicit_axis=True),
        origin='origin', development='valuation', columns='values')
    assert a == b


def test_groupby_getitem(clrd):
    assert clrd.groupby('LOB')['CumPaidLoss'].sum() == clrd['CumPaidLoss'].groupby('LOB').sum()


def test_virtual_column(prism):
    prism['P'] = prism['Paid']
    prism['Paid'] = lambda x : x['P']
    assert prism['Paid'] == prism['P']


def test_correct_valutaion(raa):
    new = cl.Triangle(
        raa.iloc[..., :-3, :].latest_diagonal.to_frame(keepdims=True, implicit_axis=True),
        origin='origin', development='valuation', columns='values')
    assert new.valuation_date == raa.valuation_date


@pytest.mark.xfail
@pytest.mark.parametrize(
    'prop', ['cdf_', 'ibnr_', 'full_expectation_', 'full_triangle_'])
def test_no_fitted(raa, prop):
    getattr(raa, prop)

def test_pipe(raa):
    f = lambda x: x.loc[..., 48:]
    assert raa.loc[..., 48:] == raa.pipe(f)

def test_repr_html(raa, clrd):
    assert type(raa._repr_html_()) == str
    assert type(clrd._repr_html_()) == str


@pytest.mark.xfail(HTML is None, reason="ipython needed for test")
def test_heatmap(raa):
    raa.link_ratio.heatmap()


def test_agg_sparse():
    a = cl.load_sample('raa')
    b = cl.load_sample('raa').set_backend('sparse')
    assert a.mean().mean() == b.mean().mean()

def test_inplace(raa):
    t = raa.copy()
    t.dev_to_val(inplace=True)
    t.val_to_dev(inplace=True)
    t.grain('OYDY', inplace=True)
