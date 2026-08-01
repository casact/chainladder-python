from __future__ import annotations

import chainladder as cl
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chainladder.core.common import Common
from chainladder.utils.utility_functions import date_delta_adjustment
from chainladder.utils.sparse import COO

from io import StringIO

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle

def test_repr(raa):
    np.testing.assert_array_equal(
        pd.read_html(StringIO(raa._repr_html_()))[0].set_index("Unnamed: 0").values,
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

def test_align_pattern(raa, atol):
    with pytest.raises(ValueError):
        raa.align_pattern(raa)
        
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

    genins.rename('columns',{'foo':'newfoo'})

    assert genins.columns.to_list() == ['newfoo']

    genins.rename('columns',{'foo':'newnewfoo'})

    assert genins.columns.to_list() == ['newfoo']

def test_rename_index() -> None:
    """
    Test the renaming of triangle columns.
    """
    auto = cl.load_sample('auto')
    new_index = ['CommAuto','PersAuto']
    auto.rename('index',new_index)
    assert np.all(auto.index.values.flatten() == new_index)
    
def test_rename_exception(genins, clrd) -> None:
    # Test incorrect value argument - misspelling of string.
    with pytest.raises(ValueError):
        genins.rename('origin', {'oldfoo':'foo'})

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


def test_development_before_origin_warns_and_drops() -> None:
    """
    Rows where development precedes origin are invalid. Triangle.__init__ should
    emit a UserWarning and silently drop those rows.

    Returns
    -------
    None
    """
    df = pd.DataFrame({
        "origin":      [2000, 2000, 2001, 2001],
        "development": [2001, 2002, 2000, 2002],  # 2001/2000 row is invalid
        "value":       [100,  200,  999,  300],
    })
    with pytest.warns(UserWarning, match="development before"):
        tri = cl.Triangle(
            df, origin="origin", development="development",
            columns="value", cumulative=True,
        )
    # The invalid row (value=999) must not appear in the triangle.
    assert 999 not in tri.to_frame().values


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


def test_dropna_latest_diagonal(raa: Triangle) -> None:
    """
    dropna() on a single-development-period triangle (shape[-1] == 1), where first origin period is nan.
    First origin period should be eliminated.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    t = raa.copy()
    t.values[:, :, 0, :] = np.nan
    result = t.latest_diagonal.dropna()
    assert result.shape == (1, 1, 9, 1)
    assert result.origin.min().year == 1982


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
        clrd.to_frame(origin_as_datetime=False).T.groupby(groups).sum().T
        == clrd.groupby(groups, axis=1).sum().to_frame(origin_as_datetime=False)
    )
    assert np.all(
        clrd.to_frame(origin_as_datetime=False).groupby("LOB").sum()
        == clrd.groupby("LOB").sum().to_frame(origin_as_datetime=False)
    )
    prism.sum().grain("OYDY")


def test_groupby_agg_auto_sparse(prism: Triangle) -> None:
    """
    Verify _auto_sparse functionality in groupby agg func.

    For prism, the grouped result is sparse
    but small enough that _auto_sparse() converts it back to numpy; passing
    auto_sparse=False bypasses that conversion and leaves it sparse.

    Parameters
    ----------
    prism: Triangle
        The prism sample data set Triangle.

    Returns
    -------
    None
    """
    result_default   = prism.groupby("Line").sum()
    result_no_sparse = prism.groupby("Line").sum(auto_sparse=False)

    assert result_default.array_backend == "numpy"
    assert result_no_sparse.array_backend == "sparse"
    assert result_default == result_no_sparse


def test_auto_sparse_disabled_returns_self(prism: Triangle) -> None:
    """
    When cl.options.AUTO_SPARSE is False, _auto_sparse() returns the triangle
    unchanged without switching backends.

    Parameters
    ----------
    prism : Triangle
        The prism sample data set Triangle.

    Returns
    -------
    None
    """
    dense = prism.set_backend("numpy")
    cl.options.set_option("AUTO_SPARSE", False)
    try:
        result = dense._auto_sparse()
        assert result is dense
        assert result.array_backend == "numpy"
    finally:
        cl.options.reset_option("AUTO_SPARSE")


def test_auto_sparse_converts_numpy_to_sparse(prism: Triangle) -> None:
    """
    _auto_sparse() should convert a numpy-backed triangle to sparse when it is
    large enough (> 30Mb) and sparse enough (density <= 20%).

    Parameters
    ----------
    prism: Triangle
        The prism sample data set Triangle.

    Returns
    -------
    None
    """
    # Slice down to the fewest claims (66) whose dense (index, columns,
    # origin, development) shape still clears the 30Mb/8-byte-float
    # threshold in _auto_sparse(); the full prism triangle is ~2B cells and
    # would need ~15GB as a dense numpy array.
    small_prism = prism.iloc[:66]
    dense = small_prism.set_backend("numpy")
    assert dense.array_backend == "numpy"

    result = dense._auto_sparse()

    assert result is dense
    assert result.array_backend == "sparse"


def test_subtriangles(raa: Triangle) -> None:
    """
    subtriangles should list the attributes on a Triangle instance that are
    themselves Triangle instances, e.g. the ldf_/sigma_/std_err_ triangles
    attached by Development.fit_transform. A plain Triangle with no such
    attributes should report an empty list.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    assert raa.subtriangles == []

    fit = cl.Development().fit_transform(raa)

    assert set(fit.subtriangles) == {
        "std_err_", "ldf_", "sigma_", "std_residuals_", "w_v2_"
    }


def test_array_dunder(raa: Triangle) -> None:
    """
    __array__ lets numpy treat a Triangle as array-like, e.g. via np.asarray()
    or np.array(), returning the underlying values.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    arr = np.asarray(raa)

    assert arr is raa.values
    np.testing.assert_array_equal(np.array(raa), raa.values)


def test_triangle_from_dataframe_interchange_protocol() -> None:
    """
    Triangle() should accept any object supporting the __dataframe__
    interchange protocol (e.g. a polars DataFrame), converting it to a
    pandas DataFrame via _interchange_dataframe() under the hood.

    Returns
    -------
    None
    """
    polars = pytest.importorskip("polars")

    df = pd.DataFrame(
        {
            "origin": ["2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01"],
            "development": ["2020-12-31", "2021-12-31", "2021-12-31", "2022-12-31"],
            "values": [100, 150, 120, 180],
        }
    )
    pl_df = polars.from_pandas(df)
    assert hasattr(pl_df, "__dataframe__")
    assert not isinstance(pl_df, pd.DataFrame)

    tri = cl.Triangle(
        pl_df,
        origin="origin",
        development="development",
        columns="values",
        cumulative=True,
    )
    expected = cl.Triangle(
        df,
        origin="origin",
        development="development",
        columns="values",
        cumulative=True,
    )

    assert tri == expected


def test_array_function_unhandled_raises(raa: Triangle) -> None:
    """
    __array_function__ should return NotImplemented for numpy functions that
    are neither explicitly handled (e.g. np.concatenate, np.round) nor
    aliases of a Triangle method of the same name (e.g. np.sum). numpy then
    turns that NotImplemented into a TypeError.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    assert "stack" not in dir(raa)

    with pytest.raises(TypeError):
        np.stack([raa, raa])


def test_array_function_mixed_types_raises(raa: Triangle) -> None:
    """
    __array_function__ should return NotImplemented when one of the
    dispatching argument types is not a Triangle subclass, even for a
    handled function like np.concatenate. numpy then turns that
    NotImplemented into a TypeError.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """

    class NotATriangle:
        @staticmethod
        def __array_function__(_func, _types, _args, _kwargs):
            return NotImplemented

    with pytest.raises(TypeError):
        np.concatenate([raa, NotATriangle()])


def test_get_axis_none(clrd: Triangle) -> None:
    """
    Pass axis=None to TriangleGroupBy. Should be the same as passing axis=0.
    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    assert clrd.groupby("LOB", axis=None).sum() == clrd.groupby("LOB", axis=0).sum()


def test_astype(raa: Triangle) -> None:
    """
    Cast values in triangle to another data type.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    assert raa.astype("float32").values.dtype == np.float32


def test_head(clrd: Triangle) -> None:
    """
    Triangle.head(n) returns a Triangle limited to the first n rows of the index axis.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    assert clrd.head(3).shape[0] == 3
    assert list(clrd.head(3).index['LOB']) == ['othliab', 'ppauto', 'comauto']


def test_tail(clrd: Triangle) -> None:
    """
    Triangle.tail(n) returns a Triangle limited to the last n rows of the index axis.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    assert clrd.tail(3).shape[0] == 3
    assert list(clrd.tail(3).index['LOB']) == ['wkcomp', 'comauto', 'wkcomp']


def test_add_df_passthru(raa: Triangle) -> None:
    """
    Check equivalent behavior between Triangle and DataFrame passed-thru methods.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    frame = raa.to_frame()

    # String serialization
    assert raa.to_csv() == frame.to_csv()
    assert raa.to_html() == frame.to_html()

    # DataFrame-returning methods
    assert raa.describe().equals(frame.describe())
    assert raa.drop_duplicates().equals(frame.drop_duplicates())
    assert raa.melt().equals(frame.melt())
    assert raa.unstack().equals(frame.unstack())


def test_plot(raa: Triangle) -> None:
    """
    TrianglePandas.plot() delegates to to_frame(origin_as_datetime=False).plot().
    Each plotted line must carry the same y-values as the equivalent DataFrame plot.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    matplotlib.use("Agg")


    try:
        ax_tri = raa.plot()
        ax_df  = raa.to_frame(origin_as_datetime=False).plot()

        lines_tri = ax_tri.get_lines()
        lines_df  = ax_df.get_lines()

        assert len(lines_tri) == len(lines_df)
        for lt, ld in zip(lines_tri, lines_df):
            np.testing.assert_array_equal(lt.get_ydata(orig=True), ld.get_ydata(orig=True))
    finally:
        plt.close("all")


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


def test_to_frame_keepdims_virtual_column(clrd: Triangle) -> None:
    """
    to_frame(keepdims=True) computes virtual columns and propagates NaN
    for rows where the underlying physical column has no data.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    t = clrd.copy()
    t["BulkLossDoubled"] = lambda x: x["BulkLoss"] * 2
    df = t.to_frame(keepdims=True)

    # Where BulkLoss has no data, the virtual column must also be NaN.
    assert df.loc[df["BulkLoss"].isna(), "BulkLossDoubled"].isna().all()

    # Where BulkLoss has data, the virtual column must compute correctly.
    mask = df["BulkLoss"].notna()
    assert (df.loc[mask, "BulkLossDoubled"] == df.loc[mask, "BulkLoss"] * 2).all()


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


@pytest.mark.parametrize(
    "prop", ["cdf_", "ibnr_", "full_expectation_", "full_triangle_"]
)
def test_no_fitted(raa, prop):
    with pytest.raises(AttributeError, match=f"no attribute '{prop}'"):
        getattr(raa, prop)


def test_pipe(raa):
    def f(x):
        return x.loc[..., 48:]

    assert raa.loc[..., 48:] == raa.pipe(f)


def test_repr_html(raa, clrd):
    assert type(raa._repr_html_()) == str
    assert type(clrd._repr_html_()) == str


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


def test_full_triangle_and_full_expectation(raa,atol):
    raa_cum = raa
    assert raa_cum.is_cumulative == True

    raa_incr = raa_cum.cum_to_incr()
    assert raa_incr.is_cumulative == False
    assert raa_incr.incr_to_cum().is_cumulative == True
    assert raa_incr.incr_to_cum() == raa_cum

    cl_fit_incr = cl.Chainladder().fit(X=raa_incr)
    cl_predict_incr = cl.Chainladder().fit_predict(X=raa_incr)
    assert cl_fit_incr.X_.is_cumulative == False

    cl_fit_cum = cl.Chainladder().fit(X=raa_cum)
    cl_predict_cum = cl.Chainladder().fit_predict(X=raa_cum)
    assert cl_fit_cum.X_.is_cumulative == True

    assert cl_fit_incr.cdf_ == cl_fit_cum.cdf_
    assert cl_fit_incr.ultimate_ == cl_fit_cum.ultimate_

    assert (np.allclose(
        cl_fit_cum.full_expectation_.values,
        cl_predict_cum.full_expectation_.values,
        atol=atol
        )
    )
    assert (np.allclose(
        cl_fit_incr.full_expectation_.fillzero().values,
        cl_predict_incr.full_expectation_.fillzero().values,
        atol=atol
        )
    )
    assert (
        cl_fit_cum.full_expectation_ - cl_fit_incr.full_expectation_.incr_to_cum()
        < atol
    )
    assert (
        cl_fit_cum.full_triangle_ - cl_fit_incr.full_triangle_.incr_to_cum() < atol
    )
    assert (cl_fit_cum.full_triangle_ - raa_cum) - (
        cl_fit_incr.full_triangle_.incr_to_cum() - raa_incr.incr_to_cum()
    ) < atol

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
        < atol
    )
    assert (
        bf_fit_cum.full_triangle_ - bf_fit_incr.full_triangle_.incr_to_cum() < atol
    )
    assert (bf_fit_cum.full_triangle_ - raa_cum) - (
        bf_fit_incr.full_triangle_.incr_to_cum() - raa_incr.incr_to_cum()
    ) < atol

    assert (
        cl.Chainladder().fit(raa_incr).full_triangle_
        - cl.Chainladder().fit(raa_cum).full_triangle_.cum_to_incr()
        <= atol
    )
    bk_fit_incr = cl.Benktander(apriori=1.00, n_iters=2).fit(
        X=raa_incr, sample_weight=raa_incr.incr_to_cum().latest_diagonal * 0
    )
    bk_fit_cum = cl.Benktander(apriori=1.00, n_iters=2).fit(
        X=raa_cum, sample_weight=raa_cum.latest_diagonal * 0
    )

    assert (
        bk_fit_cum.full_expectation_ - bk_fit_incr.full_expectation_.incr_to_cum()
        < atol
    )
    assert (
        bk_fit_cum.full_triangle_ - bk_fit_incr.full_triangle_.incr_to_cum() < atol
    )
    assert (bk_fit_cum.full_triangle_ - raa_cum) - (
        bk_fit_incr.full_triangle_.incr_to_cum() - raa_incr.incr_to_cum()
    ) < atol


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
    val_date_exp: str = date_delta_adjustment("2025-11-01")
    assert triangle.valuation_date == pd.Timestamp(val_date_exp)
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
    
def test_2x2_triangle():
        
    df = pd.DataFrame(data={
        'origin': [2022, 2022, 2023],
        'development': [2022, 2023, 2023],
        'reported': [78000, 222000, 78000]}
    )
    tri_from_df = cl.Triangle(
        data=df,
        origin='origin',
        development='development',
        columns=['reported'],
        cumulative=True
    )
    tri_from_df
    assert np.array_equal(tri_from_df.cum_to_incr().values,np.array([[[[ 78000., 144000.],
    [ 78000., np.float64(np.nan)]]]]), equal_nan=True)


def test_triangle_init_from_dict() -> None:
    """
    Tests the initialization of a triangle by supplying a dict to the data parameter. The triangle
    created should be equal to that of one created with a pandas DataFrame containing the same data.
    """

    # Common data.
    data_dict = {
            'origin': [1981, 1981, 1981, 1981, 1982, 1982, 1982, 1983, 1983, 1984],
            'development': [1981, 1982, 1983, 1984, 1982, 1983, 1984, 1983, 1984, 1984],
            'reported': [5012, 8269, 10907, 11805, 106, 4285, 5396, 3410, 8992, 5655]
    }

    # Initialze via DataFrame.
    df = pd.DataFrame(
        data=data_dict
    )
    tri_from_df = cl.Triangle(
        data=df,
        origin='origin',
        development='development',
        columns=['reported'],
        cumulative=True
    )

    # Initialize via dict.
    tri_from_dict = cl.Triangle(
        data=data_dict,
        origin='origin',
        development='development',
        columns=['reported'],
        cumulative=True
    )

    assert tri_from_df == tri_from_dict


def test_validate_assumption(raa: Triangle) -> None:
    """
    Tests Common._validate_assumption.
    """

    # Check incorrect type provided to value argument.
    with pytest.raises(TypeError):
        raa._validate_assumption(
            triangle=raa,
            value=raa, axis=3  # noqa - incorrect type provided on purpose.
        )


@pytest.mark.parametrize("value", [1, 1.5, "volume"])
def test_validate_assumption_scalar(raa: Triangle, value: int | float | str) -> None:
    """
    Scalar int, float, and str values are broadcast across the axis.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.
    value: int | float | str
        A user-supplied assumption.

    Returns
    -------
    None
    """
    result = raa._validate_assumption(raa, value, axis=3)
    assert result.shape == (1, 1, 1, raa.shape[3])
    assert (result.flat[0] == value)


@pytest.mark.parametrize("value", [
    [1] * 10,
    (1,) * 10,
    np.ones(10),
])
def test_validate_assumption_sequence(raa: Triangle, value: list | tuple | np.ndarray) -> None:
    """
    list, tuple, and ndarray values are wrapped in np.array and reshaped.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.
    value: list | tuple | np.ndarray
        A user-supplied assumption.

    Returns
    -------
    None
    """
    result = raa._validate_assumption(raa, value, axis=3)
    assert result.shape == (1, 1, 1, raa.shape[3])


def test_validate_assumption_set(raa: Triangle) -> None:
    """
    set values reach the sequence branch without raising.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    # np.array(set) produces a 0-d object array in NumPy 2.x, so we only
    # assert the call succeeds, not the resulting shape.
    raa._validate_assumption(raa, {1, 2, 3}, axis=3)


def test_validate_assumption_dict(raa: Triangle) -> None:
    """
    Dict values are mapped by axis label.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    dev_periods = raa._get_axis_value(3).tolist()
    value = {p: float(i) for i, p in enumerate(dev_periods)}
    result = raa._validate_assumption(raa, value, axis=3)
    assert result.shape == (1, 1, 1, raa.shape[3])
    np.testing.assert_array_equal(result.flat[:], list(value.values()))


def test_validate_assumption_callable(raa: Triangle) -> None:
    """
    Callable values are applied to each axis label.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    result = raa._validate_assumption(raa, lambda x: x * 2, axis=3)
    assert result.shape == (1, 1, 1, raa.shape[3])
    expected = np.array([p * 2 for p in raa._get_axis_value(3).tolist()])
    np.testing.assert_array_equal(result.flatten(), expected)


def test_validate_assumption_axis2(raa: Triangle) -> None:
    """
    axis=2 produces shape (1, 1, n_origin, 1).

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.

    Returns
    -------
    None
    """
    result = raa._validate_assumption(raa, 1, axis=2)
    assert result.shape == (1, 1, raa.shape[2], 1)


def test_xs(clrd):
    # when slicing with .loc on the first term in the index, Triangle will drop the term 
    assert clrd.xs('Adriatic Ins Co') == clrd.loc['Adriatic Ins Co']
    assert clrd.xs('Adriatic Ins Co').index.equals(clrd.loc['Adriatic Ins Co'].index)
    # when slicing with .loc on the all term in the index, Triangle will not drop any term 
    assert clrd.xs(('Agway Ins Co','comauto'), drop_level=False) == clrd.loc['Agway Ins Co','comauto']
    assert clrd.xs(('Agway Ins Co','comauto'), drop_level=False).index.equals(clrd.loc['Agway Ins Co','comauto'].index)
    # when all index terms are included in xs and drop_level is True, the default 'Total' index value is provided
    assert clrd.xs(('Agway Ins Co','comauto'), drop_level=True).index.equals(cl.load_sample('genins').index)
    # when slicing with .loc on the second or subsequent terms in the index, Triangle will not drop the term 
    assert clrd.xs('comauto',level=1, drop_level=False) == clrd.loc[clrd['LOB'] == 'comauto']
    assert clrd.xs('comauto',level=1, drop_level=False).index.equals(clrd.loc[clrd['LOB'] == 'comauto'].index)
    # level works with either integer index or name of the index column
    assert clrd.xs('comauto',level=1) == clrd.xs('comauto',level='LOB')
    assert clrd.xs('comauto',level=1).index.equals(clrd.xs('comauto',level='LOB').index)


def test_get_array_module_with_explicit_arr(raa: Triangle) -> None:
    """
    Supply a ndarray to Triangle.get_array_module(), which should be np (numpy).

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    assert raa.get_array_module(np.array([1.0])) is np


def test_get_array_module_invalid_backend_raises(raa: Triangle) -> None:
    """
    Simulate typo in setting array backend, should raise an exception.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    tri = raa.copy()
    tri.array_backend = 'mumpy'
    with pytest.raises(Exception, match="Array backend is invalid or not properly set"):
        tri.get_array_module()


def test_set_development_no_development_column() -> None:
    """
    Initialize a triangle without a development dimension specified. Development will default to being
    a single period set to the latest origin period.

    Returns
    -------
    None

    """
    df = pd.DataFrame(
        {
            'origin': [1995, 1996, 1997],
            'reported': [1.0, 2.0, 3.0]
        }
    )
    tri = cl.Triangle(
        data=df,
        origin='origin',
        columns='reported',
        cumulative=True
    )
    assert tri.shape[-1] == 1
    assert tri.development[0] == str(tri.origin[-1])


def test_set_development_age_instead_of_date_raises() -> None:
    """
    Initialize a triangle with incorrect development periods specified. Should raise a ValueError.

    Returns
    -------
    None

    """
    df = pd.DataFrame(
        {
            'origin': [1995, 1996],
            'development': [12, 24],
            'reported': [1.0, 2.0]
        }
    )
    with pytest.raises(ValueError, match="Development lags could not be determined"):
        cl.Triangle(
            data=df,
            origin='origin',
            development='development',
            columns='reported',
            cumulative=True
        )


def test_input_validation_non_numeric_columns_raises() -> None:
    """
    Initialize a triangle with reported losses as an invalid string data type. Should raise a TypeError.

    Returns
    -------
    None

    """
    df = pd.DataFrame({
        'origin': [1995, 1996],
        'development': [1995, 1996],
        'reported': ['1000', '2000'],
    })
    with pytest.raises(TypeError, match="column attribute must be numeric"):
        cl.Triangle(
            data=df,
            origin='origin',
            development='development',
            columns='reported',
            cumulative=True
        )


def test_input_validation_duplicate_columns_raises() -> None:
    """
    Initialize a triangle with duplicate column names. Raise an AttributeError.

    Returns
    -------
    None

    """
    df = pd.DataFrame(
        [[1995, 1995, 1.0, 2.0], [1996, 1996, 3.0, 4.0]],
        columns=pd.Index(['origin', 'development', 'reported', 'reported'])
    )
    with pytest.raises(AttributeError, match="Columns are required to have unique names"):
        cl.Triangle(
            data=df,
            origin='origin',
            development='development',
            columns='reported',
            cumulative=True
        )


def test_get_axis_value(raa) -> None:
    """
    Extract the Triangle axes using integer indices and string specifications.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    assert raa._get_axis_value(0).equals(raa.index)
    assert raa._get_axis_value("index").equals(raa.index)
    assert raa._get_axis_value(1).equals(raa.columns)
    assert raa._get_axis_value("columns").equals(raa.columns)
    assert raa._get_axis_value(2).equals(raa.origin)
    assert raa._get_axis_value("origin").equals(raa.origin)
    assert raa._get_axis_value(3).equals(raa.development)
    assert raa._get_axis_value("development").equals(raa.development)
    # Negative index support in TrianglePandas().get_axis()
    assert raa._get_axis_value(-4).equals(raa.index)
    assert raa._get_axis_value(-1).equals(raa.development)
    with pytest.raises(ValueError):
        raa._get_axis_value("dev")


def test_to_datetime_uninferrable_format_raises() -> None:
    """
    Initialize a triangle with incorrect date format on the origin axis. Should raise a ValueError.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="Unable to infer datetime"):
        cl.Triangle(
            data={
                'origin': ['1995/Q1', '1996/Q1'],
                'development': ['1995Q1', '1996Q1'],
                'value': [1.0, 2.0]},
            origin='origin',
            development='development',
            columns='value',
            cumulative=True
        )


def test_set_backend_via_ldf(raa: Triangle) -> None:
    """
    Call set_backend on a fitted estimator. The estimator has no array_backend attribute
    of its own, so set_backend resolves the old backend through ldf_.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    dev = cl.Development().fit(raa)
    dev.set_backend("sparse", inplace=True, deep=True)
    assert dev.ldf_.array_backend == "sparse"


def test_set_backend_no_array_backend_raises() -> None:
    """
    Call set_backend on an unfitted estimator. The estimator has neither array_backend
    nor ldf_, so set_backend raises ValueError.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError, match="Unable to determine array backend"):
        cl.Development().set_backend("sparse", inplace=True)


def test_set_backend_inplace_updates_array_backend_attr(raa: Triangle) -> None:
    """
    set_backend(inplace=True) updates self.array_backend when the attribute exists.
    When called on an object without array_backend, the attribute is not added.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    # Triangle has array_backend, updated in-place.
    tri = raa.set_backend("numpy")
    tri.set_backend("sparse", inplace=True)
    assert tri.array_backend == "sparse"

    # Estimator has no array_backend, attribute is not added.
    dev = cl.Development().fit(raa.set_backend("numpy"))
    dev.set_backend("sparse", inplace=True)
    assert not hasattr(dev, "array_backend")


def test_set_backend_deep_propagates_to_nested_common(raa: Triangle) -> None:
    """
    set_backend with deep=True iterates vars(self) and recursively converts every
    nested Common instance. Without deep=True, the nested
    attributes keep their original backend; with deep=True all are converted.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    dev = cl.Development().fit(raa.set_backend("numpy"))
    common_attrs = [k for k, v in vars(dev).items() if isinstance(v, Common)]

    # deep=False: nested Triangle attributes keep numpy
    dev.set_backend("sparse", inplace=True, deep=False)
    assert all(getattr(dev, k).array_backend == "numpy" for k in common_attrs)

    # deep=True: every nested Common attribute is converted to sparse
    dev.set_backend("sparse", inplace=True, deep=True)
    assert all(getattr(dev, k).array_backend == "sparse" for k in common_attrs)


def test_set_backend_inplace_mutates_values(raa: Triangle) -> None:
    """
    set_backend(inplace=True) reassigns self.values.
    Verify the in-place mutation produces the correct array type: numpy to sparse yields a COO,
    sparse to numpy yields an ndarray.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    numpy_tri = raa.set_backend("numpy")
    numpy_tri.set_backend("sparse", inplace=True)
    assert isinstance(numpy_tri.values, COO)

    numpy_tri.set_backend("numpy", inplace=True)
    assert isinstance(numpy_tri.values, np.ndarray)


def test_set_backend_invalid_raises(raa: Triangle) -> None:
    """
    Pass an unsupported backend name to set_backend. Should raise AttributeError.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    with pytest.raises(AttributeError):
        raa.set_backend("invalid_backend", inplace=True)


def test_set_backend_roundtrip(raa: Triangle) -> None:
    """
    Convert numpy to sparse and back to numpy, and verify values are preserved.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    sparse_raa = raa.set_backend("sparse")
    assert sparse_raa.array_backend == "sparse"
    restored = sparse_raa.set_backend("numpy")
    assert restored.array_backend == "numpy"
    assert restored == raa


def test_has_zeta_true(raa: Triangle) -> None:
    """
    has_zeta returns True after fitting IncrementalAdditive, which sets zeta_.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    fitted = cl.IncrementalAdditive().fit(raa, sample_weight=raa.latest_diagonal)
    assert fitted.has_zeta is True


def test_has_zeta_false(raa: Triangle) -> None:
    """
    has_zeta returns False for an estimator that does not set zeta_.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    assert cl.Development().fit(raa).has_zeta is False


def test_cum_zeta_raises_when_no_zeta(raa: Triangle) -> None:
    """
    cum_zeta_ raises AttributeError when the estimator has no zeta_.

    Parameters
    ----------
    raa : Triangle
        The raa sample data set.

    Returns
    -------
    None
    """
    with pytest.raises(AttributeError):
        _ = cl.Development().fit(raa).cum_zeta_


def test_cum_zeta_returns_incr_to_cum(atol) -> None:
    """
    cum_zeta_ returns zeta_.incr_to_cum() when zeta_ is present.

    Parameters
    ----------
    atol : float
        Absolute tolerance fixture.

    Returns
    -------
    None
    """
    ia = cl.load_sample("ia_sample")
    fitted = cl.IncrementalAdditive().fit(ia["loss"], sample_weight=ia["exposure"].latest_diagonal)
    np.testing.assert_allclose(
        fitted.cum_zeta_.values.flatten(),
        [0.888447, 0.645235, 0.423275, 0.269296, 0.127443, 0.036770],
        atol=atol,
    )
