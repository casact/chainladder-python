import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone


def test_trend1(clrd):
    tri = clrd[["CumPaidLoss", "EarnedPremDIR"]].sum()
    lhs = (
        cl
        .CapeCod(0.05)
        .fit(tri["CumPaidLoss"], sample_weight=tri["EarnedPremDIR"].latest_diagonal)
        .ibnr_
    )
    rhs = (
        cl
        .CapeCod()
        .fit(
            cl.Trend(0.05).fit_transform(tri["CumPaidLoss"]),
            sample_weight=tri["EarnedPremDIR"].latest_diagonal,
        )
        .ibnr_
    )
    assert np.round(lhs, 0) == np.round(rhs, 0)


def test_trend2(raa):
    """Two 5% segments that meet at 1985 compound to a single 5% trend."""
    trended = (
        cl
        .Trend(
            trends=[0.05, 0.05],
            dates=[(None, "1985"), ("1985", None)],
            axis="origin",
        )
        .fit(raa)
        .trend_
        * raa
    )
    expected = raa.trend(0.05, axis="origin")
    assert np.allclose(
        np.nan_to_num(trended.set_backend("numpy").values),
        np.nan_to_num(expected.set_backend("numpy").values),
        atol=1e-6,
    )


@pytest.mark.parametrize("trend", [0.05, 0.0, -0.03])
def test_trend_origin_factors_compound_to_the_latest_origin(raa, trend):
    """
    On the origin axis every cell of an origin year carries the same factor:
    (1 + trend) raised to the years from that origin to the latest, which is
    1.0 because it is the anchor. raa runs 1981 through 1990.
    """
    trend_ = cl.Trend(trend, axis="origin").fit(raa).trend_.set_backend("numpy").values
    expected = np.tile(((1 + trend) ** np.arange(9, -1, -1))[:, None], (1, 10))
    observed = ~np.isnan(trend_[0, 0])
    assert np.allclose(trend_[0, 0][observed], expected[observed], rtol=1e-12)


@pytest.mark.parametrize("trend", [0.05, 0.0, -0.03])
def test_trend_valuation_factors_compound_to_the_valuation_date(raa, trend):
    """
    On the valuation axis the factor follows the cell's diagonal rather than its
    row: (1 + trend) raised to the years from that cell's valuation to the
    valuation date. raa cell (i, j) is valued at year-end 1981 + i + j.
    """
    trend_ = (
        cl.Trend(trend, axis="valuation").fit(raa).trend_.set_backend("numpy").values
    )
    i, j = np.indices((10, 10))
    expected = (1 + trend) ** (9 - i - j)
    observed = ~np.isnan(trend_[0, 0])
    assert np.allclose(trend_[0, 0][observed], expected[observed], rtol=1e-12)


@pytest.mark.parametrize("axis", ["origin", "valuation"])
def test_trend_segments_apply_over_their_own_date_ranges(raa, axis):
    """
    Each segment trends only across the dates it covers, and the segments
    compound. Here 10% runs from 1990 back to the 1985 boundary -- 1.0, 1.1,
    1.21, ... up the origins -- and 5% carries the rest of the way back to 1981.
    The boundary is 1985-01-01 while origin periods end on 12-31, so the years
    either side of it are split part-way through.
    """
    trend_ = (
        cl
        .Trend(trends=[0.05, 0.10], dates=[("1985", None), (None, "1985")], axis=axis)
        .fit(raa)
        .trend_.set_backend("numpy")
        .values
    )
    diagonal = [
        2.042868,
        1.945589,
        1.852942,
        1.764707,
        1.610510,
        1.464100,
        1.331000,
        1.210000,
        1.100000,
        1.000000,
    ]
    if axis == "origin":
        # Constant across development: the factor depends only on the origin.
        expected = np.tile(np.array(diagonal)[:, None], (1, 10))
    else:
        # Constant along each diagonal: the factor depends only on the valuation.
        # Cells past the last diagonal fall outside the list and stay NaN, which
        # `observed` then drops -- but only after `trend_` has agreed they are NaN.
        i, j = np.indices((10, 10))
        expected = np.where(
            i + j < 10, np.array(diagonal)[np.minimum(i + j, 9)], np.nan
        )
    observed = ~np.isnan(trend_[0, 0])
    assert np.allclose(trend_[0, 0][observed], expected[observed], rtol=1e-6)


@pytest.mark.parametrize("axis", ["origin", "valuation"])
def test_trend_is_shaped_like_the_triangle(raa, axis):
    """By default `trend_` is defined exactly where the Triangle is."""
    trend_ = cl.Trend(0.05, axis=axis).fit(raa).trend_.set_backend("numpy").values
    assert np.array_equal(np.isnan(trend_), np.isnan(raa.set_backend("numpy").values))


def test_trend_leaves_internal_gaps_empty(clrd):
    """
    Cells missing inside the triangle are empty in `trend_` too, not just the
    ones past the latest diagonal.
    """
    tri = clrd["CumPaidLoss"].set_backend("numpy")
    # Guard the premise: some cells are missing inside the triangle, not merely
    # beyond the latest diagonal. Without these the test proves nothing.
    inside = ~np.isnan(np.asarray(tri.nan_triangle, dtype="float64"))
    assert np.isnan(tri.values[:, :, inside]).any()

    trend_ = cl.Trend(0.05).fit(tri).trend_.values
    assert np.array_equal(np.isnan(trend_), np.isnan(tri.values))


@pytest.mark.parametrize("axis", ["origin", "valuation"])
def test_trend_full_fills_the_rectangle(raa, axis):
    """
    `full_triangle` returns a factor for every cell of the origin x development rectangle,
    including the cells past the valuation date that the default leaves empty.
    """
    default = cl.Trend(0.05, axis=axis).fit(raa).trend_.set_backend("numpy").values
    full_triangle = (
        cl
        .Trend(0.05, axis=axis, full_triangle=True)
        .fit(raa)
        .trend_.set_backend("numpy")
        .values
    )
    assert np.isnan(default).sum() > 0
    assert np.isnan(full_triangle).sum() == 0
    assert full_triangle.shape == default.shape


@pytest.mark.parametrize("axis", ["origin", "valuation"])
def test_trend_full_agrees_with_default_where_both_defined(raa, axis):
    """
    Filling the rectangle must not disturb the cells the default already covers.
    raa's valuation date lands on an origin period end, so the two agree exactly;
    a Triangle valued part-way through an origin period would not, because the
    default clips that final step and `full_triangle` does not.
    """
    default = cl.Trend(0.05, axis=axis).fit(raa).trend_.set_backend("numpy").values
    full_triangle = (
        cl
        .Trend(0.05, axis=axis, full_triangle=True)
        .fit(raa)
        .trend_.set_backend("numpy")
        .values
    )
    observed = ~np.isnan(default)
    assert np.allclose(default[observed], full_triangle[observed], rtol=1e-12)


@pytest.mark.parametrize("axis", ["origin", "valuation"])
def test_trend_full_keeps_accruing_past_the_valuation_date(raa, axis):
    """
    Cells beyond the valuation date must carry the trend forward rather than
    repeat the boundary value. `start` is both the zero point and the clip
    boundary in Triangle.trend, so getting this wrong flattens the extension.
    """
    full_triangle = cl.Trend(0.05, axis=axis, full_triangle=True).fit(raa).trend_
    default = cl.Trend(0.05, axis=axis).fit(raa).trend_
    future = np.isnan(default.set_backend("numpy").values)
    extended = full_triangle.set_backend("numpy").values[future]
    assert len(np.unique(np.round(extended, 6))) > 1


def test_trend_base_period_moves_the_anchor(raa):
    """
    The anchored period gets a factor of 1.0, and every other period is stated
    against it. `trend_` stays a multiplier TO that period's level, so later
    origins fall below 1.0 when anchoring on the earliest.
    """
    trend_ = (
        cl
        .Trend(0.05, axis="origin", base_period=1981)
        .fit(raa)
        .trend_.set_backend("numpy")
        .values
    )
    assert trend_[0, 0, 0, 0] == pytest.approx(1.0)
    assert trend_[0, 0, -1, 0] == pytest.approx(1 / 1.05**9, rel=1e-6)


@pytest.mark.parametrize("full_triangle", [False, True])
def test_trend_base_period_at_latest_origin_is_a_noop(raa, full_triangle):
    """
    The default already anchors on the latest period, so naming it explicitly
    must change nothing.
    """
    implicit = (
        cl.Trend(0.05, axis="origin", full_triangle=full_triangle).fit(raa).trend_
    )
    explicit = (
        cl
        .Trend(0.05, axis="origin", full_triangle=full_triangle, base_period=1990)
        .fit(raa)
        .trend_
    )
    implicit = implicit.set_backend("numpy").values
    explicit = explicit.set_backend("numpy").values
    assert np.allclose(np.nan_to_num(implicit), np.nan_to_num(explicit), rtol=1e-12)


def test_trend_base_period_rescales_uniformly(raa):
    """
    Rebasing only moves the zero point, so every factor shifts by one common
    ratio rather than changing shape.
    """
    default = cl.Trend(0.05, axis="origin").fit(raa).trend_.set_backend("numpy").values
    rebased = (
        cl
        .Trend(0.05, axis="origin", base_period=1985)
        .fit(raa)
        .trend_.set_backend("numpy")
        .values
    )
    observed = ~np.isnan(default)
    ratio = default[observed] / rebased[observed]
    assert np.allclose(ratio, ratio[0], rtol=1e-12)


def test_trend_base_period_defined_where_data_is_missing(clrd):
    """
    The anchor is read off a full grid, not off the data, so a base period the
    Triangle happens to be missing still rebases rather than poisoning the
    result with NaN.
    """
    tri = clrd["CumPaidLoss"]
    trend_ = cl.Trend(0.05, axis="origin", base_period=1995).fit(tri).trend_
    default = cl.Trend(0.05, axis="origin").fit(tri).trend_
    trend_ = trend_.set_backend("numpy").values
    default = default.set_backend("numpy").values
    assert np.array_equal(np.isnan(trend_), np.isnan(default))


def test_trend_base_period_ignores_the_development_grain(qtr):
    """
    Only the trended axis is searched: qtr has quarterly development but annual
    origins, so a bare year still resolves against the origins alone.
    """
    trend_ = (
        cl
        .Trend(0.05, axis="origin", full_triangle=True, base_period=1997)
        .fit(qtr["paid"])
        .trend_.set_backend("numpy")
        .values
    )
    position = list(qtr.origin.astype(str)).index("1997")
    assert trend_[0, 0, position, 0] == pytest.approx(1.0)


def _fiscal_year_origin_triangle():
    """
    The smallest Triangle with a fiscal (July-June) annual origin axis: three
    fiscal years, each valued through fiscal year-end.
    """
    years = pd.period_range("2018", "2020", freq="Y-JUN")
    rows = [
        (origin.to_timestamp(how="s"), valuation.to_timestamp(how="e"), 1.0)
        for i, origin in enumerate(years)
        for valuation in years[i:]
    ]
    return cl.Triangle(
        pd.DataFrame(rows, columns=["origin", "valuation", "paid"]),
        origin="origin",
        development="valuation",
        columns=["paid"],
        cumulative=True,
        trailing=True,
    )


def test_trend_default_anchor_preserves_fiscal_origin_frequency():
    """
    Trend should work on a fiscal year.
    """
    tri = _fiscal_year_origin_triangle()
    trend_ = cl.Trend(0.10, axis="origin").fit(tri).trend_

    origins = list(tri.origin.astype(str))
    assert trend_.values[0, 0, origins.index("2020"), 0] == pytest.approx(1.0)
    assert trend_.values[0, 0, origins.index("2019"), 0] == pytest.approx(1.10)
    assert trend_.values[0, 0, origins.index("2018"), 0] == pytest.approx(1.21)


def _quarterly_origin_triangle():
    """
    The smallest Triangle with a quarterly origin axis: the four quarters of 2017,
    each valued through 2017Q4. Enough for a bare year to span several origin
    periods, which is all the base-period resolution tests need.
    """
    quarters = pd.period_range("2017Q1", "2017Q4", freq="Q")
    rows = [
        (origin.to_timestamp(), valuation.to_timestamp(), 1.0)
        for i, origin in enumerate(quarters)
        for valuation in quarters[i:]
    ]
    return cl.Triangle(
        pd.DataFrame(rows, columns=["origin", "valuation", "paid"]),
        origin="origin",
        development="valuation",
        columns=["paid"],
        cumulative=True,
    )


def test_trend_base_period_coarser_than_grain_takes_earliest():
    """
    Against a quarterly origin axis a bare year spans four periods, which carry
    different factors. It resolves to the earliest of them, so 2017 anchors on
    2017Q1 rather than on 2017Q4.
    """
    tri = _quarterly_origin_triangle()
    origins = list(tri.origin.astype(str))

    coarse = (
        cl
        .Trend(0.05, axis="origin", full_triangle=True, base_period=2017)
        .fit(tri)
        .trend_.values
    )
    assert coarse[0, 0, origins.index("2017Q1"), 0] == pytest.approx(1.0)
    assert coarse[0, 0, origins.index("2017Q4"), 0] != pytest.approx(1.0)

    exact = (
        cl
        .Trend(0.05, axis="origin", full_triangle=True, base_period="2017Q3")
        .fit(tri)
        .trend_.values
    )
    assert exact[0, 0, origins.index("2017Q3"), 0] == pytest.approx(1.0)
    assert exact[0, 0, origins.index("2017Q1"), 0] != pytest.approx(1.0)


@pytest.mark.parametrize("axis", ["origin", "valuation"])
def test_trend_base_period_outside_the_triangle_raises(raa, axis):
    """A base period that matches nothing should say so rather than yield NaN."""
    with pytest.raises(ValueError, match="does not match"):
        cl.Trend(0.05, axis=axis, base_period=1800).fit(raa)


@pytest.mark.parametrize("full_triangle", [False, True])
@pytest.mark.parametrize("base_period", [None, 1981])
def test_trend_preserves_backend(raa, full_triangle, base_period):
    """
    The full grid is built in numpy internally, so the fitted factors still have
    to come back on whatever backend was passed in.
    """
    trend_ = (
        cl
        .Trend(0.05, full_triangle=full_triangle, base_period=base_period)
        .fit(raa)
        .trend_
    )
    assert trend_.array_backend == raa.array_backend


def test_trend_new_params_survive_sklearn_clone():
    """
    sklearn reads constructor arguments back off identically named attributes,
    so a mismatch breaks get_params, clone, Pipeline and serialization.
    """
    estimator = cl.Trend(0.05, base_period=1981, full_triangle=True)
    params = estimator.get_params()
    assert params["base_period"] == 1981
    assert params["full_triangle"] is True
    assert clone(estimator).get_params() == params
