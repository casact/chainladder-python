import chainladder as cl
import numpy as np
import pytest

from sklearn.base import clone


# Exhibit C1 of Sahasrabuddhe (2010), Example 1: the claim size model
# parameters at exposure year 10's cost level.
THETA_10 = [
    28138,
    84242,
    133998,
    182460,
    204649,
    228245,
    252830,
    265063,
    275707,
    280000,
]


@pytest.fixture
def paper_trend(genins):
    """Exhibit B3: the combined exposure- and calendar-period cost level index."""

    def index(**kwargs):
        return 1 / cl.Trend(**kwargs, full_triangle=True).fit(genins).trend_

    return index(
        trends=[0.02, 0.05, 0.02],
        dates=[
            ("2006-12-31", None),
            ("2007-12-31", "2006-12-31"),
            ("2010-12-31", "2007-12-31"),
        ],
        axis="origin",
        base_period=2001,
    ) * index(
        trends=[0.01, -0.05, 0.01],
        dates=[
            ("2002-12-31", None),
            ("2003-12-31", "2002-12-31"),
            (None, "2003-12-31"),
        ],
        axis="valuation",
        base_period="2001-12",
    )


@pytest.fixture
def paper_model(genins, paper_trend):
    return cl.Sahasrabuddhe(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        data_limit=1_000_000,
        target_limit=500_000,
        base_period=2010,
    ).fit(genins)


def test_reproduces_published_exhibit_d1(paper_model):
    """
    Exhibit D1, the restated triangle.

    One cell is excluded: the paper's exhibit A1 prints 359,840 at exposure
    period 8 age 12, where genins -- and the Taylor-Ashe triangle it comes
    from -- has 359,480. A transposed digit in the paper's own input, so its
    D1 is internally consistent with it and ours is consistent with genins.
    """
    published = np.array(
        [
            [
                452881,
                1419731,
                2279040,
                2793772,
                3377947,
                3978376,
                4038185,
                4142394,
                4349865,
                4405265,
            ],
            [
                432566,
                1610197,
                2766439,
                4100116,
                4538378,
                4794873,
                5260266,
                5484617,
                5887561,
                None,
            ],
            [
                368296,
                1634013,
                2745402,
                3840401,
                4623708,
                4671636,
                5090015,
                5324759,
                None,
                None,
            ],
            [
                382236,
                1741436,
                2636785,
                4330559,
                4539482,
                4811270,
                4902613,
                None,
                None,
                None,
            ],
            [
                529368,
                1353815,
                2481777,
                3242747,
                3722312,
                4131335,
                None,
                None,
                None,
                None,
            ],
            [459320, 1541795, 2468413, 3244186, 3922258, None, None, None, None, None],
            [481990, 1405037, 2583135, 3571425, None, None, None, None, None, None],
            [381903, 1504277, 2968365, None, None, None, None, None, None, None],
            [388062, 1400758, None, None, None, None, None, None, None, None],
            [344014, None, None, None, None, None, None, None, None, None],
        ],
        dtype="float64",
    )
    comparable = ~np.isnan(published)
    comparable[7, 0] = False  # the A1 typo, see docstring

    adjusted = paper_model.adjusted_.set_backend("numpy").values[0, 0]
    assert np.abs(adjusted[comparable] - published[comparable]).max() <= 3.0


def test_reproduces_published_exhibits_d2_and_d3(genins, paper_model):
    """Exhibits D2 and D3 follow from ordinary Development on the restated data."""
    dev = cl.Development(average="volume").fit(paper_model.adjusted_)
    published_ldf = [3.511, 1.714, 1.399, 1.147, 1.076, 1.057, 1.039, 1.063, 1.013]
    published_cdf = [12.291, 3.501, 2.042, 1.460, 1.273, 1.183, 1.119, 1.077, 1.013]
    assert np.abs(dev.ldf_.values[0, 0, 0] - published_ldf).max() <= 0.0005
    assert np.abs(dev.cdf_.values[0, 0, 0] - published_cdf).max() <= 0.002


def test_pipeline_matches_a_manual_fit(genins, paper_trend, paper_model):
    """The transformer hands adjusted values downstream, not the original."""
    pipe = cl.Pipeline([
        (
            "basis",
            cl.Sahasrabuddhe(
                means=dict(zip(genins.development, THETA_10)),
                trend=paper_trend,
                data_limit=1_000_000,
                target_limit=500_000,
                base_period=2010,
            ),
        ),
        ("dev", cl.Development(average="volume")),
    ]).fit(genins)
    manual = cl.Development(average="volume").fit(paper_model.adjusted_)
    assert np.allclose(pipe.named_steps.dev.ldf_.values, manual.ldf_.values)
    # Guard the premise: fitting the raw triangle would give something else.
    raw = cl.Development(average="volume").fit(genins)
    assert not np.allclose(pipe.named_steps.dev.ldf_.values, raw.ldf_.values)


def test_preserves_the_index_axis(clrd):
    """
    The adjustment is per-cell, so a Triangle with many index entries has to
    come back with all of them. Slicing the base period with [0, 0, ...]
    instead of [..., ] would silently return one.
    """
    tri = clrd["CumPaidLoss"]
    trend = 1 / cl.Trend(0.04, axis="origin", full_triangle=True).fit(tri).trend_
    means = dict(zip(tri.development, np.linspace(30000, 300000, 10)))
    model = cl.Sahasrabuddhe(
        means=means, trend=trend, data_limit=1_000_000, target_limit=500_000
    ).fit(tri)
    assert model.adjusted_.shape == tri.shape
    assert np.array_equal(
        np.isnan(model.adjusted_.set_backend("numpy").values),
        np.isnan(tri.set_backend("numpy").values),
    )


def test_base_period_defaults_to_the_latest_origin(genins, paper_trend):
    kwargs = dict(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        data_limit=1_000_000,
        target_limit=500_000,
    )
    implicit = cl.Sahasrabuddhe(**kwargs).fit(genins).adjusted_.set_backend("numpy")
    explicit = (
        cl
        .Sahasrabuddhe(**kwargs, base_period=2010)
        .fit(genins)
        .adjusted_.set_backend("numpy")
    )
    assert np.allclose(np.nan_to_num(implicit.values), np.nan_to_num(explicit.values))


def test_trend_rescaling_is_immaterial(genins, paper_trend, paper_model):
    """
    Only ratios of the index are used, so its base period cancels. This is why
    the estimator does not care how the caller anchored the trend.
    """
    rescaled = cl.Sahasrabuddhe(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend * 7.3,
        data_limit=1_000_000,
        target_limit=500_000,
        base_period=2010,
    ).fit(genins)
    assert np.allclose(
        np.nan_to_num(rescaled.adjusted_.set_backend("numpy").values),
        np.nan_to_num(paper_model.adjusted_.set_backend("numpy").values),
    )


def test_rejects_a_trend_from_a_different_triangle(genins):
    """Matching shapes are not enough -- the periods have to match too."""
    other = cl.load_sample("clrd")["CumPaidLoss"]
    trend = 1 / cl.Trend(0.04, axis="origin", full_triangle=True).fit(other).trend_
    assert trend.shape[-2:] == genins.shape[-2:]  # would pass a shape-only check
    with pytest.raises(ValueError, match="does not share X's origin axis"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=trend,
            data_limit=1_000_000,
            target_limit=500_000,
        ).fit(genins)


def test_rejects_a_trend_with_gaps_in_the_base_period(genins):
    """
    Without full_triangle=True the base period's row is mostly empty, and every
    cell divides by it -- which would quietly turn most of the result to NaN.
    """
    trend = 1 / cl.Trend(0.05, axis="origin").fit(genins).trend_
    with pytest.raises(ValueError, match="full_triangle=True"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=trend,
            data_limit=1_000_000,
            target_limit=500_000,
        ).fit(genins)


@pytest.mark.parametrize("missing", ["trend", "data_limit", "target_limit"])
def test_required_arguments_raise(genins, paper_trend, missing):
    kwargs = dict(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        data_limit=1_000_000,
        target_limit=500_000,
    )
    kwargs[missing] = None
    with pytest.raises(ValueError, match=missing):
        cl.Sahasrabuddhe(**kwargs).fit(genins)


def test_base_period_outside_the_triangle_raises(genins, paper_trend):
    with pytest.raises(ValueError, match="does not match any origin"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=paper_trend,
            data_limit=1_000_000,
            target_limit=500_000,
            base_period=1800,
        ).fit(genins)


def test_params_survive_sklearn_clone():
    estimator = cl.Sahasrabuddhe(
        means={12: 1.0}, data_limit=1e6, target_limit=5e5, base_period=2010
    )
    assert clone(estimator).get_params() == estimator.get_params()


def test_lev_at_zero_is_zero(genins):
    """
    `0 / means_` is NaN in Triangle arithmetic and `1 - NaN` is then 1, so the
    closed form would hand back the means untouched at a limit of zero. That
    would silently corrupt every layer attaching at 0, which is the common case.
    """
    lev = cl.LEV(means=dict(zip(genins.development, THETA_10))).fit(genins)
    assert np.allclose(lev.at(0).set_backend("numpy").values, 0.0)
    assert np.allclose(
        lev.layer(0, 500_000).set_backend("numpy").values,
        lev.at(500_000).set_backend("numpy").values,
    )


def test_adjusted_keeps_the_triangles_valuation_date(genins, paper_model):
    """
    The limited expected values span the whole rectangle, so the product that
    builds `adjusted_` inherits their valuation date unless it is put back.
    Leaving it would give the result a nan_triangle and a latest diagonal
    belonging to a Triangle running years past the data.
    """
    assert paper_model.adjusted_.valuation_date == genins.valuation_date
    adjusted = paper_model.adjusted_.set_backend("numpy")
    assert np.array_equal(
        np.isnan(np.asarray(adjusted.nan_triangle, dtype="float64")),
        np.isnan(np.asarray(genins.set_backend("numpy").nan_triangle, dtype="float64")),
    )
    assert (
        np.isfinite(
            paper_model.adjusted_.latest_diagonal.set_backend("numpy").values
        ).sum()
        == genins.shape[-2]
    )


@pytest.fixture
def paper_development(paper_model):
    return cl.Development(average="volume").fit(paper_model.adjusted_)


@pytest.mark.parametrize(
    ("name", "layer", "published"),
    [
        (
            "E1",
            (0.0, None),
            [12.633, 3.590, 2.193, 1.531, 1.319, 1.211, 1.133, 1.084, 1.015, 1.000],
        ),
        (
            "E2",
            (500_000, 2_000_000),
            [None, 652.420, 32.802, 5.924, 3.380, 2.175, 1.499, 1.257, 1.057, 1.000],
        ),
        (
            "E3",
            (2_000_000, np.inf),
            [
                None,
                None,
                84538.278,
                279.503,
                48.056,
                11.155,
                3.254,
                1.887,
                1.183,
                1.000,
            ],
        ),
        (
            "E4",
            (0.0, 1_000_000),
            [13.776, 3.913, 2.375, 1.629, 1.387, 1.255, 1.155, 1.096, 1.018, 1.000],
        ),
    ],
)
def test_by_layer_reproduces_published_exhibits(
    paper_model, paper_development, name, layer, published
):
    """
    Exhibits E1 to E4 -- exposure period 1's row of each. The paper prints
    "very large" where a layer has essentially no expected loss at that age,
    and those cells are non-finite here for the same reason, so only the cells
    it prints a number for are compared.
    """
    factors = paper_model.by_layer(paper_development, *layer)
    computed = factors.set_backend("numpy").values[0, 0, 0]
    published = np.array(published, dtype="float64")
    comparable = np.isfinite(published)
    deviation = np.abs(computed[comparable] / published[comparable] - 1).max()
    assert deviation <= 0.01, f"{name} deviates from the paper by {deviation:.5f}"


def test_by_layer_defaults_to_the_fitted_layer(paper_model, paper_development):
    """Omitting the layer reproduces the one the pattern was fitted on."""
    implicit = paper_model.by_layer(paper_development)
    explicit = paper_model.by_layer(paper_development, 0.0, paper_model.target_limit)
    assert np.allclose(
        np.nan_to_num(implicit.set_backend("numpy").values),
        np.nan_to_num(explicit.set_backend("numpy").values),
    )


def test_by_layer_full_triangle_fills_the_rectangle(paper_model, paper_development):
    masked = paper_model.by_layer(paper_development).set_backend("numpy").values
    full = (
        paper_model
        .by_layer(paper_development, full_triangle=True)
        .set_backend("numpy")
        .values
    )
    observed = ~np.isnan(masked)
    assert np.isnan(masked).sum() > 0
    assert np.isnan(full).sum() == 0
    assert np.allclose(full[observed], masked[observed], rtol=1e-12)
    # Every row reaches ultimate by the last age, whatever its cost level.
    assert np.allclose(full[0, 0, :, -1], 1.0)


def test_by_layer_accepts_several_pattern_forms(paper_model, paper_development):
    """A fitted estimator, a Triangle it transformed, or a bare cdf_."""
    reference = paper_model.by_layer(paper_development).set_backend("numpy").values
    transformed = cl.Development(average="volume").fit_transform(paper_model.adjusted_)
    for pattern in (transformed, paper_development.cdf_):
        assert np.allclose(
            np.nan_to_num(paper_model.by_layer(pattern).set_backend("numpy").values),
            np.nan_to_num(reference),
        )


def test_by_layer_aligns_the_pattern_by_age_not_position(paper_model):
    """
    A fitted tail makes cdf_ longer than the triangle's development axis, so
    zipping the two by position would silently misalign every factor.
    """
    transformed = cl.Development(average="volume").fit_transform(paper_model.adjusted_)
    tailed = cl.TailConstant(1.05).fit(transformed)
    assert tailed.cdf_.shape[-1] > paper_model.adjusted_.shape[-1]

    factors = paper_model.by_layer(tailed).set_backend("numpy").values
    no_tail = paper_model.by_layer(transformed).set_backend("numpy").values
    # The tail scales every factor by 1.05; ages beyond the triangle are dropped.
    observed = ~np.isnan(no_tail)
    assert np.allclose(factors[observed] / no_tail[observed], 1.05, rtol=1e-9)


def test_by_layer_rejects_something_that_is_not_a_pattern(paper_model, genins):
    with pytest.raises(ValueError, match="cumulative development factors"):
        paper_model.by_layer(genins)


def test_by_layer_style_returns_the_latest_diagonal(paper_model, paper_development):
    """
    The pattern is the latest diagonal of the factor surface -- one cell per
    row, each exposure period at its own current age and cost level. Reading a
    single row instead would apply one period's cost level to all of them.
    """
    surface = paper_model.by_layer(paper_development).set_backend("numpy").values[0, 0]
    pattern = paper_model.by_layer(paper_development, style="cdf")

    origins = surface.shape[0]
    diagonal = np.array([surface[origins - 1 - j, j] for j in range(origins)])
    assert np.allclose(
        pattern.set_backend("numpy").values[0, 0, 0][:origins], diagonal, rtol=1e-12
    )


def test_by_layer_style_labels(paper_model, paper_development):
    cdf = paper_model.by_layer(paper_development, style="cdf")
    ldf = paper_model.by_layer(paper_development, style="ldf")
    assert list(cdf.development)[:3] == ["12-Ult", "24-Ult", "36-Ult"]
    assert list(ldf.development)[:3] == ["12-24", "24-36", "36-48"]


def test_by_layer_ldf_and_cdf_are_consistent(paper_model, paper_development):
    """Chaining the age-to-age factors from the right rebuilds the cumulative ones."""
    cdf = paper_model.by_layer(paper_development, style="cdf")
    ldf = paper_model.by_layer(paper_development, style="ldf")
    rebuilt = np.cumprod(ldf.set_backend("numpy").values[0, 0, 0][::-1])[::-1]
    assert np.allclose(rebuilt, cdf.set_backend("numpy").values[0, 0, 0], rtol=1e-12)


def test_by_layer_style_respects_the_layer(paper_model, paper_development):
    """A wider layer truncates less, so it develops faster at every age."""
    basic = paper_model.by_layer(paper_development, style="cdf")
    wider = paper_model.by_layer(paper_development, 0, np.inf, style="cdf")
    basic = basic.set_backend("numpy").values[0, 0, 0]
    wider = wider.set_backend("numpy").values[0, 0, 0]
    assert (wider[:-1] > basic[:-1]).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"style": "nonsense"}, "must be 'ldf', 'cdf' or None"),
        ({"style": "ldf", "full_triangle": True}, "cannot be combined"),
    ],
)
def test_by_layer_style_argument_errors(paper_model, paper_development, kwargs, match):
    with pytest.raises(ValueError, match=match):
        paper_model.by_layer(paper_development, **kwargs)


def test_by_layer_style_rejects_an_incomplete_diagonal(qtr):
    """
    Quarterly development on annual origins means the latest diagonal touches
    only every fourth age. Filling the rest would assert no development at
    those ages, which is false -- so it raises instead.
    """
    tri = qtr["paid"]
    trend = 1 / cl.Trend(0.03, axis="origin", full_triangle=True).fit(tri).trend_
    means = dict(zip(tri.development, np.linspace(20000, 200000, tri.shape[-1])))
    model = cl.Sahasrabuddhe(
        means=means, trend=trend, data_limit=1_000_000, target_limit=500_000
    ).fit(tri)
    development = cl.Development(average="volume").fit(model.adjusted_)

    with pytest.raises(ValueError, match="has reached development age"):
        model.by_layer(development, style="cdf")
    # The surface itself is still perfectly well defined.
    assert model.by_layer(development).shape == tri.shape
