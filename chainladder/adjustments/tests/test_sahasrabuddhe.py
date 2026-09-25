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
        basic_limit=500_000,
        target_layer=(0, 500_000),
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

    adjusted = paper_model.triangle_.set_backend("numpy").values[0, 0]
    assert np.abs(adjusted[comparable] - published[comparable]).max() <= 3.0


def test_reproduces_published_exhibits_d2_and_d3(genins, paper_model):
    """Exhibits D2 and D3 follow from ordinary Development on the restated data."""
    dev = cl.Development(average="volume").fit(paper_model.triangle_)
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
                basic_limit=500_000,
                target_layer=(0, 500_000),
                base_period=2010,
            ),
        ),
        ("dev", cl.Development(average="volume")),
    ]).fit(genins)
    manual = cl.Development(average="volume").fit(paper_model.triangle_)
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
        means=means,
        trend=trend,
        data_limit=1_000_000,
        basic_limit=500_000,
        target_layer=(0, 500_000),
    ).fit(tri)
    assert model.triangle_.shape == tri.shape
    assert np.array_equal(
        np.isnan(model.triangle_.set_backend("numpy").values),
        np.isnan(tri.set_backend("numpy").values),
    )


def test_base_period_defaults_to_the_latest_origin(genins, paper_trend):
    kwargs = dict(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        data_limit=1_000_000,
        basic_limit=500_000,
        target_layer=(0, 500_000),
    )
    implicit = cl.Sahasrabuddhe(**kwargs).fit(genins).triangle_.set_backend("numpy")
    explicit = (
        cl
        .Sahasrabuddhe(**kwargs, base_period=2010)
        .fit(genins)
        .triangle_.set_backend("numpy")
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
        basic_limit=500_000,
        target_layer=(0, 500_000),
        base_period=2010,
    ).fit(genins)
    assert np.allclose(
        np.nan_to_num(rescaled.triangle_.set_backend("numpy").values),
        np.nan_to_num(paper_model.triangle_.set_backend("numpy").values),
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
            basic_limit=500_000,
            target_layer=(0, 500_000),
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
            basic_limit=500_000,
            target_layer=(0, 500_000),
        ).fit(genins)


@pytest.mark.parametrize(
    "missing", ["trend", "data_limit", "basic_limit", "target_layer"]
)
def test_required_arguments_raise(genins, paper_trend, missing):
    kwargs = dict(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        data_limit=1_000_000,
        basic_limit=500_000,
        target_layer=(0, 500_000),
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
            basic_limit=500_000,
            target_layer=(0, 500_000),
            base_period=1800,
        ).fit(genins)


def test_params_survive_sklearn_clone():
    estimator = cl.Sahasrabuddhe(
        means={12: 1.0},
        data_limit=1e6,
        basic_limit=5e5,
        target_layer=(0, 5e5),
        base_period=2010,
    )
    assert clone(estimator).get_params() == estimator.get_params()


def test_lev_at_zero_is_zero(genins):
    """
    `0 / means_` is NaN in Triangle arithmetic and `1 - NaN` is then 1, so the
    closed form would hand back the means untouched at a limit of zero. That
    would silently corrupt every layer attaching at 0, which is the common case.
    """
    means = dict(zip(genins.development, THETA_10))
    # A degenerate layer is rejected outright, so zero is reached as an
    # attachment rather than as a limit.
    with pytest.raises(ValueError, match="must exceed attachment"):
        cl.LEV(means=means, limit=0).fit(genins)
    assert np.allclose(
        cl
        .LEV(means=means, attachment=0, limit=500_000)
        .fit(genins)
        .lev_.set_backend("numpy")
        .values,
        cl.LEV(means=means, limit=500_000).fit(genins).lev_.set_backend("numpy").values,
    )


def _means_row(genins, year, unmask=True):
    """THETA_10 as a one-origin Triangle labelled ``year``."""
    base = list(genins.origin.year).index(year)
    row = genins.iloc[0, 0, base : base + 1, :].copy()
    if unmask:
        row.valuation_date = row.valuation.max()
    return (row * 0 + 1).fillna(1) * np.array(THETA_10, dtype="float64")


@pytest.mark.parametrize("base_period", [None, 2005])
def test_one_origin_triangle_matches_a_mapping(genins, paper_trend, base_period):
    year = 2010 if base_period is None else base_period
    from_triangle = cl.LEV(
        means=_means_row(genins, year), trend=paper_trend, base_period=base_period
    ).fit(genins)
    from_mapping = cl.LEV(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        base_period=base_period,
    ).fit(genins)
    assert np.allclose(
        from_triangle.means_.set_backend("numpy").values,
        from_mapping.means_.set_backend("numpy").values,
    )


def test_rejects_a_one_origin_triangle_off_the_base_period(genins, paper_trend):
    """
    A single row broadcasts over every origin whatever it is labelled, so means
    stated at 2005 would be restated as though they were at 2010.
    """
    with pytest.raises(ValueError, match="Pass base_period=2005"):
        cl.LEV(means=_means_row(genins, 2005), trend=paper_trend).fit(genins)


def test_one_origin_triangle_label_is_immaterial_without_a_trend(genins):
    lev = cl.LEV(means=_means_row(genins, 2005)).fit(genins)
    assert lev.means_.shape == (1, 1, 1, 10)


@pytest.mark.parametrize("with_trend", [True, False])
def test_rejects_a_masked_one_origin_triangle(genins, paper_trend, with_trend):
    """
    A row sliced straight from the Triangle is masked past its first age, which
    would quietly turn every later age to NaN.
    """
    trend = paper_trend if with_trend else None
    with pytest.raises(ValueError, match="gaps in its single origin row"):
        cl.LEV(
            means=_means_row(genins, 2010, unmask=False), trend=trend
        ).fit(genins)


def test_triangle_keeps_the_triangles_valuation_date(genins, paper_model):
    """
    The limited expected values span the whole rectangle, so the product that
    builds `triangle_` inherits their valuation date unless it is put back.
    Leaving it would give the result a nan_triangle and a latest diagonal
    belonging to a Triangle running years past the data.
    """
    assert paper_model.triangle_.valuation_date == genins.valuation_date
    adjusted = paper_model.triangle_.set_backend("numpy")
    assert np.array_equal(
        np.isnan(np.asarray(adjusted.nan_triangle, dtype="float64")),
        np.isnan(np.asarray(genins.set_backend("numpy").nan_triangle, dtype="float64")),
    )
    assert (
        np.isfinite(
            paper_model.triangle_.latest_diagonal.set_backend("numpy").values
        ).sum()
        == genins.shape[-2]
    )


@pytest.fixture
def paper_development(paper_model):
    return cl.Development(average="volume").fit(paper_model.triangle_)


@pytest.fixture
def paper_layer(paper_trend, genins):
    """Sahasrabuddhe configured for a layer, ready to fit a pattern."""

    def make(**kwargs):
        return cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=paper_trend,
            data_limit=1_000_000,
            basic_limit=500_000,
            base_period=2010,
            **kwargs,
        )

    return make


@pytest.mark.parametrize(
    ("name", "layer", "published"),
    [
        (
            "E1",
            {"target_layer": (0, 500_000)},
            [12.633, 3.590, 2.193, 1.531, 1.319, 1.211, 1.133, 1.084, 1.015, 1.000],
        ),
        (
            "E2",
            {"target_layer": (500_000, 2_000_000)},
            [None, 652.420, 32.802, 5.924, 3.380, 2.175, 1.499, 1.257, 1.057, 1.000],
        ),
        (
            "E3",
            {"target_layer": (2_000_000, np.inf)},
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
            {"target_layer": (0, 1_000_000)},
            [13.776, 3.913, 2.375, 1.629, 1.387, 1.255, 1.155, 1.096, 1.018, 1.000],
        ),
    ],
)
def test_reproduces_published_exhibits_e1_to_e4(
    paper_layer, paper_development, name, layer, published
):
    """
    Exhibits E1 to E4 -- exposure period 1's row of each. The paper prints
    "very large" where a layer has essentially no expected loss at that age,
    and those cells are non-finite here for the same reason, so only the cells
    it prints a number for are compared.
    """
    surface = paper_layer(**layer).fit(paper_development).full_cdf_
    computed = surface.set_backend("numpy").values[0, 0, 0]
    published = np.array(published, dtype="float64")
    comparable = np.isfinite(published)
    deviation = np.abs(computed[comparable] / published[comparable] - 1).max()
    assert deviation <= 0.01, f"{name} deviates from the paper by {deviation:.5f}"


def test_target_equal_to_basic_returns_the_pattern_itself(
    paper_layer, paper_development
):
    """
    Restating onto the layer the pattern already sits on is the identity, on
    the base period's row where no cost level adjustment applies either.
    """
    model = paper_layer(target_layer=(0, 500_000)).fit(paper_development)
    base_row = model.full_cdf_.set_backend("numpy").values[0, 0, -1]
    given = paper_development.cdf_.set_backend("numpy").values[0, 0, 0]
    assert np.allclose(base_row[: len(given)], given, rtol=1e-12)
    assert np.isclose(base_row[-1], 1.0)


def test_full_cdf_fills_the_whole_rectangle(paper_layer, paper_development):
    surface = (
        paper_layer(target_layer=(0, 500_000))
        .fit(paper_development)
        .full_cdf_.set_backend("numpy")
        .values
    )
    assert np.isnan(surface).sum() == 0
    # Every row reaches ultimate by the last age, whatever its cost level.
    assert np.allclose(surface[0, 0, :, -1], 1.0)


def test_accepts_an_estimator_or_a_bare_pattern(paper_layer, paper_development):
    """A fitted development estimator, or the cdf_ Triangle it carries."""
    reference = (
        paper_layer(target_layer=(0, 500_000))
        .fit(paper_development)
        .full_cdf_.set_backend("numpy")
        .values
    )
    from_pattern = (
        paper_layer(target_layer=(0, 500_000))
        .fit(paper_development.cdf_)
        .full_cdf_.set_backend("numpy")
        .values
    )
    assert np.allclose(reference, from_pattern, rtol=1e-12)


def test_a_transformed_triangle_is_claims_not_a_pattern(paper_layer, paper_model):
    """
    A Triangle that has been through a development step carries a cdf_ of its
    own. It is still claims data, and dispatching on that cdf_ would restate it
    as though it were a pattern.
    """
    transformed = cl.Development(average="volume").fit_transform(paper_model.triangle_)
    assert hasattr(transformed, "cdf_") and not transformed.is_pattern

    model = paper_layer(target_layer=(0, 500_000)).fit(transformed)
    assert hasattr(model, "triangle_")
    assert not hasattr(model, "full_cdf_")


def test_aligns_the_pattern_by_age_not_position(paper_layer, paper_model):
    """
    A fitted tail makes cdf_ longer than the development axis, so zipping the
    two by position would silently misalign every factor.
    """
    transformed = cl.Development(average="volume").fit_transform(paper_model.triangle_)
    tailed = cl.TailConstant(1.05).fit(transformed)
    assert tailed.cdf_.shape[-1] > paper_model.triangle_.shape[-1]

    with_tail = (
        paper_layer(target_layer=(0, 500_000))
        .fit(tailed)
        .full_cdf_.set_backend("numpy")
        .values
    )
    without = (
        paper_layer(target_layer=(0, 500_000))
        .fit(cl.Development(average="volume").fit(paper_model.triangle_))
        .full_cdf_.set_backend("numpy")
        .values
    )
    # The tail scales every factor by 1.05; ages beyond the triangle are dropped.
    assert np.allclose(with_tail / without, 1.05, rtol=1e-9)


def test_cdf_is_the_latest_diagonal_of_full_cdf(paper_layer, paper_development):
    """
    The pattern is the diagonal of the surface -- one cell per row, each
    exposure period at its own current age and cost level. Reading a single row
    instead would apply one period's cost level to all of them.
    """
    model = paper_layer(target_layer=(0, 500_000)).fit(paper_development)
    surface = model.full_cdf_.set_backend("numpy").values[0, 0]
    origins = surface.shape[0]
    diagonal = np.array([surface[origins - 1 - j, j] for j in range(origins)])
    assert np.allclose(
        model.cdf_.set_backend("numpy").values[0, 0, 0][:origins],
        diagonal,
        rtol=1e-12,
    )


def test_cdf_and_ldf_labels(paper_layer, paper_development):
    model = paper_layer(target_layer=(0, 500_000)).fit(paper_development)
    assert list(model.cdf_.development)[:3] == ["12-Ult", "24-Ult", "36-Ult"]
    assert list(model.ldf_.development)[:3] == ["12-24", "24-36", "36-48"]


def test_ldf_and_cdf_are_consistent(paper_layer, paper_development):
    """Chaining the age-to-age factors from the right rebuilds the cumulative ones."""
    model = paper_layer(target_layer=(0, 500_000)).fit(paper_development)
    ldf = model.ldf_.set_backend("numpy").values[0, 0, 0]
    rebuilt = np.cumprod(ldf[::-1])[::-1]
    assert np.allclose(
        rebuilt, model.cdf_.set_backend("numpy").values[0, 0, 0], rtol=1e-12
    )


def test_a_wider_layer_develops_faster(paper_layer, paper_development):
    """A wider layer truncates less, so it develops faster at every age."""
    basic = (
        paper_layer(target_layer=(0, 500_000))
        .fit(paper_development)
        .cdf_.set_backend("numpy")
        .values[0, 0, 0]
    )
    wider = (
        paper_layer(target_layer=(0, np.inf))
        .fit(paper_development)
        .cdf_.set_backend("numpy")
        .values[0, 0, 0]
    )
    assert (wider[:-1] > basic[:-1]).all()


def test_degenerate_layer_raises(paper_layer, paper_development):
    with pytest.raises(ValueError, match="exhaustion must be the larger"):
        paper_layer(target_layer=(500_000, 500_000)).fit(paper_development)


def test_triangle_rejects_a_target_other_than_basic(paper_layer, genins):
    """
    Restating claims is a move onto the basis. Naming another layer would be a
    silently different calculation, so it is refused rather than ignored.
    """
    with pytest.raises(ValueError, match="target layer must be"):
        paper_layer(target_layer=(0, 2_000_000)).fit(genins)


def test_incomplete_diagonal_warns_but_keeps_the_rectangle(qtr):
    """
    Quarterly development on annual origins means the latest diagonal touches
    only every fourth age, so no usable pattern comes out. The surface is
    defined for every cell regardless, so it is kept and a warning is raised.
    """
    tri = qtr["paid"]
    trend = 1 / cl.Trend(0.03, axis="origin", full_triangle=True).fit(tri).trend_
    means = dict(zip(tri.development, np.linspace(20000, 200000, tri.shape[-1])))
    kwargs = dict(
        means=means,
        trend=trend,
        data_limit=1_000_000,
        basic_limit=500_000,
    )
    restated = cl.Sahasrabuddhe(**kwargs, target_layer=(0, 500_000)).fit(tri).triangle_
    development = cl.Development(average="volume").fit(restated)

    with pytest.warns(UserWarning, match="has reached development age"):
        model = cl.Sahasrabuddhe(**kwargs, target_layer=(0, 500_000)).fit(development)
    assert not hasattr(model, "cdf_")
    assert model.full_cdf_.shape == trend.shape


def test_full_ldf_rebuilds_full_cdf(paper_layer, paper_development):
    """
    Age-to-age over the rectangle is the ratio of neighbouring cumulative
    factors, so chaining a row from the right rebuilds that row.
    """
    model = paper_layer(target_layer=(0, 500_000)).fit(paper_development)
    cdf = model.full_cdf_.set_backend("numpy").values[0, 0]
    ldf = model.full_ldf_.set_backend("numpy").values[0, 0]

    assert np.allclose(np.cumprod(ldf[:, ::-1], axis=1)[:, ::-1], cdf, rtol=1e-12)
    assert np.allclose(ldf[:, -1], 1.0)
    assert model.full_ldf_.shape == model.full_cdf_.shape


def test_full_ldf_diagonal_is_not_ldf(paper_layer, paper_development):
    """
    A row of full_ldf_ divides within itself, so it is one exposure period's
    own pattern at one cost level. ldf_ divides along the diagonal of
    full_cdf_, which steps up a row with every age -- so consecutive factors
    come from different exposure periods. The two are close but not equal, and
    conflating them would silently mix cost levels.
    """
    model = paper_layer(target_layer=(0, 500_000)).fit(paper_development)
    ldf = model.full_ldf_.set_backend("numpy").values[0, 0]
    origins = ldf.shape[0]
    diagonal = np.array([ldf[origins - 1 - j, j] for j in range(origins)])
    pattern = model.ldf_.set_backend("numpy").values[0, 0, 0]

    assert not np.allclose(diagonal, pattern)
    assert np.allclose(diagonal, pattern, atol=0.05)  # close, as expected


def test_full_cdf_keeps_ages_so_valuation_still_works(paper_layer, paper_development):
    """
    The rectangle keeps the Triangle's own development ages rather than pattern
    labels. Relabelling it "12-24" and so on would break valuation arithmetic
    on an object that is still an origin x development grid.
    """
    model = paper_layer(target_layer=(0, 500_000)).fit(paper_development)
    assert list(model.full_cdf_.development) == list(model.full_ldf_.development)
    assert model.full_cdf_.valuation is not None
