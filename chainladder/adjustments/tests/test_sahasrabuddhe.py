from __future__ import annotations

import chainladder as cl
import numpy as np
import pytest

from types import SimpleNamespace

from sklearn.base import clone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle
    from collections.abc import Callable


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


def _ones(tri: Triangle) -> Triangle:
    """
    A full triangle of ones. Used to generate full triangle trends.

    Parameters
    ----------
    tri: Triangle
        The Triangle with the desired shape.

    Returns
    -------
    Triangle
        A full triangle of ones over every origin and development cell.

    """
    ones = tri.copy().set_backend("numpy")
    ones.valuation_date = ones.valuation.max()
    return ones.fill(1)


def _array(tri: Triangle) -> np.ndarray:
    """
    The values of a Triangle as a numpy array, whichever backend it is on.

    Parameters
    ----------
    tri: Triangle
        The Triangle to read.

    Returns
    -------
    np.ndarray
        The values, shaped (index, columns, origin, development).

    """
    return np.asarray(tri.set_backend("numpy").values)


def _index(tri: Triangle, **kwargs: Any) -> Triangle:
    """
    A cost level index built from a Trend, stated relative to the latest origin
    period as the estimators take a Triangle.

    Parameters
    ----------
    tri: Triangle
        The Triangle the cost level index is built against.
    **kwargs: Any
        Passed to cl.Trend.

    Returns
    -------
    Triangle
        The cost level index, with the latest origin period's row equal to one.

    """
    index = 1 / cl.Trend(**kwargs).fit(_ones(tri)).trend_
    return index / index.iloc[..., -1:, :]


def _means_row(genins: Triangle, year: int, unmask: bool = True) -> Triangle:
    """
    THETA_10 as a one-origin Triangle labeled ``year``.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    year: int
        The origin year the row is labeled with.
    unmask: bool
        Whether to move the valuation date to the latest valuation, so the row is
        not masked.

    Returns
    -------
    Triangle
        One origin row of the claim size means.

    """
    base = [period.year for period in genins.origin].index(year)
    row = genins.iloc[0, 0, base : base + 1, :].copy()
    if unmask:
        row.valuation_date = row.valuation.max()
    row = row.fill(1)
    row.values = row.values * np.array(THETA_10, dtype="float64")
    return row


def _latest_origin_index(tri: Triangle, trend: float | int = 0.02) -> Triangle:
    """
    A cost level index for ``tri``, stated at the latest origin period.

    Parameters
    ----------
    tri: Triangle
        The Triangle the cost level index is built against.
    trend: float
        The annual origin trend, as a decimal.

    Returns
    -------
    Triangle
        The cost level index, with the latest origin period's row equal to one.

    """
    index = 1 / cl.Trend(trend, axis="origin").fit(_ones(tri)).trend_
    return index / index.iloc[..., -1:, :]


def _values(tri: Triangle, i: int, c: int) -> np.ndarray:
    """
    The (i, c) slice of a Triangle, whichever of the axes it spans.

    Parameters
    ----------
    tri: Triangle
        The Triangle to take a slice of.
    i: int
        Position along the index axis.
    c: int
        Position along the columns axis.

    Returns
    -------
    np.ndarray
        The slice with missing values set to zero.

    """
    values = np.nan_to_num(_array(tri))
    return values[i if values.shape[0] > 1 else 0, c if values.shape[1] > 1 else 0]


def _fit_without_and_with_constant_trend(
    genins: Triangle, pattern: Any
) -> tuple[cl.Sahasrabuddhe, cl.Sahasrabuddhe]:
    """
    Fit a pattern to the ground-up layer to $1,000,000, once with no trend and
    once with a constant (all ones) trend, which means the same thing.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set, which sets the ages the means are stated at.
    pattern: Any
        The development pattern to fit, a Triangle or a fitted estimator.

    Returns
    -------
    tuple[cl.Sahasrabuddhe, cl.Sahasrabuddhe]
        The fit with no trend, then the fit with a constant trend.

    """
    kwargs: dict[str, Any] = {
        "means": dict(zip(genins.development, THETA_10)),
        "basic_limit": 500_000,
        "target_layer": (0, 1_000_000),
    }
    without = cl.Sahasrabuddhe(**kwargs).fit(pattern)
    constant = cl.Sahasrabuddhe(trend=_ones(genins), **kwargs).fit(pattern)
    return without, constant


def _assert_same_pattern(
    first: cl.Sahasrabuddhe, second: cl.Sahasrabuddhe, atol: float
) -> None:
    """
    Assert that two fits give the same cdf_ and ldf_.

    Parameters
    ----------
    first: cl.Sahasrabuddhe
        The first fitted estimator.
    second: cl.Sahasrabuddhe
        The second fitted estimator.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    for name in ("cdf_", "ldf_"):
        assert np.allclose(
            _values(getattr(first, name), 0, 0),
            _values(getattr(second, name), 0, 0),
            atol=atol,
        ), name


@pytest.fixture
def paper_trend(genins: Triangle) -> Triangle:
    """
    Exhibit B3: the combined exposure- and calendar-period cost level index,
    restated relative to the latest origin period as the estimators take it.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    Triangle
        The index, a full origin by development triangle.

    """

    def index(**kwargs: Any) -> Triangle:
        """
        Reciprocal of a Trend's factors, fitted on a full triangle of ones.

        Parameters
        ----------
        **kwargs: Any
            Passed to cl.Trend.

        Returns
        -------
        Triangle
            The reciprocal of the Trend's factors, before they are rebased.

        """
        return 1 / cl.Trend(**kwargs).fit(_ones(genins)).trend_

    paper = index(
        trends=[0.02, 0.05, 0.02],
        dates=[
            ("2006-12-31", None),
            ("2007-12-31", "2006-12-31"),
            ("2010-12-31", "2007-12-31"),
        ],
        axis="origin",
    ) * index(
        trends=[0.01, -0.05, 0.01],
        dates=[
            ("2002-12-31", None),
            ("2003-12-31", "2002-12-31"),
            (None, "2003-12-31"),
        ],
        axis="valuation",
    )
    return paper / paper.iloc[..., -1:, :]


@pytest.fixture
def paper_model(genins: Triangle, paper_trend: Triangle) -> cl.Sahasrabuddhe:
    """
    Sahasrabuddhe fitted on genins as in the paper's Example 1, restated to the
    basic limit.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.

    Returns
    -------
    cl.Sahasrabuddhe
        The fitted estimator, exposing triangle_ and means_.

    """
    return cl.Sahasrabuddhe(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        observed_limit=1_000_000,
        basic_limit=500_000,
        target_layer=(0, 500_000),
    ).fit(genins)


@pytest.fixture
def paper_development(paper_model: cl.Sahasrabuddhe) -> cl.Development:
    """
    Volume-weighted development fitted on the restated triangle -- exhibits D2 and D3.

    Parameters
    ----------
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.

    Returns
    -------
    cl.Development
        The fitted development estimator, exposing ldf_ and cdf_.

    """
    return cl.Development(average="volume").fit(paper_model.triangle_)


@pytest.fixture
def paper_layer(
    paper_trend: Triangle, genins: Triangle
) -> Callable[..., cl.Sahasrabuddhe]:
    """
    Sahasrabuddhe configured for a layer, ready to fit a pattern.

    Parameters
    ----------
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    Callable[..., cl.Sahasrabuddhe]
        The factory.

    """

    def make(**kwargs: Any) -> cl.Sahasrabuddhe:
        """
        Build a Sahasrabuddhe for the paper's layer.

        Parameters
        ----------
        **kwargs: Any
            Passed to cl.Sahasrabuddhe, in addition to the paper's means, trend and
            basic limit.

        Returns
        -------
        cl.Sahasrabuddhe
            The unfitted estimator.

        """
        return cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=paper_trend,
            observed_limit=1_000_000,
            basic_limit=500_000,
            **kwargs,
        )

    return make


@pytest.fixture
def basic_layer(
    paper_layer: Callable[..., cl.Sahasrabuddhe], paper_development: cl.Development
) -> cl.Sahasrabuddhe:
    """
    The paper's basic limit pattern restated onto its own layer, ground up to
    the basic limit.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.

    Returns
    -------
    cl.Sahasrabuddhe
        The estimator fitted to the pattern, with target layer (0, 500,000).

    """
    return paper_layer(target_layer=(0, 500_000)).fit(paper_development)


@pytest.fixture
def lobs(clrd: Triangle) -> Triangle:
    """
    Six lines of business and two columns: a (6, 2, 10, 10) Triangle.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    Triangle
        The grouped Triangle.

    """
    # The aggregations are attached to TriangleGroupBy when pandas.py is loaded.
    # noinspection PyUnresolvedReferences
    return clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss"]]


def test_reproduces_published_exhibit_d1(paper_model: cl.Sahasrabuddhe) -> None:
    """
    Exhibit D1, the restated triangle.

    One cell is excluded: the paper's exhibit A1 prints 359,840 at exposure
    period 8 age 12, where genins -- and the Taylor-Ashe triangle it comes
    from -- has 359,480.

    Parameters
    ----------
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.

    Returns
    -------
    None

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

    adjusted = _array(paper_model.triangle_)[0, 0]
    assert np.abs(adjusted[comparable] - published[comparable]).max() <= 3.0


def test_reproduces_published_exhibits_d2_and_d3(paper_model: cl.Sahasrabuddhe) -> None:
    """
    Exhibits D2 and D3 follow from ordinary Development on the restated data.

    Parameters
    ----------
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.

    Returns
    -------
    None

    """
    dev = cl.Development(average="volume").fit(paper_model.triangle_)
    published_ldf = [3.511, 1.714, 1.399, 1.147, 1.076, 1.057, 1.039, 1.063, 1.013]
    published_cdf = [12.291, 3.501, 2.042, 1.460, 1.273, 1.183, 1.119, 1.077, 1.013]
    assert np.abs(dev.ldf_.values[0, 0, 0] - published_ldf).max() <= 0.0005
    assert np.abs(dev.cdf_.values[0, 0, 0] - published_cdf).max() <= 0.002


def test_pipeline_matches_a_manual_fit(
    genins: Triangle,
    paper_trend: Triangle,
    paper_model: cl.Sahasrabuddhe,
    atol: float,
) -> None:
    """
    The transformer hands adjusted values downstream, not the original.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    pipe = cl.Pipeline([
        (
            "basis",
            cl.Sahasrabuddhe(
                means=dict(zip(genins.development, THETA_10)),
                trend=paper_trend,
                observed_limit=1_000_000,
                basic_limit=500_000,
                target_layer=(0, 500_000),
            ),
        ),
        ("dev", cl.Development(average="volume")),
    ]).fit(genins)
    manual = cl.Development(average="volume").fit(paper_model.triangle_)
    assert np.allclose(pipe.named_steps.dev.ldf_.values, manual.ldf_.values, atol=atol)
    raw = cl.Development(average="volume").fit(genins)
    assert not np.allclose(pipe.named_steps.dev.ldf_.values, raw.ldf_.values)


def test_preserves_the_index_axis(clrd: Triangle) -> None:
    """
    A Triangle with many index entries has to
    come back with all of them.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    tri = clrd[["CumPaidLoss"]]
    trend = _index(
        tri=tri,
        trends=0.04,
        axis="origin",
    )
    means = dict(zip(tri.development, np.linspace(30000, 300000, 10)))
    model = cl.Sahasrabuddhe(
        means=means,
        trend=trend,
        observed_limit=1_000_000,
        basic_limit=500_000,
        target_layer=(0, 500_000),
    ).fit(tri)
    assert model.triangle_.shape == tri.shape
    assert np.array_equal(
        np.isnan(_array(model.triangle_)),
        np.isnan(_array(tri)),
    )


def test_triangle_trend_is_taken_to_be_at_the_latest_origin(
    genins: Triangle,
    paper_trend: Triangle,
    atol: float,
) -> None:
    """
    A trend supplied as a fitted Triangle should assume the cost level is at the latest origin.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    theta = np.array(THETA_10, dtype="float64")
    means = dict(zip(genins.development, THETA_10))
    lev = cl.LEV(
        means=means,
        trend=paper_trend,
    ).fit(genins)
    fitted = _array(lev.means_)
    index = _array(paper_trend)
    assert np.allclose(index[..., -1, :], 1.0, atol=atol)
    assert np.allclose(fitted[..., -1, :], theta, atol=atol)
    assert np.allclose(fitted, theta * index, atol=atol)

    doubled = cl.LEV(
        means=means,
        trend=paper_trend * 2,
    ).fit(genins)
    assert np.allclose(_array(doubled.means_)[..., -1, :], 2 * theta, atol=atol)


def test_rejects_a_trend_from_a_different_triangle(genins: Triangle) -> None:
    """
    Matching shapes are not enough -- the periods have to match too.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    other = cl.load_sample("clrd")[["CumPaidLoss"]]
    trend = _index(tri=other, trends=0.04, axis="origin")
    assert trend.shape[-2:] == genins.shape[-2:]  # would pass a shape-only check
    with pytest.raises(ValueError, match="does not share X's origin axis"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=trend,
            observed_limit=1_000_000,
            basic_limit=500_000,
            target_layer=(0, 500_000),
        ).fit(genins)


def test_rejects_a_trend_with_gaps_in_the_latest_period(genins: Triangle) -> None:
    """
    A trend cannot have NaNs in the latest period.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    trend = 1 / cl.Trend(0.05, axis="origin").fit(genins).trend_
    with pytest.raises(ValueError, match="full triangle of ones"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=trend,
            observed_limit=1_000_000,
            basic_limit=500_000,
            target_layer=(0, 500_000),
        ).fit(genins)


@pytest.mark.parametrize("missing", ["observed_limit", "target_layer"])
def test_required_arguments_raise(
    genins: Triangle,
    paper_trend: Triangle,
    missing: str,
) -> None:
    """
    Missing required arguments should raise an error.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    missing: str
        The name of the argument that is left out.

    Returns
    -------
    None

    """
    kwargs: dict[str, Any] = dict(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        observed_limit=1_000_000,
        basic_limit=500_000,
        target_layer=(0, 500_000),
    )
    kwargs[missing] = None
    with pytest.raises(ValueError, match=missing):
        cl.Sahasrabuddhe(**kwargs).fit(genins)


def test_params_survive_sklearn_clone() -> None:
    """
    A clone of the estimator has the same parameters.

    Returns
    -------
    None

    """
    estimator = cl.Sahasrabuddhe(
        means={12: 1.0},
        observed_limit=1e6,
        basic_limit=5e5,
        target_layer=(0, 5e5),
    )
    cloned = clone(estimator)
    assert isinstance(cloned, cl.Sahasrabuddhe)
    assert cloned.get_params() == estimator.get_params()


def test_lev_at_zero_is_zero(genins: Triangle, atol: float) -> None:
    """
    The limited expected value at a limit of zero is zero, so a layer attaching
    at zero keeps its whole limited expected value.

    A layer's expected loss is ``LEV(limit) - LEV(attachment)``, and a ground-up
    layer subtracts ``LEV(0)``. ``0 / means_`` is NaN in Triangle arithmetic and
    ``1 - NaN`` is then 1, so a naive closed form would hand back the means
    untouched at a limit of zero. Every layer attaching at 0, which is the common
    case, would then lose a whole mean.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    model = cl.LEV(
        means=means,
        limit=500_000,
    ).fit(genins)

    # noinspection PyProtectedMember
    at_zero = cl.LEV._limited_expected_value(means=model.means_, limit=0)
    assert np.array_equal(
        _array(at_zero),
        np.zeros(model.means_.shape),
    )

    # The ground-up layer is the closed form itself, with nothing subtracted.
    theta = np.array(THETA_10, dtype="float64")
    closed_form = theta * (1 - np.exp(-500_000 / theta))
    assert np.allclose(_array(model.lev_)[0, 0, 0], closed_form, atol=atol)


def test_rejects_a_limit_that_does_not_exceed_the_attachment(
    genins: Triangle,
) -> None:
    """
    A layer needs a limit above its attachment point.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    with pytest.raises(ValueError, match="must exceed attachment"):
        cl.LEV(means=means, limit=0).fit(genins)


@pytest.mark.parametrize("year", [2005, 2010])
def test_one_origin_triangle_matches_a_mapping(
    genins: Triangle, paper_trend: Triangle, year: int, atol: float
) -> None:
    """
    LEV means parameter should be able to take a single-row Triangle of means or an equivalent dictionary mapping.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    year: int
        The origin year the one-origin Triangle is labeled with.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    from_triangle = cl.LEV(
        means=_means_row(genins, year),
        trend=paper_trend,
    ).fit(genins)
    from_mapping = cl.LEV(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
    ).fit(genins)
    assert np.allclose(
        _array(from_triangle.means_),
        _array(from_mapping.means_),
        atol=atol,
    )


@pytest.mark.parametrize("with_trend", [True, False])
def test_rejects_a_masked_one_origin_triangle(
    genins: Triangle, paper_trend: Triangle, with_trend: bool
) -> None:
    """
    A row sliced straight from the Triangle is masked past its first age, which
    would quietly turn every later age to NaN.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    with_trend: bool
        Whether the trend is supplied.

    Returns
    -------
    None

    """
    trend = paper_trend if with_trend else None
    with pytest.raises(ValueError, match="gaps in its single origin row"):
        cl.LEV(
            means=_means_row(genins=genins, year=2010, unmask=False), trend=trend
        ).fit(genins)


def test_triangle_keeps_the_triangles_valuation_date(
    genins: Triangle, paper_model: cl.Sahasrabuddhe
) -> None:
    """
    The limited expected values span the whole triangle, so the product that
    builds `triangle_` inherits their valuation date unless it is put back.
    Leaving it would give the result a nan_triangle and a latest diagonal
    belonging to a Triangle running years past the data.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.

    Returns
    -------
    None

    """
    assert paper_model.triangle_.valuation_date == genins.valuation_date
    adjusted = paper_model.triangle_.set_backend("numpy")
    assert np.array_equal(
        np.isnan(np.asarray(adjusted.nan_triangle, dtype="float64")),
        np.isnan(np.asarray(genins.set_backend("numpy").nan_triangle, dtype="float64")),
    )
    assert (
        np.isfinite(_array(paper_model.triangle_.latest_diagonal)).sum()
        == genins.shape[-2]
    )


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
    paper_layer: Callable[..., cl.Sahasrabuddhe],
    paper_development: cl.Development,
    name: str,
    layer: dict[str, Any],
    published: list[float],
) -> None:
    """
    Exhibits E1 to E4 -- exposure period 1's row of each. The paper prints
    "very large" where a layer has essentially no expected loss at that age,
    and those cells are non-finite here for the same reason, so only the cells
    it prints a number for are compared.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.
    name: str
        The exhibit's name.
    layer: dict[str, Any]
        The keyword arguments that select the exhibit's target layer.
    published: list[float]
        The paper's printed factors for exposure period 1.

    Returns
    -------
    None

    """
    surface = paper_layer(**layer).fit(paper_development).full_cdf_
    computed = _array(surface)[0, 0, 0]
    published = np.array(published, dtype="float64")
    comparable = np.isfinite(published)
    deviation = np.abs(computed[comparable] / published[comparable] - 1).max()
    assert deviation <= 0.01, f"{name} deviates from the paper by {deviation:.5f}"


def test_target_equal_to_basic_returns_the_pattern_itself(
    paper_development: cl.Development, basic_layer: cl.Sahasrabuddhe, atol: float
) -> None:
    """
    Restating onto the layer the pattern already sits on is the identity.

    Parameters
    ----------
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    base_row = _array(basic_layer.full_cdf_)[0, 0, -1]
    given = _array(paper_development.cdf_)[0, 0, 0]
    assert np.allclose(base_row[: len(given)], given, atol=atol)
    assert np.isclose(base_row[-1], 1.0, atol=atol)


def test_full_cdf_fills_the_whole_triangle(
    basic_layer: cl.Sahasrabuddhe, atol: float
) -> None:
    """
    full_cdf_ has a value for every cell and reaches ultimate at the last age.

    Parameters
    ----------
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    surface = _array(basic_layer.full_cdf_)
    assert np.isnan(surface).sum() == 0
    # Every row reaches ultimate by the last age, whatever its cost level.
    assert np.allclose(surface[0, 0, :, -1], 1.0, atol=atol)


def test_accepts_an_estimator_or_a_bare_pattern(
    paper_layer: Callable[..., cl.Sahasrabuddhe],
    paper_development: cl.Development,
    basic_layer: cl.Sahasrabuddhe,
    atol: float,
) -> None:
    """
    ``fit()`` should take a fitted development estimator, or the cdf_ Triangle it carries.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    reference = _array(basic_layer.full_cdf_)
    from_pattern = _array(
        paper_layer(target_layer=(0, 500_000)).fit(paper_development.cdf_).full_cdf_
    )
    assert np.allclose(reference, from_pattern, atol=atol)


def test_a_transformed_triangle_is_claims_not_a_pattern(
    paper_layer: Callable[..., cl.Sahasrabuddhe],
    paper_model: cl.Sahasrabuddhe,
) -> None:
    """
    A Triangle that has been through a development step carries a cdf_ of its
    own. It is still claims data, and fitting on that cdf_ would restate it
    as though it were a pattern.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.

    Returns
    -------
    None

    """
    transformed = cl.Development(average="volume").fit_transform(paper_model.triangle_)
    assert hasattr(transformed, "cdf_") and not transformed.is_pattern

    model = paper_layer(target_layer=(0, 500_000)).fit(transformed)
    assert hasattr(model, "triangle_")
    assert not hasattr(model, "full_cdf_")


def test_aligns_the_pattern_by_age_not_position(
    paper_layer: Callable[..., cl.Sahasrabuddhe],
    paper_model: cl.Sahasrabuddhe,
    atol: float | int,
) -> None:
    """
    A fitted tail makes cdf_ longer than the development axis, so zipping the
    two by position would silently misalign every factor.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    transformed = cl.Development(average="volume").fit_transform(paper_model.triangle_)
    tailed = cl.TailConstant(1.05).fit(transformed)
    assert tailed.cdf_.shape[-1] > paper_model.triangle_.shape[-1]

    with_tail = _array(paper_layer(target_layer=(0, 500_000)).fit(tailed).full_cdf_)
    without = _array(
        paper_layer(target_layer=(0, 500_000))
        .fit(cl.Development(average="volume").fit(paper_model.triangle_))
        .full_cdf_
    )
    # The tail scales every factor by 1.05; ages beyond the triangle are dropped.
    assert np.allclose(with_tail / without, 1.05, atol=atol)


def test_cdf_is_the_latest_diagonal_of_full_cdf(
    basic_layer: cl.Sahasrabuddhe, atol: float | int
) -> None:
    """
    The pattern is the diagonal of the full triangle -- one cell per row, each
    exposure period at its own current age and cost level. Reading a single row
    instead would apply one period's cost level to all of them.

    Parameters
    ----------
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    pattern_array = _array(basic_layer.full_cdf_)[0, 0]
    origins = pattern_array.shape[0]
    diagonal = np.array([pattern_array[origins - 1 - j, j] for j in range(origins)])
    assert np.allclose(
        _array(basic_layer.cdf_)[0, 0, 0][:origins],
        diagonal,
        atol=atol,
    )


def test_cdf_and_ldf_labels(basic_layer: cl.Sahasrabuddhe) -> None:
    """
    cdf_ is labeled to ultimate and ldf_ by age-to-age interval.

    Parameters
    ----------
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.

    Returns
    -------
    None

    """
    assert list(basic_layer.cdf_.development)[:3] == ["12-Ult", "24-Ult", "36-Ult"]
    assert list(basic_layer.ldf_.development)[:3] == ["12-24", "24-36", "36-48"]


def test_ldf_and_cdf_are_consistent(
    basic_layer: cl.Sahasrabuddhe, atol: float | int
) -> None:
    """
    Chaining the age-to-age factors from the right rebuilds the cumulative ones.

    Parameters
    ----------
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    ldf = _array(basic_layer.ldf_)[0, 0, 0]
    rebuilt = np.cumprod(ldf[::-1])[::-1]
    assert np.allclose(rebuilt, _array(basic_layer.cdf_)[0, 0, 0], atol=atol)


def test_a_taller_layer_has_larger_cumulative_factors(
    paper_layer: Callable[..., cl.Sahasrabuddhe],
    paper_development: cl.Development,
    basic_layer: cl.Sahasrabuddhe,
) -> None:
    """
    A taller layer truncates less of each claim, so more of its ultimate loss
    emerges late and its cumulative factors to ultimate are larger at every age
    but the last, where both are one.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.

    Returns
    -------
    None

    """
    basic = _array(basic_layer.cdf_)[0, 0, 0]
    taller = _array(paper_layer(target_layer=(0, np.inf)).fit(paper_development).cdf_)[
        0, 0, 0
    ]
    assert (taller[:-1] > basic[:-1]).all()


def test_degenerate_layer_raises(
    paper_layer: Callable[..., cl.Sahasrabuddhe], paper_development: cl.Development
) -> None:
    """
    A layer that exhausts at its attachment point is refused.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="limit must be the larger"):
        paper_layer(target_layer=(500_000, 500_000)).fit(paper_development)


def test_triangle_is_restated_to_the_target_layer(
    genins: Triangle,
    paper_trend: Triangle,
    atol: float | int,
) -> None:
    """
    Check equation 3.2 for any layer.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    attachment, limit = 750_000, 1_250_000
    model = cl.Sahasrabuddhe(
        means=means,
        trend=paper_trend,
        observed_limit=1_000_000,
        target_layer=(attachment, limit),
    ).fit(genins)

    def expected(**kwargs: Any) -> np.ndarray:
        """
        Expected loss in a layer, at every cell.

        Parameters
        ----------
        **kwargs: Any
            Passed to cl.LEV.

        Returns
        -------
        np.ndarray
            The layer's expected loss, shaped (origin, development).

        """
        lev = (
            cl
            .LEV(
                means=means,
                trend=paper_trend,
                **kwargs,
            )
            .fit(genins)
            .lev_
        )
        return _array(lev)[0, 0]

    layer = expected(
        attachment=attachment,
        limit=limit,
    )[-1]
    observed = expected(limit=1_000_000)
    by_hand = _array(genins)[0, 0] * layer / observed
    assert np.allclose(
        np.nan_to_num(_array(model.triangle_)[0, 0]),
        np.nan_to_num(by_hand),
        atol=atol,
    )
    assert model.triangle_.valuation_date == genins.valuation_date


def test_triangle_in_the_ground_up_basic_layer_is_the_paper_restatement(
    genins: Triangle,
    paper_trend: Triangle,
    paper_model: cl.Sahasrabuddhe,
    atol: float | int,
) -> None:
    """
    (0, B) is equation 3.2 as published, with or without a basic_limit.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    # No basic limit is supplied, but the limit is the same as the basic limit.
    without = cl.Sahasrabuddhe(
        means=dict(zip(genins.development, THETA_10)),
        trend=paper_trend,
        observed_limit=1_000_000,
        target_layer=(0, 500_000),
    ).fit(genins)
    assert np.allclose(
        np.nan_to_num(_array(without.triangle_)),
        np.nan_to_num(_array(paper_model.triangle_)),
        atol=atol,
    )


def test_transform_restates_to_the_target_layer(
    paper_layer: Callable[..., cl.Sahasrabuddhe],
    genins: Triangle,
    atol: float | int,
) -> None:
    """
    transform restates a Triangle to the target layer, as triangle_ does.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.
    genins: Triangle
        The genins sample data set.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    model = paper_layer(target_layer=(750_000, 1_250_000)).fit(genins)
    assert np.allclose(
        np.nan_to_num(_array(model.transform(genins))),
        np.nan_to_num(_array(model.triangle_)),
        atol=atol,
    )


def test_pattern_requires_a_basic_limit(
    genins: Triangle, paper_trend: Triangle, paper_development: cl.Development
) -> None:
    """
    A pattern cannot be restated without the basic limit it was fitted at.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="basic_limit is required"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=paper_trend,
            target_layer=(0, 500_000),
        ).fit(paper_development)


def test_incomplete_diagonal_warns_but_keeps_the_triangle(qtr: Triangle) -> None:
    """
    Quarterly development on annual origins means the latest diagonal touches
    only every fourth age, so no usable pattern comes out. The full triangle is
    defined for every cell regardless, so it is kept and a warning is raised.

    Parameters
    ----------
    qtr: Triangle
        The quarterly sample data set.

    Returns
    -------
    None

    """
    tri = qtr[["paid"]]
    trend = _index(tri, trends=0.03, axis="origin")
    means = dict(zip(tri.development, np.linspace(20000, 200000, tri.shape[-1])))
    kwargs: dict[str, Any] = dict(
        means=means,
        trend=trend,
        observed_limit=1_000_000,
        basic_limit=500_000,
    )
    restated = cl.Sahasrabuddhe(**kwargs, target_layer=(0, 500_000)).fit(tri).triangle_
    development = cl.Development(average="volume").fit(restated)

    with pytest.warns(UserWarning, match="has reached development age"):
        model = cl.Sahasrabuddhe(**kwargs, target_layer=(0, 500_000)).fit(development)
    assert not hasattr(model, "cdf_")
    assert model.full_cdf_.shape == trend.shape


def test_full_ldf_rebuilds_full_cdf(
    basic_layer: cl.Sahasrabuddhe,
    atol: float | int,
) -> None:
    """
    Age-to-age over the triangle is the ratio of neighboring cumulative
    factors, so chaining a row from the right rebuilds that row.

    Parameters
    ----------
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    cdf = _array(basic_layer.full_cdf_)[0, 0]
    ldf = _array(basic_layer.full_ldf_)[0, 0]

    assert np.allclose(np.cumprod(ldf[:, ::-1], axis=1)[:, ::-1], cdf, atol=atol)
    assert np.allclose(ldf[:, -1], 1.0, atol=atol)
    assert basic_layer.full_ldf_.shape == basic_layer.full_cdf_.shape


def test_full_ldf_diagonal_is_not_ldf(basic_layer: cl.Sahasrabuddhe) -> None:
    """
    A row of full_ldf_ divides within itself, so it is one exposure period's
    own pattern at one cost level. ldf_ divides along the diagonal of
    full_cdf_, which steps up a row with every age -- so consecutive factors
    come from different exposure periods. The two are close but not equal, and
    conflating them would silently mix cost levels.

    Parameters
    ----------
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.

    Returns
    -------
    None

    """
    ldf = _array(basic_layer.full_ldf_)[0, 0]
    origins = ldf.shape[0]
    diagonal = np.array([ldf[origins - 1 - j, j] for j in range(origins)])
    pattern = _array(basic_layer.ldf_)[0, 0, 0]

    assert not np.allclose(diagonal, pattern)
    assert np.allclose(
        diagonal, pattern, atol=0.05
    )  # close, as expected  # close, as expected


def test_full_cdf_keeps_ages_so_valuation_still_works(
    basic_layer: cl.Sahasrabuddhe,
) -> None:
    """
    The triangle keeps the Triangle's own development ages rather than pattern
    labels. Relabeling it "12-24" and so on would break valuation arithmetic
    on an object that is still an origin x development grid.

    Parameters
    ----------
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.

    Returns
    -------
    None

    """
    assert list(basic_layer.full_cdf_.development) == list(
        basic_layer.full_ldf_.development
    )
    assert basic_layer.full_cdf_.valuation is not None


def test_means_are_required(genins: Triangle) -> None:
    """
    LEV cannot be fitted without means.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="means is required"):
        cl.LEV().fit(genins)


def test_rejects_a_mapping_missing_an_age(genins: Triangle) -> None:
    """
    A mapping of means has to cover every development age.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    del means[60]
    with pytest.raises(ValueError, match=r"missing development age\(s\) \[60\]"):
        cl.LEV(means=means).fit(genins)


def test_rejects_a_negative_limit(genins: Triangle, paper_trend: Triangle) -> None:
    """
    The observed limit has to be non-negative.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="limit must be non-negative"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=paper_trend,
            observed_limit=-1,
            basic_limit=500_000,
            target_layer=(0, 500_000),
        ).fit(genins)


def test_lev_transform_carries_the_fitted_state(genins: Triangle) -> None:
    """
    LEV.transform leaves the data alone and attaches means_ and lev_.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    lev = cl.LEV(
        means=dict(zip(genins.development, THETA_10)),
        limit=500_000,
    ).fit(genins)
    transformed = lev.transform(genins)
    assert getattr(transformed, "means_") is lev.means_
    assert getattr(transformed, "lev_") is lev.lev_
    assert np.array_equal(
        np.nan_to_num(_array(transformed)),
        np.nan_to_num(_array(genins)),
    )


@pytest.mark.parametrize("target_layer", [500_000, ("a", "b"), (0, 1, 2)])
def test_rejects_a_malformed_target_layer(
    genins: Triangle,
    paper_trend: Triangle,
    target_layer: Any,
) -> None:
    """
    target_layer has to be a pair of limits.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_trend: Triangle
        Exhibit B3, the paper's cost level index, stated at the latest origin period.
    target_layer: Any
        A malformed target layer.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="target_layer must be a pair of limits"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=paper_trend,
            observed_limit=1_000_000,
            basic_limit=500_000,
            target_layer=target_layer,
        ).fit(genins)


def test_transform_after_a_pattern_fit_carries_the_pattern(
    genins: Triangle, basic_layer: cl.Sahasrabuddhe, atol: float | int
) -> None:
    """
    Fitted to a pattern there is nothing to restate, so transform passes X
    through and hands the restated pattern downstream instead.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    basic_layer: cl.Sahasrabuddhe
        The paper's basic limit pattern restated onto its own layer, ground up
        to the basic limit.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    transformed = basic_layer.transform(genins)
    for item in ("means_", "ldf_", "full_cdf_", "full_ldf_"):
        assert getattr(transformed, item) is getattr(basic_layer, item)
    assert np.allclose(
        _array(transformed.cdf_),
        _array(basic_layer.cdf_),
        atol=atol,
    )
    assert np.array_equal(
        np.nan_to_num(_array(transformed)),
        np.nan_to_num(_array(genins)),
    )


def test_rejects_something_that_is_not_a_pattern(
    paper_layer: Callable[..., cl.Sahasrabuddhe],
) -> None:
    """
    An object whose cdf_ is not a Triangle of factors fails on use.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.

    Returns
    -------
    None

    """
    impostor = SimpleNamespace(cdf_=np.ones(10))
    with pytest.raises(ValueError, match="must be a fitted development estimator"):
        # The wrong type is the point of the test.
        # noinspection PyTypeChecker
        paper_layer(target_layer=(0, 500_000)).fit(impostor)


def test_lev_takes_an_unfitted_trend(
    genins: Triangle,
    atol: float | int,
) -> None:
    """
    An unfitted Trend is fitted on X and inverted into a cost level index.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    index = _index(genins, trends=0.04, axis="origin")
    from_triangle = cl.LEV(means=means, trend=index).fit(genins)
    from_trend = cl.LEV(means=means, trend=cl.Trend(0.04, axis="origin")).fit(genins)
    assert np.allclose(
        _array(from_trend.means_),
        _array(from_triangle.means_),
        atol=atol,
    )
    assert from_trend.trend_ is not None
    assert np.allclose(
        _array(from_trend.trend_),
        _array(index),
        atol=atol,
    )


def test_trend_object_is_left_unfitted(genins: Triangle) -> None:
    """
    fit works on a copy: the caller's Trend is neither fitted nor changed.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    trend = cl.Trend(0.04, axis="origin")
    cl.LEV(
        means=dict(zip(genins.development, THETA_10)),
        trend=trend,
    ).fit(genins)
    assert not hasattr(trend, "trend_")
    assert trend.get_params() == cl.Trend(0.04, axis="origin").get_params()


def test_trend_is_none_without_a_trend(genins: Triangle) -> None:
    """
    LEV records no trend_ when none was supplied.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.

    Returns
    -------
    None

    """
    lev = cl.LEV(means=dict(zip(genins.development, THETA_10))).fit(genins)
    assert lev.trend_ is None


def test_sahasrabuddhe_takes_an_unfitted_trend(
    genins: Triangle,
    atol: float | int,
) -> None:
    """
    An unfitted Trend restates the data as its index does when passed as a Triangle.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    kwargs: dict[str, Any] = dict(
        means=dict(zip(genins.development, THETA_10)),
        observed_limit=1_000_000,
        basic_limit=500_000,
        target_layer=(0, 500_000),
    )
    index = _index(genins, trends=0.04, axis="origin")
    from_triangle = cl.Sahasrabuddhe(trend=index, **kwargs).fit(genins)
    from_trend = cl.Sahasrabuddhe(trend=cl.Trend(0.04, axis="origin"), **kwargs).fit(
        genins
    )
    assert np.allclose(
        np.nan_to_num(_array(from_trend.triangle_)),
        np.nan_to_num(_array(from_triangle.triangle_)),
        atol=atol,
    )
    # transform re-uses the fitted claim size model, with no trend to refit.
    assert np.allclose(
        np.nan_to_num(_array(from_trend.transform(genins))),
        np.nan_to_num(_array(from_triangle.transform(genins))),
        atol=atol,
    )


def test_pattern_rejects_an_unfitted_trend(
    genins: Triangle,
    paper_development: cl.Development,
) -> None:
    """
    A pattern has one origin row, so there is no Triangle to fit a Trend on.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="no Triangle to fit trend on"):
        cl.Sahasrabuddhe(
            means=dict(zip(genins.development, THETA_10)),
            trend=cl.Trend(0.04, axis="origin"),
            basic_limit=500_000,
            target_layer=(0, 250_000),
        ).fit(paper_development)


def test_params_survive_sklearn_clone_with_a_trend_object() -> None:
    """
    Cloning copies a Trend parameter rather than sharing it.

    Returns
    -------
    None

    """
    original = cl.Trend(0.04, axis="origin")
    estimator = cl.Sahasrabuddhe(
        means={12: 1.0},
        trend=original,
        observed_limit=1e6,
        basic_limit=5e5,
        target_layer=(0, 5e5),
    )
    cloned = clone(estimator)
    assert isinstance(cloned, cl.Sahasrabuddhe)
    cloned_trend = cloned.get_params()["trend"]
    assert isinstance(cloned_trend, cl.Trend)
    assert cloned_trend is not original
    assert cloned_trend.get_params() == original.get_params()


def test_unfitted_trend_is_rebased_to_the_latest_origin(
    genins: Triangle,
    atol: float | int,
) -> None:
    """
    A valuation trend is not constant across the latest origin's row, so the
    index a Trend object yields is divided through by that row.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    trend = cl.Trend(0.01, axis="valuation")
    index = 1 / trend.fit(_ones(genins)).trend_
    assert not np.allclose(_array(index)[..., -1, :], 1.0)
    from_trend = cl.LEV(means=means, trend=trend).fit(genins)
    from_triangle = cl.LEV(means=means, trend=index / index.iloc[..., -1:, :]).fit(
        genins
    )
    assert np.allclose(
        _array(from_trend.means_),
        _array(from_triangle.means_),
        atol=atol,
    )
    assert np.allclose(
        _array(from_trend.means_)[..., -1, :],
        np.array(THETA_10, dtype="float64"),
        atol=atol,
    )


def test_multidimensional_triangle_matches_slice_by_slice_fits(
    lobs: Triangle, atol: float | int
) -> None:
    """
    Fitting a multidimensional Triangle equals fitting each index and column on its own.

    Parameters
    ----------
    lobs: Triangle
        clrd summed by line of business, with two columns: a (6, 2, 10, 10) Triangle.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = {age: 20_000 * (1 + i) for i, age in enumerate(lobs.development)}
    kwargs: dict[str, Any] = {"observed_limit": 1_000_000, "basic_limit": 500_000}
    model = cl.Sahasrabuddhe(
        means=means,
        trend=_latest_origin_index(lobs),
        target_layer=(0, 500_000),
        **kwargs,
    ).fit(lobs)
    assert model.triangle_.shape == lobs.shape
    for i in range(lobs.shape[0]):
        for c in range(lobs.shape[1]):
            single = lobs.iloc[i, c]
            expected = cl.Sahasrabuddhe(
                means=means,
                trend=_latest_origin_index(single),
                target_layer=(0, 500_000),
                **kwargs,
            ).fit(single)
            assert np.allclose(
                _values(model.triangle_, i, c),
                _values(expected.triangle_, 0, 0),
                atol=atol,
            )


@pytest.mark.parametrize("trend_slice", [False, True])
@pytest.mark.parametrize("pattern_slice", [False, True])
def test_multidimensional_pattern_matches_slice_by_slice_fits(
    lobs: Triangle,
    trend_slice: bool,
    pattern_slice: bool,
    atol: float | int,
) -> None:
    """
    Every index and column gets its own pattern, whether the pattern, the
    trend, or both span them.

    Parameters
    ----------
    lobs: Triangle
        clrd summed by line of business, with two columns: a (6, 2, 10, 10) Triangle.
    trend_slice: bool
        Whether the trend is a single (0, 0) slice rather than spanning every index
        and column.
    pattern_slice: bool
        Whether the pattern is a single (0, 0) slice rather than spanning every
        index and column.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = {age: 20_000 * (1 + i) for i, age in enumerate(lobs.development)}
    kwargs: dict[str, Any] = {
        "basic_limit": 500_000,
        "target_layer": (1_000_000, 1_500_000),
    }
    cdf = cl.Development().fit(lobs).cdf_
    trend = _latest_origin_index(lobs)
    fitted_on = cdf.iloc[0, 0] if pattern_slice else cdf
    trend_used = trend.iloc[0, 0] if trend_slice else trend
    model = cl.Sahasrabuddhe(means=means, trend=trend_used, **kwargs).fit(fitted_on)

    # The result spans whichever of the inputs span the index and columns,
    # and is labeled by them.
    spans = not (trend_slice and pattern_slice)
    expected_shape = lobs.shape[:2] if spans else (1, 1)
    assert model.cdf_.shape[:2] == expected_shape
    assert model.ldf_.shape[:2] == expected_shape
    assert model.full_cdf_.shape[:2] == expected_shape
    assert len(model.full_cdf_.index) == expected_shape[0]
    assert len(model.cdf_.index) == expected_shape[0]
    if spans:
        assert list(model.full_cdf_.columns) == list(lobs.columns)
        assert list(model.cdf_.columns) == list(lobs.columns)

    for i in range(expected_shape[0]):
        for c in range(expected_shape[1]):
            single = cl.Sahasrabuddhe(
                means=means,
                trend=trend_used.iloc[i, c] if trend_used.shape[0] > 1 else trend_used,
                **kwargs,
            ).fit(fitted_on.iloc[i, c] if fitted_on.shape[0] > 1 else fitted_on)
            for name in ("cdf_", "ldf_", "full_cdf_", "full_ldf_"):
                assert np.allclose(
                    _values(getattr(model, name), i, c),
                    _values(getattr(single, name), 0, 0),
                    atol=atol,
                ), name


def test_multidimensional_means(lobs: Triangle, atol: float | int) -> None:
    """
    A Triangle of means with its own values for every index and column.

    Parameters
    ----------
    lobs: Triangle
        clrd summed by line of business, with two columns: a (6, 2, 10, 10) Triangle.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = _ones(lobs).iloc[..., -1:, :]
    scale = np.arange(1, 13).reshape(6, 2, 1, 1) * np.arange(1, 11).reshape(1, 1, 1, 10)
    means.values = means.values * (10_000.0 * scale)
    trend = _latest_origin_index(lobs)
    kwargs: dict[str, Any] = {
        "basic_limit": 500_000,
        "target_layer": (1_000_000, 1_500_000),
    }
    cdf = cl.Development().fit(lobs).cdf_
    model = cl.Sahasrabuddhe(means=means, trend=trend, **kwargs).fit(cdf)
    for i in range(6):
        for c in range(2):
            single = cl.Sahasrabuddhe(
                means=means.iloc[i, c], trend=trend.iloc[i, c], **kwargs
            ).fit(cdf.iloc[i, c])
            for name in ("cdf_", "ldf_", "full_cdf_", "full_ldf_"):
                assert np.allclose(
                    _values(getattr(model, name), i, c),
                    _values(getattr(single, name), 0, 0),
                    atol=atol,
                ), name


def test_pattern_without_a_trend_is_a_constant_cost_level(
    genins: Triangle, paper_development: cl.Development, atol: float | int
) -> None:
    """
    Left out, every origin period sits at the same cost level: one origin row,
    which is every row of the same fit with a constant (all ones) trend.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    bare, constant = _fit_without_and_with_constant_trend(genins, paper_development)
    _assert_same_pattern(bare, constant, atol)
    assert bare.full_cdf_.shape[-2] == 1
    assert bare.full_ldf_.shape[-2] == 1
    for row in range(constant.full_cdf_.shape[-2]):
        assert np.allclose(
            _values(bare.full_cdf_, 0, 0)[0],
            _values(constant.full_cdf_, 0, 0)[row],
            atol=atol,
        )


def test_pattern_without_a_trend_needs_means_without_gaps(
    genins: Triangle,
    paper_development: cl.Development,
) -> None:
    """
    Without a trend, means has to be stated at every age of the pattern up to
    its own last age. A gap would silently drop that age from the rectangle.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    del means[60]
    with pytest.raises(ValueError, match=r"missing development age\(s\) \[60\]"):
        cl.Sahasrabuddhe(
            means=means,
            basic_limit=500_000,
            target_layer=(0, 1_000_000),
        ).fit(paper_development)


def test_pattern_without_a_trend_ignores_tail_ages_beyond_the_means(
    genins: Triangle,
    paper_model: cl.Sahasrabuddhe,
    atol: float,
) -> None:
    """
    A fitted tail adds ages beyond the triangle's own, such as 132 or 9999.
    Means are not stated there, and the pattern is restated on the ages they
    are stated at, as it is with a trend.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_model: cl.Sahasrabuddhe
        Sahasrabuddhe fitted on genins as in the paper's Example 1.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    transformed = cl.Development(average="volume").fit_transform(paper_model.triangle_)
    tailed = cl.TailConstant(1.05).fit(transformed)
    assert np.asarray(tailed.cdf_.ddims).max() > np.asarray(genins.development).max()

    bare, constant = _fit_without_and_with_constant_trend(genins, tailed)
    _assert_same_pattern(bare, constant, atol)


def test_pattern_without_a_trend_takes_means_as_a_triangle(
    genins: Triangle, paper_development: cl.Development, atol: float
) -> None:
    """
    Means as a one-origin Triangle give the same pattern as a mapping.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    paper_development: cl.Development
        Volume-weighted development fitted on the paper's restated triangle.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    means = dict(zip(genins.development, THETA_10))
    row = _ones(genins).iloc[..., -1:, :]
    row.values = row.values * np.array(THETA_10, dtype="float64")
    kwargs: dict[str, Any] = {"basic_limit": 500_000, "target_layer": (0, 1_000_000)}
    from_dict = cl.Sahasrabuddhe(means=means, **kwargs).fit(paper_development)
    from_triangle = cl.Sahasrabuddhe(means=row, **kwargs).fit(paper_development)
    assert np.allclose(
        _values(from_dict.cdf_, 0, 0), _values(from_triangle.cdf_, 0, 0), atol=atol
    )


def test_triangle_without_a_trend_assumes_no_trend(
    genins: Triangle,
    atol: float | int,
) -> None:
    """
    No trend is a constant cost level: the same as a trend of all ones.

    Parameters
    ----------
    genins: Triangle
        The genins sample data set.
    atol: float
        The absolute tolerance used for comparisons.

    Returns
    -------
    None

    """
    kwargs: dict[str, Any] = {
        "means": dict(zip(genins.development, THETA_10)),
        "observed_limit": 1_000_000,
        "target_layer": (0, 500_000),
    }
    bare = cl.Sahasrabuddhe(**kwargs).fit(genins)
    constant = cl.Sahasrabuddhe(trend=_ones(genins), **kwargs).fit(genins)
    assert np.allclose(
        _values(bare.triangle_, 0, 0), _values(constant.triangle_, 0, 0), atol=atol
    )
    assert bare.means_.shape[-2] == 1


def test_rejects_an_unfitted_development_estimator(
    paper_layer: Callable[..., cl.Sahasrabuddhe],
) -> None:
    """
    An unfitted estimator carries no pattern, and is not claims either, so it is
    refused rather than treated as a Triangle of claims.

    Parameters
    ----------
    paper_layer: Callable[..., cl.Sahasrabuddhe]
        Builds a Sahasrabuddhe for the paper's layer, ready to fit a pattern.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError, match="X must be a Triangle of claims"):
        paper_layer(target_layer=(0, 500_000)).fit(cl.Development())
