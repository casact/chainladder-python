from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import chainladder as cl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle

# Tests against the two numerical examples of Hürlimann (2009), Credible Loss Ratio Claims
# Reserves, Astin Bulletin 39(1), Section 7. Where the paper's printed table is wrong, the
# values are those of the CAS Exam 7 Syllabus Committee's errata of January 2020.
# The notebook docs/user_guide/hurlimann_verification.ipynb walks through the same tables.

METHODS = ["collective", "individual", "neuhaus", "benktander", "optimal"]

ROWS_7_1 = [
    [4370, 1923, 3999, 2168, 1200, 647],
    [2701, 2590, 1871, 1783, 393],
    [4483, 2246, 3345, 1068],
    [3254, 2550, 2547],
    [8010, 4108],
    [5582],
]
ROWS_7_6 = [
    [3789045, 2860826, 506651, 151996, 65141, 24203],
    [3582774, 2687080, 1250163, 535784, 880143],
    [4221853, 3166390, 2249388, 207853],
    [4074429, 2949557, 1162885],
    [1227618, 3906617],
    [6839930],
]

EXAMPLES = {
    # Tables 7.1 to 7.5
    "mack": {
        "rows": ROWS_7_1,
        "V": [13085, 14258, 16114, 15142, 16905, 20224],
        # Table 7.2: m, p, q, t, Z*
        "parameters": {
            "m": ([0.29667, 0.17770, 0.20072, 0.11549, 0.05826, 0.04945], 5e-6),
            "p": ([1, 0.94496, 0.88010, 0.75153, 0.52808, 0.33026], 5e-6),
            "q": ([0, 0.05504, 0.11990, 0.24847, 0.47192, 0.66974], 5e-6),
            "t": ([1, 0.97209, 0.93814, 0.86691, 0.72669, 0.57469], 5e-6),
            "Z*": ([0.5, 0.49292, 0.48404, 0.46435, 0.42086, 0.36495], 5e-6),
        },
        # Table 7.3, origins 2 to 6
        "reserves": {
            "collective": [705, 1736, 3380, 7166, 12167],
            "individual": [544, 1518, 2761, 10829, 11320],
            "neuhaus": [568, 1564, 2962, 8904, 11916],
            "benktander": [553, 1544, 2915, 9101, 11887],
            "optimal": [626, 1630, 3092, 8708, 11858],
        },
        # Table 7.4 as corrected by the errata, origins 1 to 6
        "ultimates": {
            "collective": [14307, 10043, 12878, 11731, 19284, 17749],
            "individual": [14307, 9882, 12660, 11112, 22947, 16902],
            "neuhaus": [14307, 9906, 12706, 11313, 21022, 17498],
            "benktander": [14307, 9891, 12686, 11266, 21219, 17469],
            "optimal": [14307, 9964, 12772, 11443, 20826, 17440],
        },
        # Table 7.5 as corrected by the errata, origins 2 to 6
        "mse": {
            "collective": [1.027133, 1.058037, 1.115379, 1.198610, 1.244422],
            "individual": [1.028713, 1.065943, 1.153525, 1.376096, 1.740080],
            "neuhaus": [1.014146, 1.023277, 1.023764, 1.003211, 1.008555],
            "benktander": [1.022818, 1.038856, 1.044128, 1.012892, 1.002208],
            "optimal": [1, 1, 1, 1, 1],
        },
    },
    # Tables 7.6 to 7.10
    "project": {
        "rows": ROWS_7_6,
        "V": [8_000_000, 9_000_000, 10_000_000, 10_000_000, 10_000_000, 12_000_000],
        # Table 7.7. The printed m_5 and q_2 are one unit out in the last digit.
        "parameters": {
            "m": ([0.40230, 0.33129, 0.13971, 0.03317, 0.05561, 0.00303], 1e-5),
            "p": ([1, 0.99687, 0.93925, 0.90488, 0.76012, 0.41685], 5e-6),
            "q": ([0, 0.00314, 0.06075, 0.09512, 0.23988, 0.58315], 1e-5),
            "t": ([1, 0.99843, 0.96915, 0.95125, 0.87185, 0.64564], 5e-6),
            "Z*": ([0.5, 0.49961, 0.49217, 0.48751, 0.46577, 0.39233], 5e-6),
        },
        # Table 7.8
        "reserves": {
            "collective": [27228, 586303, 918019, 2315070, 6753523],
            "individual": [28101, 636809, 860619, 1620276, 9568672],
            "neuhaus": [28067, 632085, 867892, 1805379, 7886055],
            "benktander": [28098, 633741, 866079, 1786943, 7927018],
            "optimal": [27664, 611161, 890036, 1991456, 7858000],
        },
        # Table 7.9
        "ultimates": {
            "collective": [7397862, 8963172, 10431787, 9104890, 7449305, 13593453],
            "individual": [7397862, 8964045, 10482293, 9047490, 6754511, 16408602],
            "neuhaus": [7397862, 8964011, 10477569, 9054763, 6939614, 14725985],
            "benktander": [7397862, 8964042, 10479225, 9052950, 6921178, 14766948],
            "optimal": [7397862, 8963608, 10456645, 9076907, 7125691, 14697930],
        },
        # Table 7.10 as corrected by the errata, origins 2 to 6
        "mse": {
            "collective": [1.001566, 1.029900, 1.046373, 1.111729, 1.228789],
            "individual": [1.001571, 1.031833, 1.051248, 1.146989, 1.548852],
            "neuhaus": [1.001342, 1.021187, 1.029039, 1.036940, 1.000148],
            "benktander": [1.001551, 1.024673, 1.033989, 1.044623, 1.000893],
            "optimal": [1, 1, 1, 1, 1],
        },
    },
}


def incremental_triangle(rows) -> Triangle:
    """The paper's incremental triangle, with periods 1 to 6 labelled 2001 to 2006."""
    long = pd.DataFrame(
        [
            (2001 + i, 2001 + i + k, v)
            for i, r in enumerate(rows)
            for k, v in enumerate(r)
        ],
        columns=["origin", "valuation", "paid"],
    )
    return cl.Triangle(
        long, origin="origin", valuation="valuation", columns="paid", cumulative=False
    )


def exposure(V) -> Triangle:
    return cl.Triangle(
        pd.DataFrame({"origin": range(2001, 2007), "valuation": 2006, "premium": V}),
        origin="origin",
        valuation="valuation",
        columns="premium",
        cumulative=True,
    )


def values(tri: Triangle) -> np.ndarray:
    """A Triangle's values, one per origin or development period."""
    return tri.to_frame().values.ravel()


@pytest.fixture(params=list(EXAMPLES))
def example(request):
    ex = EXAMPLES[request.param]
    return ex, incremental_triangle(ex["rows"]), exposure(ex["V"])


@pytest.fixture
def fits(example):
    _, X, w = example
    return {m: cl.HurlimannCLR(method=m).fit(X, sample_weight=w) for m in METHODS}


# Reproducing the paper


def test_parameters(example, fits):
    """Tables 7.2 and 7.7"""
    ex, _, _ = example
    fit = fits["optimal"]
    p = values(fit.p_)
    actual = {
        "m": values(fit.m_),
        "p": p,
        "q": 1 - p,
        "t": np.sqrt(p),
        "Z*": values(fit.Z_),
    }
    for name, (printed, atol) in ex["parameters"].items():
        assert np.allclose(actual[name], printed, rtol=0, atol=atol), name


@pytest.mark.parametrize("method", METHODS)
def test_reserves(example, fits, method):
    """Tables 7.3 and 7.8"""
    ex, _, _ = example
    ibnr = values(fits[method].ibnr_)
    # Origin 1 is fully paid out, so its IBNR is nan rather than zero.
    assert np.isnan(ibnr[0]) or ibnr[0] == 0
    assert np.allclose(ibnr[1:], ex["reserves"][method], rtol=0, atol=1)


@pytest.mark.parametrize("method", METHODS)
def test_ultimates(example, fits, method):
    """Tables 7.4 and 7.9"""
    ex, _, _ = example
    assert np.allclose(
        values(fits[method].ultimate_), ex["ultimates"][method], rtol=0, atol=1
    )


@pytest.mark.parametrize("method", METHODS)
def test_mse_ratio_to_optimal(example, fits, method):
    """Tables 7.5 and 7.10"""
    ex, _, _ = example
    ratio = values(fits[method].mse_)[1:] / values(fits["optimal"].mse_)[1:]
    assert np.allclose(ratio, ex["mse"][method], rtol=0, atol=5e-6)


# Properties of the estimator


def test_sample_weight_required(example):
    _, X, _ = example
    with pytest.raises(ValueError, match="sample_weight"):
        cl.HurlimannCLR().fit(X)


def test_invalid_method(example):
    _, X, w = example
    with pytest.raises(ValueError, match="method"):
        cl.HurlimannCLR(method="mack").fit(X, sample_weight=w)


def test_default_method_is_optimal(example):
    _, X, w = example
    default = cl.HurlimannCLR().fit(X, sample_weight=w)
    optimal = cl.HurlimannCLR(method="optimal").fit(X, sample_weight=w)
    assert np.allclose(
        np.nan_to_num(default.ultimate_.values), np.nan_to_num(optimal.ultimate_.values)
    )


def test_predict_not_supported(example):
    _, X, w = example
    with pytest.raises(NotImplementedError):
        cl.HurlimannCLR().fit(X, sample_weight=w).predict(X, sample_weight=w)


def test_origin_mismatch(example):
    _, X, w = example
    with pytest.raises(ValueError, match="origin"):
        cl.HurlimannCLR().fit(X, sample_weight=w.iloc[..., :5, :])


def test_attributes_are_triangles(example, fits):
    _, X, _ = example
    fit = fits["optimal"]
    for name in (
        "p_",
        "Z_",
        "r_individual_",
        "r_collective_",
        "mse_",
        "ultimate_",
        "ibnr_",
    ):
        tri = getattr(fit, name)
        assert isinstance(tri, cl.Triangle), name
        assert tri.shape == (1, 1, 6, 1), name
        assert list(tri.origin.year) == list(range(2001, 2007)), name
    assert fit.m_.shape == (1, 1, 1, 6)
    assert fit.elr_.shape == (1, 1, 1, 1)
    # The burning cost loss ratio is the sum of the loss ratio pattern.
    assert np.isclose(values(fit.elr_)[0], values(fit.m_).sum())


def test_incremental_equals_cumulative(example, fits):
    _, X, w = example
    fit = cl.HurlimannCLR().fit(X.incr_to_cum(), sample_weight=w)
    assert np.allclose(
        np.nan_to_num(fit.ultimate_.values),
        np.nan_to_num(fits["optimal"].ultimate_.values),
    )


def test_credibility_weights(example, fits):
    _, X, _ = example
    p = values(fits["optimal"].p_)
    elr = values(fits["optimal"].elr_)[0]
    assert np.allclose(values(fits["collective"].Z_), 0)
    assert np.allclose(values(fits["individual"].Z_), 1)
    assert np.allclose(values(fits["benktander"].Z_), p)
    assert np.allclose(values(fits["neuhaus"].Z_), p * elr)
    assert np.allclose(values(fits["optimal"].Z_), p / (p + np.sqrt(p)))
    # The optimal weight never exceeds 1/2 (eq. 6.4).
    assert (values(fits["optimal"].Z_) <= 0.5).all()


@pytest.mark.parametrize("method", METHODS)
def test_credibility_mixture(fits, method):
    """The IBNR is the credibility mixture of the individual and collective reserves (eq. 3.1)."""
    fit = fits[method]
    z = values(fit.Z_)
    expected = z * values(fit.r_individual_) + (1 - z) * values(fit.r_collective_)
    ibnr = np.nan_to_num(values(fit.ibnr_))
    assert np.allclose(ibnr[1:], expected[1:])


def test_pattern_independent_of_method(fits):
    for method in METHODS:
        assert np.allclose(values(fits[method].p_), values(fits["optimal"].p_))
        assert np.allclose(values(fits[method].m_), values(fits["optimal"].m_))


def test_fully_developed_origin(fits):
    """The oldest origin has q = 0, so no reserve and a finite mean squared error."""
    for fit in fits.values():
        assert np.isclose(values(fit.p_)[0], 1)
        assert np.isclose(values(fit.mse_)[0], 0)
        assert np.isclose(values(fit.ultimate_)[0], values(fit.X_.latest_diagonal)[0])


@pytest.mark.parametrize(
    "method", ["collective", "individual", "benktander", "optimal"]
)
def test_exposure_scale(example, method):
    """
    Rescaling the exposure rescales the burning cost loss ratio but not the reserves.
    Neuhaus is excluded: its weight, the cumulative pattern, depends on the scale.
    """
    _, X, w = example
    a = cl.HurlimannCLR(method=method).fit(X, sample_weight=w)
    b = cl.HurlimannCLR(method=method).fit(X, sample_weight=w * 7)
    assert np.isclose(values(b.elr_)[0], values(a.elr_)[0] / 7)
    assert np.allclose(values(b.p_), values(a.p_))
    assert np.allclose(values(b.ultimate_), values(a.ultimate_))


def test_estimator_api(example):
    _, X, w = example
    est = clone(cl.HurlimannCLR(method="neuhaus"))
    assert est.get_params() == {"method": "neuhaus"}
    assert est.fit(X, sample_weight=w) is est


# Triangles other than the paper's, which also run on every array backend.


def test_backends(raa: Triangle) -> None:
    """The result does not depend on the array backend."""
    w = raa.latest_diagonal * 0 + 10_000
    fit = cl.HurlimannCLR().fit(raa, sample_weight=w)
    expected = cl.HurlimannCLR().fit(
        raa.set_backend("numpy"), sample_weight=w.set_backend("numpy")
    )
    assert fit.ultimate_.array_backend == raa.array_backend
    assert np.allclose(
        np.nan_to_num(fit.ultimate_.set_backend("numpy").values),
        np.nan_to_num(expected.ultimate_.values),
    )


def test_segments_equal_separate_fits(clrd: Triangle) -> None:
    """Fitting several segments at once is the same as fitting each separately."""
    clrd = clrd.groupby("LOB").sum()
    X = clrd[["CumPaidLoss"]]
    w = clrd["EarnedPremDIR"].latest_diagonal
    fit = cl.HurlimannCLR().fit(X, sample_weight=w)
    assert fit.ultimate_.shape == (len(clrd), 1, 10, 1)
    for i in range(len(clrd)):
        single = cl.HurlimannCLR().fit(X.iloc[i], sample_weight=w.iloc[i])
        assert np.allclose(
            np.nan_to_num(single.ultimate_.set_backend("numpy").values),
            np.nan_to_num(fit.ultimate_.set_backend("numpy").values[i : i + 1]),
        )
