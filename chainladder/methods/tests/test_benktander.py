from __future__ import annotations

import pytest
import numpy as np
import chainladder as cl
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


@pytest.fixture
def atol():
    return 1e-5


data = ["RAA", "ABC", "GenIns", "MW2008", "MW2014"]


def test_bk_fit_weight(raa: Triangle) -> None:
    """
    Test validation of sample_weight requirement. Should raise a value error if no weight is supplied.
    """
    with pytest.raises(ValueError):
        cl.Benktander().fit(raa)


@pytest.mark.parametrize("data", data)
def test_benktander_to_chainladder(data, atol):
    tri = cl.load_sample(data)
    a = cl.Chainladder().fit(tri).ibnr_
    b = cl.Benktander(apriori=0.8, n_iters=255).fit(tri, sample_weight=a).ibnr_
    xp = tri.get_array_module()
    assert xp.allclose(xp.nan_to_num(a.values), xp.nan_to_num(b.values), atol=atol)


def test_bf_eq_cl_when_using_cl_apriori(qtr):
    cl_ult = cl.Chainladder().fit(qtr).ultimate_
    bf_ult = cl.BornhuetterFerguson().fit(qtr, sample_weight=cl_ult).ultimate_
    xp = cl_ult.get_array_module()
    assert xp.allclose(cl_ult.values, bf_ult.values, atol=1e-5)


def test_different_backends(clrd):
    clrd = clrd[["CumPaidLoss", "EarnedPremDIR"]]
    clrd = clrd[clrd["LOB"] == "wkcomp"]
    model = cl.BornhuetterFerguson().fit(
        clrd["CumPaidLoss"].sum().set_backend("numpy"),
        sample_weight=clrd["EarnedPremDIR"].sum().latest_diagonal.set_backend("numpy"),
    )
    assert (
        abs(
            (
                model.predict(
                    clrd["CumPaidLoss"].set_backend("sparse"),
                    sample_weight=clrd["EarnedPremDIR"].latest_diagonal.set_backend(
                        "sparse"
                    ),
                ).ibnr_.sum()
                - model.ibnr_
            ).sum()
        )
        < 1
    )


def test_odd_shaped_triangle():
    df = pd.DataFrame({
        "claim_year": 2000 + pd.Series([0] * 8 + [1] * 4),
        "claim_month": [1, 4, 7, 10] * 3,
        "dev_year": 2000 + pd.Series([0] * 4 + [1] * 8),
        "dev_month": [1, 4, 7, 10] * 3,
        "payment": [1] * 12,
    })
    tr = cl.Triangle(
        df,
        origin=["claim_year", "claim_month"],
        valuation=["dev_year", "dev_month"],
        columns="payment",
        cumulative=False,
    )
    atr = tr.grain("OYDQ")
    ult1 = (
        cl
        .Benktander(apriori=1, n_iters=10000)
        .fit(
            cl.Development(average="volume").fit_transform(atr),
            sample_weight=atr.latest_diagonal,
        )
        .ultimate_.sum()
    )
    ult2 = (
        cl
        .Benktander(apriori=1, n_iters=10000)
        .fit(
            cl.Development(average="volume").fit_transform(tr),
            sample_weight=tr.latest_diagonal,
        )
        .ultimate_.grain("OYDQ")
        .sum()
    )
    assert abs(ult1 - ult2) < 1e-5


def test_bf_apriori_sigma_is_lognormal(genins: Triangle) -> None:
    """apriori_sigma must draw a strictly-positive lognormal apriori (issue #1143)."""
    boot = cl.BootstrapODPSample(n_sims=50000, random_state=42).fit_transform(genins)
    w = boot.latest_diagonal.copy()
    w.values = np.ones_like(
        w.values
    )  # sample_weight=1 -> expectation_ == raw multiplier

    sigma = 0.8
    bf = cl.BornhuetterFerguson(apriori=1.0, apriori_sigma=sigma, random_state=7).fit(
        boot, sample_weight=w
    )

    mult = np.nanmean(bf.expectation_.values[:, 0, :, 0], axis=1)
    mult = mult[np.isfinite(mult)]

    assert (mult > 0).all()  # strictly positive -> lognormal, not normal
    assert abs(mult.mean() - 1.0) < 0.02  # mean preserved (E = apriori)
    assert abs(mult.std() - sigma) < 0.03  # SD preserved (== apriori_sigma)


def test_capecod_apriori_sigma_is_positive(genins: Triangle) -> None:
    """CapeCod scatters aprioris through the same sampler; must stay positive (#1143)."""
    boot = cl.BootstrapODPSample(n_sims=10000, random_state=42).fit_transform(genins)
    premium = boot.latest_diagonal * 0 + 8e6

    cc = cl.CapeCod(apriori_sigma=0.8, random_state=7).fit(boot, sample_weight=premium)

    vals = cc.expectation_.values
    vals = vals[np.isfinite(vals)]
    assert (vals > 0).all()  # ~10% would be negative under the old normal


years = list(range(1981, 1991))


def _premium(origins, values=None, backend="numpy"):
    """An annual premium triangle on ``origins``, for aligning with raa (#1646)."""
    if values is None:
        values = [20000.0 + 2000.0 * (o - 1982) for o in origins]
    df = pd.DataFrame({"origin": [str(o) for o in origins], "premium": values})
    tri = cl.Triangle(df, origin="origin", columns="premium", cumulative=True)
    return tri.set_backend(backend)


def _same_labels(origins, backend="numpy"):
    """The premium of ``_premium`` on raa's origins, NaN outside ``origins``."""
    values = [20000.0 + 2000.0 * (y - 1982) if y in origins else np.nan for y in years]
    return _premium(years, values, backend)


def _values(tri: Triangle) -> np.ndarray:
    return tri.set_backend("numpy").values


@pytest.mark.parametrize(
    "estimator",
    [cl.BornhuetterFerguson, cl.Benktander, cl.ExpectedLoss, cl.CapeCod],
)
def test_shifted_sample_weight_is_aligned_to_the_shared_origins(
    raa: Triangle, estimator
) -> None:
    """A weight labelled 1982-1991 on raa (1981-1990) uses 1982-1990."""
    backend = raa.array_backend
    shifted = _premium(range(1982, 1992), backend=backend)
    matched = _same_labels(range(1982, 1991), backend)
    expected = _values(estimator().fit(raa, sample_weight=matched).ultimate_)
    with pytest.warns(UserWarning, match=r"no ultimate: \['1981'\]"):
        est = estimator().fit(raa, sample_weight=shifted)
    with pytest.warns(UserWarning, match="aligned to the origins it shares"):
        predicted = est.predict(raa, sample_weight=shifted)
    for ultimate in (est.ultimate_, predicted.ultimate_):
        values = _values(ultimate)
        assert np.isnan(values[0, 0, 0, 0])
        np.testing.assert_allclose(values[0, 0, 1:, 0], expected[0, 0, 1:, 0])


def test_bornhuetter_ferguson_uses_the_premium_labelled_with_the_origin(
    raa: Triangle,
) -> None:
    """The 1990 ultimate uses the premium labelled 1990, from #1646."""
    shifted = _premium(range(1982, 1992), backend=raa.array_backend)
    with pytest.warns(UserWarning):
        est = cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=shifted)
    assert _values(est.ultimate_)[0, 0, -1, 0] == pytest.approx(21241.54, abs=0.01)


def test_longer_sample_weight_is_aligned_to_the_origins_of_x(
    raa: Triangle, recwarn
) -> None:
    """A weight labelled 1979-1992 gives the same ultimates as one on 1981-1990."""
    backend = raa.array_backend
    longer = cl.BornhuetterFerguson(apriori=0.6).fit(
        raa, sample_weight=_premium(range(1979, 1993), backend=backend)
    )
    same = cl.BornhuetterFerguson(apriori=0.6).fit(
        raa, sample_weight=_premium(years, backend=backend)
    )
    np.testing.assert_allclose(_values(longer.ultimate_), _values(same.ultimate_))
    assert not [w for w in recwarn if "aligned" in str(w.message)]


def test_shorter_sample_weight_leaves_origins_without_weight_with_no_ultimate(
    raa: Triangle,
) -> None:
    """A weight labelled 1983-1988 gives no ultimate for 1981, 1982, 1989, 1990."""
    backend = raa.array_backend
    expected = cl.BornhuetterFerguson(apriori=0.6).fit(
        raa, sample_weight=_same_labels(range(1983, 1989), backend)
    )
    with pytest.warns(UserWarning, match=r"\['1981', '1982', '1989', '1990'\]"):
        est = cl.BornhuetterFerguson(apriori=0.6).fit(
            raa, sample_weight=_premium(range(1983, 1989), backend=backend)
        )
    ultimate = _values(est.ultimate_)[0, 0, :, 0]
    assert np.isnan(ultimate[[0, 1, 8, 9]]).all()
    np.testing.assert_allclose(ultimate[2:8], _values(expected.ultimate_)[0, 0, 2:8, 0])


def test_sample_weight_with_another_origin_grain_raises(raa: Triangle) -> None:
    """A quarterly weight on annual raa raises instead of lining up by position."""
    origins = pd.date_range("1981-01-01", periods=40, freq="QS")
    df = pd.DataFrame({"origin": origins, "premium": 5000.0})
    quarterly = cl.Triangle(df, origin="origin", columns="premium", cumulative=True)
    quarterly = quarterly.set_backend(raa.array_backend)
    with pytest.raises(ValueError, match="origin grain"):
        cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=quarterly)


def test_origins_that_do_not_line_up_raise(raa: Triangle) -> None:
    """Odd-year X with an even-year weight raises, as with #1649's slices of raa."""
    odd = raa[raa.origin.year.isin([1981, 1983, 1985, 1987, 1989])]
    even = _premium(years, backend=raa.array_backend)
    even = even[even.origin.year.isin([1982, 1984, 1986, 1988, 1990])]
    with pytest.raises(ValueError, match="do not line up"):
        cl.BornhuetterFerguson(apriori=0.6).fit(odd, sample_weight=even)


def test_single_origin_sample_weight_broadcasts(raa: Triangle) -> None:
    """A weight with one origin still applies to every origin of X."""
    weight = _premium([1990], backend=raa.array_backend)
    est = cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=weight)
    assert est.ultimate_.shape == (1, 1, 10, 1)
    assert not np.isnan(_values(est.ultimate_)).any()
