"""Tests for slicing ``sample_weight`` to the origins it shares with ``X`` (#1646)."""

import numpy as np
import pandas as pd
import pytest

import chainladder as cl

raa = cl.load_sample("RAA")
years = list(range(1981, 1991))


def _premium(origins, values=None):
    if values is None:
        values = [20000.0 + 2000.0 * (o - 1982) for o in origins]
    df = pd.DataFrame({"origin": [str(o) for o in origins], "premium": values})
    return cl.Triangle(df, origin="origin", columns="premium", cumulative=True)


def _same_labels(origins):
    """The premium of ``_premium`` on raa's origins, NaN outside ``origins``."""
    values = [20000.0 + 2000.0 * (y - 1982) if y in origins else np.nan for y in years]
    return _premium(years, values)


@pytest.mark.parametrize(
    "estimator",
    [cl.BornhuetterFerguson, cl.Benktander, cl.ExpectedLoss, cl.CapeCod],
)
def test_shifted_sample_weight_is_sliced_to_the_shared_origins(estimator):
    """A weight labelled 1982-1991 on raa (1981-1990) uses 1982-1990."""
    shifted = _premium(range(1982, 1992))
    matched = _same_labels(range(1982, 1991))
    expected = estimator().fit(raa, sample_weight=matched).ultimate_
    expected = expected.values[0, 0, 1:, 0]
    with pytest.warns(UserWarning, match=r"no ultimate: \['1981'\]"):
        est = estimator().fit(raa, sample_weight=shifted)
    with pytest.warns(UserWarning, match="sliced to the origins it shares"):
        predicted = est.predict(raa, sample_weight=shifted)
    for ultimate in (est.ultimate_, predicted.ultimate_):
        assert np.isnan(ultimate.values[0, 0, 0, 0])
        np.testing.assert_allclose(ultimate.values[0, 0, 1:, 0], expected)


def test_bornhuetter_ferguson_uses_the_premium_labelled_with_the_origin():
    """The 1990 ultimate uses the premium labelled 1990, from #1646."""
    shifted = _premium(range(1982, 1992))
    with pytest.warns(UserWarning):
        est = cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=shifted)
    assert est.ultimate_.values[0, 0, -1, 0] == pytest.approx(21241.54, abs=0.01)


def test_longer_sample_weight_is_sliced_to_the_origins_of_x(recwarn):
    """A weight labelled 1979-1992 gives the same ultimates as one on 1981-1990."""
    longer = cl.BornhuetterFerguson(apriori=0.6).fit(
        raa, sample_weight=_premium(range(1979, 1993))
    )
    same = cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=_premium(years))
    np.testing.assert_allclose(longer.ultimate_.values, same.ultimate_.values)
    assert not [w for w in recwarn if "sliced" in str(w.message)]


def test_shorter_sample_weight_leaves_origins_without_weight_with_no_ultimate():
    """A weight labelled 1983-1988 gives no ultimate for 1981, 1982, 1989, 1990."""
    expected = cl.BornhuetterFerguson(apriori=0.6).fit(
        raa, sample_weight=_same_labels(range(1983, 1989))
    )
    with pytest.warns(UserWarning, match=r"\['1981', '1982', '1989', '1990'\]"):
        est = cl.BornhuetterFerguson(apriori=0.6).fit(
            raa, sample_weight=_premium(range(1983, 1989))
        )
    ultimate = est.ultimate_.values[0, 0, :, 0]
    assert np.isnan(ultimate[[0, 1, 8, 9]]).all()
    np.testing.assert_allclose(ultimate[2:8], expected.ultimate_.values[0, 0, 2:8, 0])


def test_sample_weight_with_another_origin_grain_raises():
    """A quarterly weight on annual raa raises instead of lining up by position."""
    origins = pd.date_range("1981-01-01", periods=40, freq="QS")
    df = pd.DataFrame({"origin": origins, "premium": 5000.0})
    quarterly = cl.Triangle(df, origin="origin", columns="premium", cumulative=True)
    with pytest.raises(ValueError, match="origin grain"):
        cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=quarterly)


def test_origins_that_do_not_line_up_raise():
    """Odd-year X with an even-year weight raises, as with #1649's slices of raa."""
    odd = raa[raa.origin.year.isin([1981, 1983, 1985, 1987, 1989])]
    even = _premium(years)
    even = even[even.origin.year.isin([1982, 1984, 1986, 1988, 1990])]
    with pytest.raises(ValueError, match="do not line up"):
        cl.BornhuetterFerguson(apriori=0.6).fit(odd, sample_weight=even)


def test_sample_weight_slicing_works_on_the_sparse_backend():
    """Slicing gives the same ultimates on the sparse backend."""
    shifted = _premium(range(1982, 1992))
    with pytest.warns(UserWarning):
        dense = cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=shifted)
    with pytest.warns(UserWarning):
        sparse = cl.BornhuetterFerguson(apriori=0.6).fit(
            raa.set_backend("sparse"), sample_weight=shifted.set_backend("sparse")
        )
    np.testing.assert_allclose(
        sparse.ultimate_.set_backend("numpy").values, dense.ultimate_.values
    )


def test_single_origin_sample_weight_broadcasts():
    """A weight with one origin still applies to every origin of X."""
    est = cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=_premium([1990]))
    assert est.ultimate_.shape == (1, 1, 10, 1)
    assert not np.isnan(est.ultimate_.values).any()
