"""Tests for matching ``sample_weight`` to ``X`` by origin label (#1646)."""

import numpy as np
import pandas as pd
import pytest

import chainladder as cl

raa = cl.load_sample("RAA")


def _premium(origins, values=None):
    if values is None:
        values = [20000.0 + 2000.0 * i for i in range(len(origins))]
    df = pd.DataFrame({"origin": origins, "premium": values})
    return cl.Triangle(df, origin="origin", columns="premium", cumulative=True)


@pytest.mark.parametrize(
    "estimator",
    [cl.BornhuetterFerguson, cl.Benktander, cl.ExpectedLoss, cl.CapeCod],
)
def test_sample_weight_is_matched_to_x_by_origin(estimator):
    """A weight labelled 1982-1991 on raa (1981-1990) is matched by label."""
    years = list(range(1981, 1991))
    shifted = _premium([y + 1 for y in years])
    # The same premium per label on raa's origins, with none for 1981.
    matched = _premium(years, [np.nan] + list(shifted.values[0, 0, :-1, 0]))
    expected = estimator().fit(raa, sample_weight=matched).ultimate_
    expected = expected.values[0, 0, 1:, 0]
    with pytest.warns(UserWarning, match=r"\['1981'\].*\['1991'\]"):
        est = estimator().fit(raa, sample_weight=shifted)
    with pytest.warns(UserWarning, match="matched to X by origin"):
        predicted = est.predict(raa, sample_weight=shifted)
    for ultimate in (est.ultimate_, predicted.ultimate_):
        assert np.isnan(ultimate.values[0, 0, 0, 0])
        np.testing.assert_allclose(ultimate.values[0, 0, 1:, 0], expected)


def test_bornhuetter_ferguson_uses_the_premium_labelled_with_the_origin():
    """The 1990 ultimate uses the premium labelled 1990, from #1646."""
    years = list(range(1981, 1991))
    shifted = _premium([y + 1 for y in years])
    with pytest.warns(UserWarning):
        est = cl.BornhuetterFerguson(apriori=0.6).fit(raa, sample_weight=shifted)
    assert est.ultimate_.values[0, 0, -1, 0] == pytest.approx(21241.54, abs=0.01)


def test_sample_weight_matching_works_on_the_sparse_backend():
    """Matching by origin gives the same ultimates on the sparse backend."""
    years = list(range(1981, 1991))
    shifted = _premium([y + 1 for y in years])
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
