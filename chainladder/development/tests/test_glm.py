import numpy as np

import chainladder as cl


def test_basic_odp_cl(genins):
    """
    Without the penalty, the over-dispersed Poisson GLM reproduces chain
    ladder, so TweedieGLM(alpha=0) must match its factors and ultimates.
    """
    glm = cl.TweedieGLM(alpha=0).fit(genins)
    np.testing.assert_allclose(
        glm.ldf_.values[..., :1, :],
        cl.Development().fit(genins).ldf_.values,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        cl.Chainladder().fit(glm.transform(genins)).ultimate_.values,
        cl.Chainladder().fit(genins).ultimate_.values,
        rtol=1e-5,
    )


def test_drop_valuation_without_sample_weight(genins):
    """
    drop_valuation must take effect even when sample_weight is None, and
    must match the result of passing an all-ones sample_weight. The
    unweighted fit itself must be unchanged.

    https://github.com/casact/chainladder/issues/1619
    """
    base = cl.TweedieGLM().fit(genins).ldf_.values[0, 0, 0, :4]
    dropped = cl.TweedieGLM(drop_valuation="2005").fit(genins).ldf_.values[0, 0, 0, :4]
    dropped_weighted = (
        cl
        .TweedieGLM(drop_valuation="2005")
        .fit(genins, sample_weight=genins * 0 + 1)
        .ldf_.values[0, 0, 0, :4]
    )
    assert np.round(base, 4).tolist() == [3.491, 1.7474, 1.4574, 1.1739]
    assert np.round(dropped, 4).tolist() == [3.5057, 1.7861, 1.4546, 1.1711]
    assert np.round(dropped_weighted, 4).tolist() == [3.5057, 1.7861, 1.4546, 1.1711]
    # list form goes through the same TriangleWeight path
    dropped_list = (
        cl
        .TweedieGLM(drop_valuation=["2005", "2006"])
        .fit(genins)
        .ldf_.values[0, 0, 0, :4]
    )
    assert np.round(dropped_list, 4).tolist() == [3.6441, 1.8049, 1.4452, 1.1695]


def test_drop_without_sample_weight(genins):
    """
    drop must take effect even when sample_weight is None, while the
    unweighted fit itself is unchanged.

    https://github.com/casact/chainladder/issues/1619
    """
    base = cl.TweedieGLM().fit(genins).ldf_.values[0, 0, 0, :4]
    dropped = cl.TweedieGLM(drop=("2001", 12)).fit(genins).ldf_.values[0, 0, 0, :4]
    assert np.round(base, 4).tolist() == [3.491, 1.7474, 1.4574, 1.1739]
    assert np.round(dropped, 4).tolist() == [3.571, 1.755, 1.4606, 1.1748]


def test_sample_weight(genins):
    glm = cl.TweedieGLM(alpha=0).fit(genins, sample_weight=genins / genins)
    np.testing.assert_allclose(
        cl.Chainladder().fit(glm.transform(genins)).ultimate_.values,
        cl.Chainladder().fit(genins).ultimate_.values,
        rtol=1e-5,
    )


def test_incremental_triangle_with_empty_cells(monthly):
    """
    An incremental triangle can leave a cell empty, such as a development
    quarter with no records, where its cumulative triangle has a value.
    TweedieGLM must fit the incremental triangle, with or without
    drop_valuation, the same way it fits the cumulative one.

    https://github.com/casact/chainladder/issues/1640
    """
    incremental = monthly["reportedCount"].grain("OYDQ")
    cumulative = incremental.incr_to_cum()
    assert np.isnan(incremental.values).sum() > np.isnan(cumulative.values).sum()
    coef = {}
    for name, X in (("incremental", incremental), ("cumulative", cumulative)):
        for drop in (None, "2017Q2"):
            model = cl.TweedieGLM(drop_valuation=drop).fit(X)
            coef[name, drop] = model.model_.estimator_ml["model"].coef_
    for drop in (None, "2017Q2"):
        np.testing.assert_allclose(coef["incremental", drop], coef["cumulative", drop])
    # Dropping 2017Q2 must change the fit. Without this check, the comparison
    # above would also pass if both fits ignored drop_valuation (#1619).
    assert not np.allclose(coef["incremental", None], coef["incremental", "2017Q2"])


def test_alpha_reaches_regressor():
    """
    TweedieGLM must pass alpha to its TweedieRegressor. Without the penalty,
    the over-dispersed Poisson GLM reproduces chain ladder, so
    TweedieGLM(alpha=0) on ukmotor must give the chain ladder ultimates.

    https://github.com/casact/chainladder-python/issues/1584
    """
    ukmotor = cl.load_sample("ukmotor")
    glm = cl.TweedieGLM(alpha=0).fit(ukmotor)
    assert glm.model_.estimator_ml.named_steps["model"].alpha == 0
    np.testing.assert_allclose(
        glm.ldf_.values[..., :1, :],
        cl.Development().fit(ukmotor).ldf_.values,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        cl.Chainladder().fit(glm.transform(ukmotor)).ultimate_.values,
        cl.Chainladder().fit(ukmotor).ultimate_.values,
        rtol=1e-5,
    )
