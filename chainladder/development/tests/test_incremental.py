import chainladder as cl
import numpy as np


def test_schmidt():
    tri = cl.load_sample("ia_sample")
    xp = np if tri.array_backend == "sparse" else tri.get_array_module()
    ia = cl.IncrementalAdditive()
    answer = ia.fit_transform(
        tri.iloc[0, 0], sample_weight=tri.iloc[0, 1].latest_diagonal
    )
    answer = ia.incremental_.incr_to_cum().values[0, 0, :, -1]
    check = xp.array(
        [
            3483.0,
            4007.84795031,
            4654.36196862,
            5492.00685523,
            6198.10197128,
            7152.82539296,
        ]
    )
    assert xp.allclose(answer, check, atol=1e-5)

def test_IBNR_methods():
    tri = cl.load_sample("ia_sample")
    incr_est = cl.IncrementalAdditive().fit(tri['loss'], sample_weight=tri['exposure'].latest_diagonal)
    incr_trans = incr_est.transform(tri['loss'])
    incr_tri = incr_est.incremental_.incr_to_cum()
    incr_ult = incr_tri[incr_tri.development == incr_tri.development.max()]
    cl_est = cl.Chainladder().fit(incr_trans)
    bf_est = cl.BornhuetterFerguson(apriori=1).fit(incr_trans, sample_weight=incr_ult)
    incr_result = np.round(incr_est.incremental_.values,5)
    cl_result = np.round(cl_est.full_triangle_.cum_to_incr().values[...,:-2],5)
    bf_result = np.round(bf_est.full_triangle_.cum_to_incr().values[...,:-2],5)
    assert np.all(incr_result == cl_result) & np.all(incr_result == bf_result)
