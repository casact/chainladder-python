import chainladder as cl
import numpy as np


def test_schmidt():
    tri = cl.load_sample('ia_sample')
    xp = np if tri.array_backend == 'sparse' else tri.get_array_module()
    ia = cl.IncrementalAdditive()
    answer = ia.fit_transform(tri.iloc[0, 0],
                              sample_weight=tri.iloc[0, 1].latest_diagonal)
    answer = answer.incremental_.incr_to_cum().values[0, 0, :, -1]
    check = xp.array([3483., 4007.84795031, 4654.36196862, 5492.00685523,
                      6198.10197128, 7152.82539296])
    assert xp.allclose(answer, check, atol=1e-5)
