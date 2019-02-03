import chainladder as cl
import numpy as np
from numpy.testing import assert_allclose


def test_schmidt():
    tri = cl.load_dataset('ia_sample')
    ia = cl.IncrementalAdditive()
    answer = ia.fit_transform(tri.iloc[0, 1],
                              sample_weight=tri.iloc[0, 0].latest_diagonal)
    answer = answer.incremental_.incr_to_cum().triangle[0, 0, :, -1]
    check = np.array([3483., 4007.84795031, 4654.36196862, 5492.00685523,
                      6198.10197128, 7152.82539296])
    assert_allclose(answer, check, atol=1e-5)
