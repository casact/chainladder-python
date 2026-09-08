import chainladder as cl
import numpy as np


def test_pattern_cum_to_incr_zero_cells_stay_finite(raa):
    cdf = cl.Development().fit(raa).cdf_
    cdf.values[..., 1] = 0
    out = cdf.cum_to_incr()
    assert not np.isinf(out.values).any()
