### Building out a dev environment with a working copy
### of R ChainLadder is difficult.  These tests are 
### Currently inactive, but available should the compatibility
### of the installs improve at a later date.

import numpy as np
import chainladder as cl
import pytest

try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r
    CL = importr("ChainLadder")
except:
    pass


@pytest.mark.r
def test_mcl_paid():
    df = r("MunichChainLadder(MCLpaid, MCLincurred)").rx("MCLPaid")
    p = cl.MunichAdjustment(paid_to_incurred=("paid", "incurred")).fit(
        cl.Development(sigma_interpolation="mack").fit_transform(cl.load_sample("mcl"))
    )
    xp = p.ldf_.get_array_module()
    arr = xp.array(df[0])
    assert xp.allclose(arr, p.munich_full_triangle_[0, 0, 0, :, :], atol=1e-5)


@pytest.mark.r
def test_mcl_incurred():
    df = r("MunichChainLadder(MCLpaid, MCLincurred)").rx("MCLIncurred")
    p = cl.MunichAdjustment(paid_to_incurred=[("paid", "incurred")]).fit(
        cl.Development(sigma_interpolation="mack").fit_transform(cl.load_sample("mcl"))
    )
    xp = p.ldf_.get_array_module()
    arr = xp.array(df[0])
    assert xp.allclose(arr, p.munich_full_triangle_[1, 0, 0, :, :], atol=1e-5)