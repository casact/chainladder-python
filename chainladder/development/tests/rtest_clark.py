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
def test_clarkldf(genins):
    model = cl.ClarkLDF().fit(genins)
    df = r("ClarkLDF(GenIns)").rx("THETAG")
    r_omega = df[0][0]
    r_theta = df[0][1]
    assert abs(model.omega_.iloc[0, 0] - r_omega) < 1e-2
    assert abs(model.theta_.iloc[0, 0] / 12 - r_theta) < 1e-2

@pytest.mark.r
def test_clarkldf_weibull(genins):
    model = cl.ClarkLDF(growth="weibull").fit(genins)
    df = r('ClarkLDF(GenIns, G="weibull")').rx("THETAG")
    r_omega = df[0][0]
    r_theta = df[0][1]
    assert abs(model.omega_.iloc[0, 0] - r_omega) < 1e-2
    assert abs(model.theta_.iloc[0, 0] / 12 - r_theta) < 1e-2

@pytest.mark.r
def test_clarkcapecod(genins):
    df = r("ClarkCapeCod(GenIns, Premium=10000000+400000*0:9)")
    r_omega = df.rx("THETAG")[0][0]
    r_theta = df.rx("THETAG")[0][1]
    r_elr = df.rx("ELR")[0][0]
    premium = genins.latest_diagonal * 0 + 1
    premium.values = (np.arange(10) * 400000 + 10000000)[None, None, :, None]
    model = cl.ClarkLDF().fit(genins, sample_weight=premium)
    assert abs(model.omega_.iloc[0, 0] - r_omega) < 1e-2
    assert abs(model.theta_.iloc[0, 0] / 12 - r_theta) < 1e-2
    assert abs(model.elr_.iloc[0, 0] - r_elr) < 1e-2

@pytest.mark.r
def test_clarkcapcod_weibull(genins):
    df = r('ClarkCapeCod(GenIns, Premium=10000000+400000*0:9, G="weibull")')
    r_omega = df.rx("THETAG")[0][0]
    r_theta = df.rx("THETAG")[0][1]
    r_elr = df.rx("ELR")[0][0]
    premium = genins.latest_diagonal * 0 + 1
    premium.values = (np.arange(10) * 400000 + 10000000)[None, None, :, None]
    model = cl.ClarkLDF(growth="weibull").fit(genins, sample_weight=premium)
    assert abs(model.omega_.iloc[0, 0] - r_omega) < 1e-2
    assert abs(model.theta_.iloc[0, 0] / 12 - r_theta) < 1e-2
    assert abs(model.elr_.iloc[0, 0] - r_elr) < 1e-2
