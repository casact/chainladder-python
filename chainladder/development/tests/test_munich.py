import numpy as np
from chainladder.utils.cupy import cp
import chainladder as cl
from rpy2.robjects.packages import importr
from rpy2.robjects import r

CL = importr('ChainLadder')


def test_mcl_paid():
    df = r('MunichChainLadder(MCLpaid, MCLincurred)').rx('MCLPaid')
    p = cl.MunichAdjustment(paid_to_incurred={'paid':'incurred'}).fit(cl.Development(sigma_interpolation='mack').fit_transform(cl.load_dataset('mcl'))).munich_full_triangle_[0,0,0,:,:]
    xp = cp.get_array_module(p)
    arr = xp.array(df[0])
    xp.testing.assert_allclose(arr, p, atol=1e-5)


def test_mcl_incurred():
    df = r('MunichChainLadder(MCLpaid, MCLincurred)').rx('MCLIncurred')
    p = cl.MunichAdjustment(paid_to_incurred={'paid':'incurred'}).fit(cl.Development(sigma_interpolation='mack').fit_transform(cl.load_dataset('mcl'))).munich_full_triangle_[1,0,0,:,:]
    xp = cp.get_array_module(p)
    arr = xp.array(df[0])
    xp.testing.assert_allclose(arr, p, atol=1e-5)
