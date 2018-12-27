import numpy as np
from numpy.testing import assert_allclose
import chainladder as cl
from rpy2.robjects.packages import importr
from rpy2.robjects import r

CL = importr('ChainLadder')


def test_mcl_paid():
    df = r(f'MunichChainLadder(MCLpaid, MCLincurred)').rx('MCLPaid')
    p = cl.MunichAdjustment(paid_to_incurred={'paid':'incurred'}).fit(cl.Development(sigma_interpolation='mack').fit_transform(cl.load_dataset('mcl'))).munich_full_triangle_[0,0,0,:,:]
    arr = np.array(df[0])
    assert_allclose(arr, p, atol=1e-5)


def test_mcl_incurred():
    df = r(f'MunichChainLadder(MCLpaid, MCLincurred)').rx('MCLIncurred')
    p = cl.MunichAdjustment(paid_to_incurred={'paid':'incurred'}).fit(cl.Development(sigma_interpolation='mack').fit_transform(cl.load_dataset('mcl'))).munich_full_triangle_[1,0,0,:,:]
    arr = np.array(df[0])
    assert_allclose(arr, p, atol=1e-5)
