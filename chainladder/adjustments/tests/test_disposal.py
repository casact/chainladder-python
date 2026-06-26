mport chainladder as cl
import numpy as np
import pytest

def test_friedland_fidelity() -> None:
    '''
    Demonstrates 
    '''
    tri = cl.load_sample('friedland_gl_insurer')
    ccc_dev = cl.Development(n_periods=3, average='volume').fit_transform(tri['Closed Claim Counts'])
    ccc_dev.ldf_ = ccc_dev.ldf_.round(3)
    ccc_dev_wtail = cl.TailConstant(tail = 1.100, projection_period = 0).fit_transform(ccc_dev)
    ccc_ult = cl.Chainladder().fit(ccc_dev_wtail).ultimate_
    rcc_dev = cl.Development(n_periods=3, average='volume').fit_transform(tri['Reported Claim Counts'])
    rcc_dev.ldf_ = rcc_dev.ldf_.round(3)
    rcc_ult = cl.Chainladder().fit(rcc_dev).ultimate_
    ult = (ccc_ult + rcc_ult) / 2
    dr = cl.DisposalRate(n_periods = 5, average = 'simple', drop_high = 1, drop_low = 1).fit_transform(X=tri['Closed Claim Counts'],sample_weight=ult)
    assert np.all(dr.disposal_.round(3).values.flatten() == [.200,.433,.585,.710,.791,.862,.882,.912,1.000])
    #Friedland uses rounded ultimates to calculate bottom half of the triangle, which introduces some rounding discrepancies with the implementation
    lhs = (dr.full_triangle_.cum_to_incr()-tri['Closed Claim Counts'].cum_to_incr()).round(0).values.flatten()
    rhs = np.array([
                                                            77.,  
                                                    24.,    70.,  
                                            12.,    18.,    54.,  
                                    46.,    13.,    19.,    57.,  
                            52.,    45.,    13.,    19.,    56.,  
                    76.,    49.,    43.,    12.,    18.,    54.,  
            67.,    55.,    36.,    31.,    9.,     13.,    39.,  
    140.,   91.,    75.,    49.,    43.,    12.,    18.,    53.
    ])
    assert np.all(abs(lhs[~np.isnan(lhs)] - rhs <= 1))

def test_no_weight_exception(raa:Triangle) -> None:
    '''
    sample_weight is optional in the default sklearn API. however, we require sample_weight to provide the a priori ultimate. 
    '''
    with pytest.raises(ValueError):
        dr = cl.DisposalRate().fit(raa)
    ult = cl.Chainladder().fit(raa).ultimate_
    dr = cl.DisposalRate().fit(raa,sample_weight=ult)
    with pytest.raises(ValueError):
        est = dr.transform(raa)
    
def test_cl_parity(raa:Triangle) -> None:
    """
    A no-tail, full-triangle, volume-weighted Chainladder estimator coincides with the disposal rate adjustment. 
    """
    tri = raa.set_backend('sparse')
    dev = cl.Development().fit_transform(tri)
    est = cl.Chainladder().fit(dev)
    dr = cl.DisposalRate().fit_transform(raa,sample_weight=est.ultimate_)
    assert np.all(dr.full_triangle_.round(3).values[...,:-1] == est.full_triangle_.round(3).values[...,:-2])

def test_sparse_transform(raa:Triangle) -> None:
    """
    if the supplied Triangle is sparse, then the resulting full_triangle_ is also sparse 
    """
    raa_sparse = raa.set_backend('sparse')
    ult = cl.Chainladder().fit(raa_sparse).ultimate_.set_backend('sparse')
    dr = cl.DisposalRate().fit_transform(raa_sparse,sample_weight=ult)
    from chainladder.utils.sparse import sp
    assert isinstance(dr.full_triangle_.values,sp.COO)
    
