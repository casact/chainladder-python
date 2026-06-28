from __future__ import annotations

import chainladder as cl
import numpy as np
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core import Triangle

def test_friedland_fidelity() -> None:
    '''
    Reconciles to Chapter 11 Exhibit 5 of the Friedland reserving textbook
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
    dr_transformer = cl.DisposalRate(n_periods = 5, average = 'simple', drop_high = 1, drop_low = 1).fit(X=tri['Closed Claim Counts'],sample_weight=ult)
    dr_transformer.disposal_rate_ = dr_transformer.disposal_rate_.round(3)
    assert np.all(dr_transformer.disposal_rate_.values.flatten() == [.200,.433,.585,.710,.791,.862,.882,.912,1.000])
    #Friedland uses rounded ultimates to calculate bottom half of the triangle, which introduces some rounding discrepancies with the implementation
    dr = dr_transformer.transform(X=tri['Closed Claim Counts'],sample_weight=ult.round(0))
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

def test_no_disposal_exception(raa:Triangle) -> None:
    '''
    disposal attributes are available in Triangle through the mixin, but not available before fitting 
    '''
    with pytest.raises(AttributeError):
        _ = raa.disposal_rate_
    with pytest.raises(AttributeError):
        _ = raa.incr_disposal_rate_

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
    
def test_setting_incr(raa:Triangle) -> None:
    """
    DisposalMixin allows setting incremental disposal rates. Validating that the setting function works properly. 
    """
    ult = cl.Chainladder().fit(raa).ultimate_
    dr = cl.DisposalRate(n_periods = 4).fit(raa,sample_weight=ult)
    orig_disposal_rate_ = dr.disposal_rate_
    dr.incr_disposal_rate_ = dr.incr_disposal_rate_ * 2
    assert dr.disposal_rate_ == orig_disposal_rate_ * 2