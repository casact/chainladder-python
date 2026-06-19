import chainladder as cl
import numpy as np


def test_disposal():
    tri = cl.load_sample('friedland_gl_insurer')['Closed Claim Counts']
    ult_tri = cl.Triangle(
        data = {
            'Closed Claim Counts':[873,720,626,629,588,553,438,609],
            'ay': [2001,2002,2003,2004,2005,2006,2007,2008],
            'dev':[2008,2008,2008,2008,2008,2008,2008,2008],
        },
        origin = 'ay',
        development='dev',
        columns='Closed Claim Counts',
        cumulative=True,
    )
    dr = cl.DisposalRate(n_periods = 5, average = 'simple', drop_high = 1, drop_low = 1).fit_transform(X=tri,sample_weight=ult_tri)
    assert np.all(abs(dr.disposal_.round(3).values.flatten() - [.200,.433,.585,.710,.791,.862,.882,.912,1.000] <=0.001))
    lhs = (dr.full_triangle_.cum_to_incr()-tri.cum_to_incr()).round(0).values.flatten()
    rhs = np.array([
                                                            77.,  
                                                    24.,    70.,  
                                            12.,    18.,    54.,  
                                    46.,    13.,    19.,    57.,  
                            52.,    45.,    13.,    19.,    56.,  
                    76.,    49.,    43.,    12.,    18.,    54.,  
            67.,    55.,    36.,    31.,    9.,     13.,    39.,  
    140.,   91.,    75.,    49.,    42.,    12.,    18.,    53.
    ])
    assert np.all(abs(lhs[~np.isnan(lhs)] - rhs <= 1))

def test_disposal_no_weight(raa):
    tri = raa.set_backend('sparse')
    with pytest.raises(ValueError):
        dr = cl.DisposalRate().fit(tri)
    ult = cl.Chainladder().fit(tri).ultimate_
    dr = cl.DisposalRate().fit(tri,sample_weight=ult)
    with pytest.raises(ValueError):
        est = dr.transform(tri)
    
    