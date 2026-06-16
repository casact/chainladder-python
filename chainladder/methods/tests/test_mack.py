import chainladder as cl
import numpy as np

def test_mack_to_triangle():
    assert (
        cl.MackChainladder()
        .fit(
            cl.TailConstant().fit_transform(
                cl.Development().fit_transform(cl.load_sample("ABC"))
            )
        )
        .summary_
        == cl.MackChainladder()
        .fit(cl.Development().fit_transform(cl.load_sample("ABC")))
        .summary_
    )

def test_mack_malformed():
    a  = cl.load_sample('raa')
    b = a.iloc[:, :, :-1]
    x = cl.MackChainladder().fit(a) 
    y = cl.MackChainladder().fit(b)
    assert x.process_risk_.iloc[:,:,:-1] == y.process_risk_

def test_multi_triangle_mack(clrd,atol):
    tri = clrd.loc['Agway Ins Co']['IncurLoss','CumPaidLoss']
    mack = cl.MackChainladder().fit(tri)
    for i in range(len(tri.index)):
        for j in range(len(tri.columns)):
            assert np.all(abs(mack.full_std_err_.iloc[i,j].values-cl.MackChainladder().fit(tri.iloc[i,j]).full_std_err_.values) < atol)