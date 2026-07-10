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

def test_mack1997_hardcode():
    """
    Reconciles key MackChainladder statistics to values provided in the paper
    """
    #sourced from Table 1, p365 of Mack(1997)
    ldf_se = [2.24,.517,.122,.051,.042,.023,.015,.012]
    sigma = [1337,988.5,440.1,207.0,164.2,74.6,35.49,16.89]
    #sourced from Table 2, p366 of Mack(1997)
    ibnr_se = [0,61,140,319,596,1038,1298,1806,2182]

    tri = cl.load_sample("mortgage")
    dev = cl.Development(sigma_interpolation = 'mack').fit_transform(tri)
    model = cl.MackChainladder().fit(dev)
    ldf_rhs = dev.std_err_.values.flatten()
    assert np.allclose(ldf_se[0],ldf_rhs[0],atol=0.01)
    assert np.allclose(ldf_se[1:],ldf_rhs[1:],atol=0.001)
    sigma_rhs = dev.sigma_.values.flatten()
    assert np.allclose(sigma[0],sigma_rhs[0],atol=1)
    assert np.allclose(sigma[1:],sigma_rhs[1:],atol=0.1)
    ibnr_rhs = model.summary_.values[0,0,:,-1]/1000
    assert np.allclose(ibnr_se,np.nan_to_num(ibnr_rhs,nan=0),atol=1,equal_nan=True)

def test_mack1994_hardcode(raa):
    """
    Reconciles key MackChainladder statistics to values provided in the paper
    """
    #sourced from top table on p130 of Mack(1994)
    sigma_sq = [27883,1109,691,61.2,119,40.8,1.34,7.88]
    #sourced from bottom table on p130 of Mack(1994)
    ibnr_se = [206,623,747,1469,2002,2209,5358,6333,24566]

    dev = cl.Development(sigma_interpolation = 'mack').fit_transform(raa)
    model = cl.MackChainladder().fit(dev)
    sigma_rhs = dev.sigma_.values.flatten() ** 2
    print(sigma_rhs)
    assert np.allclose(sigma_sq[:3],sigma_rhs[:3],atol=1)
    assert np.allclose(sigma_sq[3:4],sigma_rhs[3:4],atol=0.1)
    assert np.allclose(sigma_sq[4:5],sigma_rhs[4:5],atol=1)
    assert np.allclose(sigma_sq[5:6],sigma_rhs[5:6],atol=0.1)
    assert np.allclose(sigma_sq[6:8],sigma_rhs[6:8],atol=0.01)
    ibnr_rhs = model.summary_.values[0,0,:,-1]
    assert np.allclose(ibnr_se,ibnr_rhs[1:],atol=1)