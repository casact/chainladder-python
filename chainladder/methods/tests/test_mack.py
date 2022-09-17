import chainladder as cl

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