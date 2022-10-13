import chainladder as cl


def test_mcl_ult():
    mcl = cl.load_sample("mcl")
    dev = cl.Development().fit_transform(mcl)
    cl_traditional = cl.Chainladder().fit(dev).ultimate_
    dev_munich = cl.MunichAdjustment(
        paid_to_incurred=[("paid", "incurred")]
    ).fit_transform(dev)
    cl_munich = cl.Chainladder().fit(dev_munich).ultimate_


def test_mcl_rollforward():
    mcl = cl.load_sample("mcl")
    mcl_prior = mcl[mcl.valuation < mcl.valuation_date]
    munich = cl.MunichAdjustment(paid_to_incurred=[("paid", "incurred")]).fit(mcl_prior)
    new = munich.transform(mcl)
    cl.Chainladder().fit(new).ultimate_
