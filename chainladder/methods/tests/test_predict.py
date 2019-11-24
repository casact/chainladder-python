import chainladder as cl
raa = cl.load_dataset('RAA')
raa_1989 = raa[raa.valuation < raa.valuation_date]
cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
apriori = cl_ult*0+(cl_ult.sum()/10)  # Mean Chainladder Ultimate
apriori_1989 = apriori[apriori.origin < '1990']


def test_cc_predict():
    cc = cl.CapeCod().fit(raa_1989, sample_weight=apriori_1989)
    cc.predict(raa, sample_weight=apriori)


def test_bf_predict():
    cc = cl.BornhuetterFerguson().fit(raa_1989, sample_weight=apriori_1989)
    cc.predict(raa, sample_weight=apriori)


def test_mack_predict():
    mack = cl.MackChainladder().fit(raa_1989)
    mack.predict(raa)
