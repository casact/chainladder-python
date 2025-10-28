import chainladder as cl

raa = cl.load_sample("RAA")

def test_val_corr():
    assert raa.valuation_correlation(p_critical=0.5, total=True)

def test_dev_corr():
    assert raa.development_correlation(p_critical=0.5)
