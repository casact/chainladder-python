import chainladder as cl

def test_basic_odp_cl(genins):
    assert abs(
        (cl.Chainladder().fit(genins).ultimate_ -
         cl.Chainladder().fit(cl.TweedieGLM().fit_transform(genins)).ultimate_) /
        genins.latest_diagonal).max()< 1e-2
    
def test_sample_weight(genins):
    assert abs(
        (cl.Chainladder().fit(genins).ultimate_ -
         cl.Chainladder().fit(cl.TweedieGLM().fit_transform(genins,sample_weight=genins/genins)).ultimate_) /
        genins.latest_diagonal).max()< 1e-2