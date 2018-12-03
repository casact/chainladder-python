import numpy as np
def test1(mack_r_simple, mack_p_simple):
    assert np.all(np.array(mack_r_simple['RAA'].rx('sigma'))
                  == mack_p_simple['RAA'].sigma_.triangle[0,0,0,:])
