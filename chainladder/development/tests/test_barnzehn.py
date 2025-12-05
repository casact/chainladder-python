import numpy as np
import chainladder as cl
import pytest

def test_basic_bz():
    abc = cl.load_sample('abc')
    assert np.all(
        np.around(cl.BarnettZehnwirth(formula='C(origin)+C(development)').fit(abc).coef_.T.values,3).flatten()
        == np.array([11.837,0.179,0.345,0.378,0.405,0.427,0.431,0.66,0.963,1.157,1.278,0.251,-0.056,-0.449,-0.829,-1.169,-1.508,-1.798,-2.023,-2.238,-2.428])
    )

def test_multiple_triangle_exception():
    d = cl.load_sample("usauto")
    with pytest.raises(ValueError):
        cl.BarnettZehnwirth(formula='C(origin)+C(development)').fit(d)