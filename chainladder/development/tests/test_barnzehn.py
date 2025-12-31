import numpy as np
import chainladder as cl
import pytest
abc = cl.load_sample('abc')

def test_basic_bz():
    assert np.all(
        np.around(cl.BarnettZehnwirth(formula='C(origin)+C(development)').fit(abc).coef_.T.values,3).flatten()
        == np.array([11.837,0.179,0.345,0.378,0.405,0.427,0.431,0.66,0.963,1.157,1.278,0.251,-0.056,-0.449,-0.829,-1.169,-1.508,-1.798,-2.023,-2.238,-2.428])
    )

def test_multiple_triangle_exception():
    d = cl.load_sample("usauto")
    with pytest.raises(ValueError):
        cl.BarnettZehnwirth(formula='C(origin)+C(development)').fit(d)

def test_drops():
    '''
    this function tests the passing in a basic drop_valuation
    '''
    assert np.all(
        np.around(cl.BarnettZehnwirth(formula='C(development)',drop_valuation='1979').fit(abc).triangle_ml_.values,3)
        == np.around(cl.BarnettZehnwirth(formula='C(development)',drop = [('1977',36),('1978',24),('1979',12)]).fit(abc).triangle_ml_.values,3)
    )

def test_bz_2008():
    '''
    this function tests the drop parameter by recreating the example in the 2008 BZ paper, section 4.1
    '''
    exposure=np.array([[2.2], [2.4], [2.2], [2.0], [1.9], [1.6], [1.6], [1.8], [2.2], [2.5], [2.6]])
    abc_adj = abc/exposure

    origin_buckets = [2,3,5]
    dev_buckets = [(24,36),(36,48),(48,84),(84,108),(108,9999)]
    val_buckets = [(1,8),(8,9),(9,999)]

    origin_formula = '+'.join([f'I(origin >= {x})' for x in origin_buckets])
    dev_formula = '+'.join([f'I((np.minimum({x[1]-12},development) - np.minimum({x[0]-12},development))/12)' for x in dev_buckets])
    val_formula = '+'.join([f'I(np.minimum({x[1]-1},valuation) - np.minimum({x[0]-1},valuation))' for x in val_buckets])
    model=cl.BarnettZehnwirth(formula=origin_formula + '+' + dev_formula + '+' + val_formula, drop=('1982',72)).fit(abc_adj)
    assert np.all(
        np.around(model.coef_.values,4).flatten()
        == np.array([11.1579,0.1989,0.0703,0.0919,0.1871,-0.3771,-0.4465,-0.3727,-0.3154,0.0432,0.0858,0.1464])
    )