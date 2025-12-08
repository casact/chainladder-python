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

def test_feat_eng_1():
    '''
    this function tests the passing in a basic engineered feature. Since test_func just returns development, C(development) and C(teatfeat) should yield identical results
    '''
    def test_func(df):
        return df["development"]

    abc = cl.load_sample('abc')
    test_dict = {'testfeat':{'func':test_func,'kwargs':{}}}

    assert np.all(
        np.around(cl.BarnettZehnwirth(formula='C(origin)+C(development)').fit(abc).coef_.T.values,3)
        == np.around(cl.BarnettZehnwirth(formula='C(origin)+C(testfeat)',feat_eng = test_dict).fit(abc).coef_.T.values,3)
    )

def test_feat_eng_2():
    '''
    this function tests more complex feature engineering. Since origin_onehot just replicates the one-hot encoding that's performed inside sklearn LinearRegression, the two BZ models should yield identical results

    this function also tests the BZ transformer
    '''
    def origin_onehot(df,ori):
        return [1 if x == ori else 0 for x in df["origin"]]

    abc = cl.load_sample('abc')
    feat_dict = {f'origin_{x}':{'func':origin_onehot,'kwargs':{'ori':float(x+1)}} for x in range(10)}

    assert np.all(
        np.around(cl.BarnettZehnwirth(formula='+'.join([f'C({x})' for x in feat_dict.keys()]),feat_eng = feat_dict).fit_transform(abc).ldf_.values,3)
        == np.around(cl.BarnettZehnwirth(formula='C(origin)').fit_transform(abc).ldf_.values,3)
    )