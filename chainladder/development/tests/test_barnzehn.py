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
        np.around(cl.BarnettZehnwirth(formula='C(origin)+development+valuation').fit(abc).coef_.T.values,3)
        == np.around(cl.BarnettZehnwirth(formula='C(origin)+testfeat+valuation',feat_eng = test_dict).fit(abc).coef_.T.values,3)
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
        np.around(cl.BarnettZehnwirth(formula='+'.join([f'C({x})' for x in feat_dict.keys()]),feat_eng = feat_dict).fit(abc).ldf_.values,3)
        == np.around(cl.BarnettZehnwirth(formula='C(origin)').fit_transform(abc).ldf_.values,3)
    )

def test_drops():
    '''
    this function tests the passing in a basic drop_valuation
    '''
    def test_func(df):
        return df["development"]

    abc = cl.load_sample('abc')
    test_dict = {'testfeat':{'func':test_func,'kwargs':{}}}

    assert np.all(
        np.around(cl.BarnettZehnwirth(formula='C(development)',drop_valuation='1979').fit(abc).coef_.T.values,3)
        == np.around(cl.BarnettZehnwirth(formula='C(testfeat)',drop = [('1977',36),('1978',24),('1979',12)],feat_eng = test_dict).fit(abc).coef_.T.values,3)
    )

def test_bz_2008():
    '''
    this function tests the drop parameter by recreating the example in the 2008 BZ paper, section 4.1
    '''
    abc = cl.load_sample('abc')
    exposure=np.array([[2.2], [2.4], [2.2], [2.0], [1.9], [1.6], [1.6], [1.8], [2.2], [2.5], [2.6]])
    abc_adj = abc/exposure

    def predictor_bins(df,pbin,axis):
        return [int(x >= min(pbin)) for x in df[axis]]
        
    origin_groups = {f'origin_{ori}'.replace('[','').replace(']','').replace(', ',''):{'func':predictor_bins,'kwargs':{'pbin':ori,'axis':'origin'}} for ori in [[2],[3,4],[5,6,7,8,9,10]]}

    def trend_piece(df,piece,axis):
        pmax = float(max(piece))
        increment=min(df[axis][df[axis]>0])
        pfirst = piece[0]-increment
        return [(x-pfirst)/increment if x in piece else (0 if x<pmax else (pmax-pfirst)/increment) for x in df[axis]]
        
    development_groups = {f'development_{dev}'.replace('[','').replace(']','').replace(', ',''):{'func':trend_piece,'kwargs':{'piece':dev,'axis':'development'}} for dev in [[24],[36],[48,60,72],[84,96],[108,120,132]]}

    valuation_groups = {f'valuation_{val}'.replace('[','').replace(']','').replace(', ',''):{'func':trend_piece,'kwargs':{'piece':val,'axis':'valuation'}} for val in [[1,2,3,4,5,6,7],[8],[9,10]]}

    abc_dict = {**origin_groups,**development_groups,**valuation_groups}
    model=cl.BarnettZehnwirth(formula='+'.join([z for z in abc_dict.keys()]),feat_eng=abc_dict, drop=('1982',72)).fit(abc_adj)
    assert np.all(
        np.around(model.coef_.values,4).flatten()
        == np.array([11.1579,0.1989,0.0703,0.0919,0.1871,-0.3771,-0.4465,-0.3727,-0.3154,0.0432,0.0858,0.1464])
    )