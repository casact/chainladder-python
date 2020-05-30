import chainladder as cl
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
import copy

tri = cl.load_sample('clrd')
qtr = cl.load_sample('quarterly')

# Test Triangle slicing
def test_slice_by_boolean():
    assert tri[tri['LOB'] == 'ppauto'].loc['Wolverine Mut Ins Co']['CumPaidLoss'] == \
                            tri.loc['Wolverine Mut Ins Co'].loc['ppauto']['CumPaidLoss']


def test_slice_by_loc():
    assert tri.loc['Aegis Grp'].loc['comauto'].index.iloc[0, 0] == 'comauto'


def test_slice_origin():
    assert cl.load_sample('raa')[cl.load_sample('raa').origin>'1985'].shape == \
        (1, 1, 5, 10)


def test_slice_development():
    assert cl.load_sample('raa')[cl.load_sample('raa').development<72].shape == \
        (1, 1, 10, 5)


def test_slice_by_loc_iloc():
    assert tri.groupby('LOB').sum().loc['comauto'].index.iloc[0, 0] == 'comauto'


def test_repr():
    tri = cl.load_sample('raa')
    np.testing.assert_array_equal(pd.read_html(tri._repr_html_())[0].set_index('Origin').values,
                            tri.to_frame().values)


def test_arithmetic_union():
    raa = cl.load_sample('raa')
    assert raa.shape == (raa-raa[raa.valuation<'1987']).shape


def test_to_frame_unusual():
    a = cl.load_sample('clrd').groupby(['LOB']).sum().latest_diagonal['CumPaidLoss'].to_frame().values
    b = cl.load_sample('clrd').latest_diagonal['CumPaidLoss'].groupby(['LOB']).sum().to_frame().values
    xp = cp.get_array_module(a)
    xp.testing.assert_array_equal(a, b)


def test_link_ratio():
    tri = cl.load_sample('RAA')
    xp = cp.get_array_module(tri.values)
    xp.testing.assert_allclose(tri.link_ratio.values*tri.values[:,:,:-1,:-1],
                               tri.values[:,:,:-1,1:], atol=1e-5)


def test_incr_to_cum():
    xp = cp.get_array_module(tri.values)
    xp.testing.assert_array_equal(tri.cum_to_incr().incr_to_cum().values, tri.values)


def test_create_new_value():
    tri2 = copy.deepcopy(tri)
    tri2['lr'] = (tri2['CumPaidLoss']/tri2['EarnedPremDIR'])
    assert (tri.shape[0], tri.shape[1]+1, tri.shape[2], tri.shape[3]) == tri2.shape


def test_multilevel_index_groupby_sum1():
    assert tri.groupby('LOB').sum().sum() == tri.sum()


def test_multilevel_index_groupby_sum2():
    a = tri.groupby('GRNAME').sum().sum()
    b = tri.groupby('LOB').sum().sum()
    assert a == b


def test_boolean_groupby_eq_groupby_loc():
    xp = cp.get_array_module(tri.values)
    xp.testing.assert_array_equal(tri[tri['LOB']=='ppauto'].sum().values,
                        tri.groupby('LOB').sum().loc['ppauto'].values)


def test_latest_diagonal_two_routes():
    assert tri.latest_diagonal.sum()['BulkLoss'] == tri.sum().latest_diagonal['BulkLoss']


def test_sum_of_diff_eq_diff_of_sum():
    assert (tri['BulkLoss']-tri['CumPaidLoss']).latest_diagonal == \
           (tri.latest_diagonal['BulkLoss'] - tri.latest_diagonal['CumPaidLoss'])


def test_append():
    assert cl.load_sample('raa').append(cl.load_sample('raa')).sum() == 2*cl.load_sample('raa')


def test_assign_existing_col():
    tri = cl.load_sample('quarterly')
    before = tri.shape
    tri['paid'] = 1/tri['paid']
    assert tri.shape == before


def test_arithmetic_across_keys():
    x = cl.load_sample('auto')
    xp = cp.get_array_module(x.values)
    xp.testing.assert_array_equal((x.sum()-x.iloc[0]).values, x.iloc[1].values)

def test_grain():
    actual = qtr.iloc[0,0].grain('OYDY').values[0,0,:,:]
    xp = cp.get_array_module(actual)
    expected = xp.array([[  44.,  621.,  950., 1020., 1070., 1069., 1089., 1094., 1097.,
        1099., 1100., 1100.],
       [  42.,  541., 1052., 1169., 1238., 1249., 1266., 1269., 1296.,
        1300., 1300.,   xp.nan],
       [  17.,  530.,  966., 1064., 1100., 1128., 1155., 1196., 1201.,
        1200.,   xp.nan,   xp.nan],
       [  10.,  393.,  935., 1062., 1126., 1209., 1243., 1286., 1298.,
          xp.nan,   xp.nan,   xp.nan],
       [  13.,  481., 1021., 1267., 1400., 1476., 1550., 1583.,   xp.nan,
          xp.nan,   xp.nan,   xp.nan],
       [   2.,  380.,  788.,  953., 1001., 1030., 1066.,   xp.nan,   xp.nan,
          xp.nan,   xp.nan,   xp.nan],
       [   4.,  777., 1063., 1307., 1362., 1411.,   xp.nan,   xp.nan,   xp.nan,
          xp.nan,   xp.nan,   xp.nan],
       [   2.,  472., 1617., 1818., 1820.,   xp.nan,   xp.nan,   xp.nan,   xp.nan,
          xp.nan,   xp.nan,   xp.nan],
       [   3.,  597., 1092., 1221.,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,
          xp.nan,   xp.nan,   xp.nan],
       [   4.,  583., 1212.,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,
          xp.nan,   xp.nan,   xp.nan],
       [  21.,  422.,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,
          xp.nan,   xp.nan,   xp.nan],
       [  13.,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,   xp.nan,
          xp.nan,   xp.nan,   xp.nan]])
    xp.testing.assert_array_equal(actual, expected)

def test_off_cycle_val_date():
    assert cl.load_sample('quarterly').valuation_date.strftime('%Y-%m-%d') == '2006-03-31'

def test_printer():
    print(cl.load_sample('abc'))


def test_value_order():
    a = tri[['CumPaidLoss','BulkLoss']]
    b = tri[['BulkLoss', 'CumPaidLoss']]
    xp = cp.get_array_module(a.values)
    xp.testing.assert_array_equal(a.values[:,-1], b.values[:, 0])


def test_trend():
    assert abs((cl.load_sample('abc').trend(0.05).trend((1/1.05)-1) -
                cl.load_sample('abc')).sum().sum()) < 1e-5


def test_arithmetic_1():
    x = cl.load_sample('mortgage')
    np.testing.assert_array_equal(-(((x/x)+0)*x), -(+x))


def test_arithmetic_2():
    x = cl.load_sample('mortgage')
    np.testing.assert_array_equal(1-(x/x), 0*x*0)


def test_rtruediv():
    raa = cl.load_sample('raa')
    xp = cp.get_array_module(raa.values)
    assert xp.nansum(abs(((1/raa)*raa).values[0,0] - raa._nan_triangle()))< .00001


def test_shift():
    x = cl.load_sample('quarterly').iloc[0,0]
    xp = cp.get_array_module(x.values)
    xp.testing.assert_array_equal(x[x.valuation<=x.valuation_date].values, x.values)

def test_quantile_vs_median():
    clrd = cl.load_sample('clrd')
    xp = cp.get_array_module(clrd.values)
    xp.testing.assert_array_equal(clrd.quantile(.5)['CumPaidLoss'].values,
                            clrd.median()['CumPaidLoss'].values)


def test_grain_returns_valid_tri():
    tri = cl.load_sample('quarterly')
    assert tri.grain('OYDY').latest_diagonal == tri.latest_diagonal


def test_base_minimum_exposure_triangle():
    raa = (cl.load_sample('raa').latest_diagonal*0+50000).to_frame().reset_index()
    raa['index'] = raa['index'].astype(str)
    cl.Triangle(raa, origin='index',
                columns=list(cl.load_sample('raa').columns))


def test_origin_and_value_setters():
    raa = cl.load_sample('raa')
    raa2 = cl.load_sample('raa')
    raa.columns = list(raa.columns)
    raa.origin = list(raa.origin)
    assert np.all((np.all(raa2.origin == raa.origin),
                   np.all(raa2.development == raa.development),
                   np.all(raa2.odims == raa.odims),
                   np.all(raa2.vdims == raa.vdims)))

def test_grain_increm_arg():
    tri = cl.load_sample('quarterly')['incurred']
    tri_i = tri.cum_to_incr()
    np.testing.assert_array_equal(tri_i.grain('OYDY').incr_to_cum(),
                            tri.grain('OYDY'))


def test_valdev1():
    a = cl.load_sample('quarterly').dev_to_val().val_to_dev().values
    b = cl.load_sample('quarterly').values
    xp = cp.get_array_module(a)
    xp.testing.assert_array_equal(a,b)


def test_valdev2():
    a = cl.load_sample('quarterly').dev_to_val().grain('OYDY').val_to_dev().values
    b = cl.load_sample('quarterly').grain('OYDY').values
    xp = cp.get_array_module(a)
    xp.testing.assert_array_equal(a,b)


def test_valdev3():
    a = cl.load_sample('quarterly').grain('OYDY').dev_to_val().val_to_dev().values
    b = cl.load_sample('quarterly').grain('OYDY').values
    xp = cp.get_array_module(a)
    xp.testing.assert_array_equal(a,b)


#def test_valdev4():
#    # Does not work with pandas 0.23, consider requiring only pandas>=0.24
#    raa = cl.load_sample('raa')
#    np.testing.assert_array_equal(raa.dev_to_val()[raa.dev_to_val().development>='1989'].values,
#        raa[raa.valuation>='1989'].dev_to_val().values)


def test_valdev5():
    raa = cl.load_sample('raa')
    xp = cp.get_array_module(raa.values)
    xp.testing.assert_array_equal(raa[raa.valuation>='1989'].latest_diagonal.values,
                            raa.latest_diagonal.values)

def test_valdev6():
    raa = cl.load_sample('raa')
    xp = cp.get_array_module(raa.values)
    xp.testing.assert_array_equal(raa.grain('OYDY').latest_diagonal.values,
                            raa.latest_diagonal.grain('OYDY').values)

def test_valdev7():
    tri = cl.load_sample('quarterly')
    xp = cp.get_array_module(tri.values)
    x = cl.Chainladder().fit(tri).full_expectation_
    xp.testing.assert_array_equal(x.dev_to_val().val_to_dev().values, x.values)

def test_reassignment():
    raa = cl.load_sample('clrd')
    raa['values'] = raa['CumPaidLoss']
    raa['values'] = raa['values'] + raa['CumPaidLoss']

def test_dropna():
    clrd = cl.load_sample('clrd')
    assert clrd.shape == clrd.dropna().shape
    assert clrd[clrd['LOB']=='wkcomp'].iloc[-5]['CumPaidLoss'].dropna().shape == (1,1,2,2)

def test_commutative():
    tri = cl.load_sample('quarterly')
    xp = cp.get_array_module(tri.values)
    full = cl.Chainladder().fit(tri).full_expectation_
    assert tri.grain('OYDY').val_to_dev() == tri.val_to_dev().grain('OYDY')
    assert tri.cum_to_incr().grain('OYDY').val_to_dev() == tri.val_to_dev().cum_to_incr().grain('OYDY')
    assert tri.grain('OYDY').cum_to_incr().val_to_dev().incr_to_cum() == tri.val_to_dev().grain('OYDY')
    assert full.grain('OYDY').val_to_dev() == full.val_to_dev().grain('OYDY')
    assert full.cum_to_incr().grain('OYDY').val_to_dev() == full.val_to_dev().cum_to_incr().grain('OYDY')
    assert xp.allclose(xp.nan_to_num(full.grain('OYDY').cum_to_incr().val_to_dev().incr_to_cum().values),
            xp.nan_to_num(full.val_to_dev().grain('OYDY').values), atol=1e-5)

def test_broadcasting():
    t1 = cl.load_sample('raa')
    t2 = tri
    assert t1.broadcast_axis('columns', t2.columns).shape[1] == t2.shape[1]
    assert t1.broadcast_axis('index', t2.index).shape[0] == t2.shape[0]

def test_slicers_honor_order():
    clrd = cl.load_sample('clrd').groupby('LOB').sum()
    assert clrd.iloc[[1,0], :].iloc[0, 1] == clrd.iloc[1, 1] #row
    assert clrd.iloc[[1,0], [1, 0]].iloc[0, 0] == clrd.iloc[1, 1] #col
    assert clrd.loc[:,['CumPaidLoss','IncurLoss']].iloc[0, 0] == clrd.iloc[0,1]
    assert clrd.loc[['ppauto', 'medmal'],['CumPaidLoss','IncurLoss']].iloc[0,0] == clrd.iloc[3]['CumPaidLoss']
    assert clrd.loc[clrd['LOB']=='comauto', ['CumPaidLoss', 'IncurLoss']] == clrd[clrd['LOB']=='comauto'].iloc[:, [1,0]]

def test_exposure_tri():
    x = cl.load_sample('auto')
    x= x[x.development==12]
    x = x['paid'].to_frame().T.unstack().reset_index()
    x.columns=['LOB', 'origin', 'paid']
    x.origin = x.origin.astype(str)
    y = cl.Triangle(x, origin='origin', index='LOB', columns='paid')
    x = cl.load_sample('auto')['paid']
    x = x[x.development==12]
    assert x == y

def test_jagged_1_add():
    raa = cl.load_sample('raa')
    raa1 = raa[raa.origin<='1984']
    raa2 = raa[raa.origin>'1984']
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa

def test_jagged_2_add():
    raa = cl.load_sample('raa')
    raa1 = raa[raa.development<=48]
    raa2 = raa[raa.development>48]
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa

def test_subtriangle_slice():
    triangle = cl.load_sample('clrd').groupby('LOB').sum()[['CumPaidLoss', 'IncurLoss']]
    dev = cl.Development(average='simple').fit_transform(triangle)
    tail = cl.TailCurve().fit_transform(dev)

    # Test dataframe commutive
    assert tail.iloc[1].tail_ == tail.tail_.iloc[1]
    assert tail.loc['comauto'].tail_ == tail.tail_.loc['comauto']
    assert tail.loc['comauto', 'CumPaidLoss'].tail_ == tail.tail_.loc['comauto', 'CumPaidLoss']
    assert tail[['IncurLoss', 'CumPaidLoss']].tail_ == tail.tail_[['IncurLoss', 'CumPaidLoss']]
    assert tail.iloc[:3, 0].tail_ == tail.tail_.iloc[:3,0]
    # Test triangle cummutative
    assert tail.iloc[1].cdf_ == tail.cdf_.iloc[1]
    assert tail.loc['comauto'].cdf_ == tail.cdf_.loc['comauto']
    assert tail.loc['comauto', 'CumPaidLoss'].cdf_ == tail.cdf_.loc['comauto', 'CumPaidLoss']
    assert tail[['IncurLoss', 'CumPaidLoss']].cdf_ == tail.cdf_[['IncurLoss', 'CumPaidLoss']]
    assert tail.iloc[:3, 0].cdf_ == tail.cdf_.iloc[:3,0]
