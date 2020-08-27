import chainladder as cl
import pandas as pd
import numpy as np
import copy

tri = cl.load_sample('clrd')
qtr = cl.load_sample('quarterly')
raa = cl.load_sample('raa')

# Test Triangle slicing
def test_slice_by_boolean():
    assert tri[tri['LOB'] == 'ppauto'].loc['Wolverine Mut Ins Co']['CumPaidLoss'] == \
           tri.loc['Wolverine Mut Ins Co'].loc['ppauto']['CumPaidLoss']


def test_slice_by_loc():
    assert tri.loc['Aegis Grp'].loc['comauto'].index.iloc[0, 0] == 'comauto'


def test_slice_origin():
    assert raa[raa.origin>'1985'].shape == (1, 1, 5, 10)


def test_slice_development():
    assert raa[raa.development<72].shape == (1, 1, 10, 5)


def test_slice_by_loc_iloc():
    assert tri.groupby('LOB').sum().loc['comauto'].index.iloc[0, 0] == 'comauto'


def test_repr():
    np.testing.assert_array_equal(
        pd.read_html(raa._repr_html_())[0].set_index('Origin').values,
        raa.to_frame().values)


def test_arithmetic_union():
    assert raa.shape == (raa-raa[raa.valuation<'1987']).shape


def test_to_frame_unusual():
    a = tri.groupby(['LOB']).sum().latest_diagonal['CumPaidLoss'].to_frame().values
    b = tri.latest_diagonal['CumPaidLoss'].groupby(['LOB']).sum().to_frame().values
    np.testing.assert_array_equal(a, b)


def test_link_ratio():
    xp = raa.get_array_module()
    assert xp.sum(xp.nan_to_num(raa.link_ratio.values*raa.values[:,:,:-1,:-1]) -
                  xp.nan_to_num(raa.values[:,:,:-1,1:]))<1e-5


def test_incr_to_cum():
    xp = tri.get_array_module()
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
    xp = tri.get_array_module()
    xp.testing.assert_array_equal(tri[tri['LOB']=='ppauto'].sum().values,
                        tri.groupby('LOB').sum().loc['ppauto'].values)


def test_latest_diagonal_two_routes():
    assert tri.latest_diagonal.sum()['BulkLoss'] == tri.sum().latest_diagonal['BulkLoss']


def test_sum_of_diff_eq_diff_of_sum():
    assert (tri['BulkLoss']-tri['CumPaidLoss']).latest_diagonal == \
           (tri.latest_diagonal['BulkLoss'] - tri.latest_diagonal['CumPaidLoss'])


def test_append():
    assert raa.append(raa).sum() == 2*raa


def test_assign_existing_col():
    tri = cl.load_sample('quarterly')
    before = tri.shape
    tri['paid'] = 1/tri['paid']
    assert tri.shape == before


def test_arithmetic_across_keys():
    x = cl.load_sample('auto')
    xp = x.get_array_module()
    xp.testing.assert_array_equal((x.sum()-x.iloc[0]).values, x.iloc[1].values)

def test_grain():
    actual = qtr.iloc[0,0].grain('OYDY').values[0,0,:,:]
    xp = qtr.get_array_module()
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
    xp = a.get_array_module()
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
    xp = raa.get_array_module()
    assert xp.nansum(abs(((1/raa)*raa).values[0,0] - raa.nan_triangle))< .00001


def test_shift():
    x = cl.load_sample('quarterly').iloc[0,0]
    xp = x.get_array_module()
    xp.testing.assert_array_equal(x[x.valuation<=x.valuation_date].values, x.values)

def test_quantile_vs_median():
    clrd = cl.load_sample('clrd')
    xp = clrd.get_array_module()
    xp.testing.assert_array_equal(clrd.quantile(.5)['CumPaidLoss'].values,
                            clrd.median()['CumPaidLoss'].values)


def test_grain_returns_valid_tri():
    tri = cl.load_sample('quarterly')
    assert tri.grain('OYDY').latest_diagonal == tri.latest_diagonal


def test_base_minimum_exposure_triangle():
    d = (raa.latest_diagonal*0+50000).to_frame().reset_index()
    d['index'] = d['index'].astype(str)
    cl.Triangle(d, origin='index', columns=d.columns[-1])


def test_origin_and_value_setters():
    raa2 = raa.copy()
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
    a = cl.load_sample('quarterly').dev_to_val().val_to_dev()
    b = cl.load_sample('quarterly')
    xp = a.get_array_module()
    xp.testing.assert_array_equal(a.values, b.values)


def test_valdev2():
    a = cl.load_sample('quarterly').dev_to_val().grain('OYDY').val_to_dev()
    b = cl.load_sample('quarterly').grain('OYDY')
    xp = a.get_array_module()
    xp.testing.assert_array_equal(a.values, b.values)


def test_valdev3():
    a = cl.load_sample('quarterly').grain('OYDY').dev_to_val().val_to_dev()
    b = cl.load_sample('quarterly').grain('OYDY')
    xp = a.get_array_module()
    xp.testing.assert_array_equal(a.values ,b.values)


#def test_valdev4():
#    # Does not work with pandas 0.23, consider requiring only pandas>=0.24
#    raa = raa
#    np.testing.assert_array_equal(raa.dev_to_val()[raa.dev_to_val().development>='1989'].values,
#        raa[raa.valuation>='1989'].dev_to_val().values)


def test_valdev5():
    xp = raa.get_array_module()
    xp.testing.assert_array_equal(raa[raa.valuation>='1989'].latest_diagonal.values,
                            raa.latest_diagonal.values)

def test_valdev6():
    xp = raa.get_array_module()
    xp.testing.assert_array_equal(raa.grain('OYDY').latest_diagonal.values,
                            raa.latest_diagonal.grain('OYDY').values)

def test_valdev7():
    tri = cl.load_sample('quarterly')
    xp = tri.get_array_module()
    x = cl.Chainladder().fit(tri).full_expectation_
    assert xp.sum(x.dev_to_val().val_to_dev().values-x.values) < 1e-5

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
    xp = tri.get_array_module()
    full = cl.Chainladder().fit(tri).full_expectation_
    assert tri.grain('OYDY').val_to_dev() == tri.val_to_dev().grain('OYDY')
    assert tri.cum_to_incr().grain('OYDY').val_to_dev() == tri.val_to_dev().cum_to_incr().grain('OYDY')
    assert tri.grain('OYDY').cum_to_incr().val_to_dev().incr_to_cum() == tri.val_to_dev().grain('OYDY')
    assert full.grain('OYDY').val_to_dev() == full.val_to_dev().grain('OYDY')
    assert full.cum_to_incr().grain('OYDY').val_to_dev() == full.val_to_dev().cum_to_incr().grain('OYDY')
    assert xp.allclose(xp.nan_to_num(full.grain('OYDY').cum_to_incr().val_to_dev().incr_to_cum().values),
            xp.nan_to_num(full.val_to_dev().grain('OYDY').values), atol=1e-5)

def test_broadcasting():
    t1 = raa
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
    raa1 = raa[raa.origin<='1984']
    raa2 = raa[raa.origin>'1984']
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa

def test_jagged_2_add():
    raa1 = raa[raa.development<=48]
    raa2 = raa[raa.development>48]
    assert raa2 + raa1 == raa
    assert raa2.dropna() + raa1.dropna() == raa

def test_df_period_input():
    d = raa.latest_diagonal
    df = d.to_frame().reset_index()
    assert cl.Triangle(df, origin='index', columns=df.columns[-1]) == d

def test_trend_on_vector():
    d = raa.latest_diagonal
    assert d.trend(.05, axis=2).to_frame().astype(int).iloc[0,0]==29216

def test_latest_diagonal_val_to_dev():
    assert raa.latest_diagonal.val_to_dev()==raa[raa.valuation==raa.valuation_date]

def test_vector_division():
    raa.latest_diagonal/raa

def test_sumdiff_to_diffsum():
    tri = cl.load_sample('clrd')['CumPaidLoss']
    assert tri.cum_to_incr().incr_to_cum().sum() == tri.sum()

def test_multiindex_broadcast():
    clrd = cl.load_sample('clrd')['CumPaidLoss']
    clrd / clrd.groupby('LOB').sum()

def test_backends():
    clrd = tri[['CumPaidLoss', 'EarnedPremDIR']]
    a = clrd.iloc[1,0].set_backend('sparse').dropna()
    b = clrd.iloc[1,0].dropna()
    assert a == b

def test_union_columns():
    assert tri.iloc[:, :3]+tri.iloc[:, 3:] == tri

def test_4loc():
    clrd = tri.groupby('LOB').sum()
    assert clrd.iloc[:3, :2, 0,0] == clrd[clrd.origin==tri.origin.min()][clrd.development==clrd.development.min()].loc['comauto':'othliab', :'CumPaidLoss', :, :]
    assert clrd.iloc[:3, :2, 0:1, -1] == clrd[clrd.development==tri.development.max()].loc['comauto':'othliab', :'CumPaidLoss', '1988', :]

def test_init_vector():
    a = raa[raa.development==12]
    b = pd.DataFrame(
        {'AccYear':[item for item in range(1981, 1991)],
         'premium': [3000000]*10})
    b = cl.Triangle(b, origin='AccYear', columns='premium')
    assert np.all(a.valuation==b.valuation)
    assert a.valuation_date == b.valuation_date
