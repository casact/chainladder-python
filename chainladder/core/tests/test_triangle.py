import chainladder as cl
import numpy as np
import copy

tri = cl.load_dataset('casresearch')
qtr = cl.load_dataset('quarterly')

# Test Triangle slicing
def test_slice_by_boolean():
    assert tri[tri['LOB'] == 'ppauto'].loc['Wolverine Mut Ins Co']['CumPaidLoss'] == \
                            tri.loc['Wolverine Mut Ins Co'].loc['ppauto']['CumPaidLoss']


def test_slice_by_loc():
    assert tri.loc['Aegis Grp'].loc['comauto'].keys.iloc[0, 0] == 'comauto'


def test_slice_by_loc_iloc():
    assert tri.groupby('LOB').sum().loc['comauto'].keys.iloc[0, 0] == 'comauto'


def test_create_new_value():
    tri2 = copy.deepcopy(tri)
    tri2['lr'] = (tri2['CumPaidLoss']/tri2['EarnedPremDIR'])
    assert (tri.shape[0], tri.shape[1]+1, tri.shape[2], tri.shape[3]) == tri2.shape


def test_multilevel_index_groupby_sum1():
    assert tri.groupby('LOB').sum().sum() == tri.sum()


def test_multilevel_index_groupby_sum2():
    assert tri.groupby('GRNAME').sum().sum() == tri.groupby('LOB').sum().sum()


def test_boolean_groupby_eq_groupby_loc():
    assert tri[tri['LOB']=='ppauto'].sum() == tri.groupby('LOB').sum().loc['ppauto']


def test_latest_diagonal_two_routes():
    assert tri.latest_diagonal.sum()['BulkLoss'] == tri.sum().latest_diagonal['BulkLoss']


def test_sum_of_diff_eq_diff_of_sum():
    assert (tri['BulkLoss']-tri['CumPaidLoss']).latest_diagonal == \
           (tri.latest_diagonal['BulkLoss'] - tri.latest_diagonal['CumPaidLoss'])


def test_grain():
    actual = qtr.iloc[0,0].grain('OYDY').triangle[0,0,:,:]
    expected = np.array([[  44.,  621.,  950., 1020., 1070., 1069., 1089., 1094., 1097.,
        1099., 1100., 1100.],
       [  42.,  541., 1052., 1169., 1238., 1249., 1266., 1269., 1296.,
        1300., 1300.,   np.nan],
       [  17.,  530.,  966., 1064., 1100., 1128., 1155., 1196., 1201.,
        1200.,   np.nan,   np.nan],
       [  10.,  393.,  935., 1062., 1126., 1209., 1243., 1286., 1298.,
          np.nan,   np.nan,   np.nan],
       [  13.,  481., 1021., 1267., 1400., 1476., 1550., 1583.,   np.nan,
          np.nan,   np.nan,   np.nan],
       [   2.,  380.,  788.,  953., 1001., 1030., 1066.,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan],
       [   4.,  777., 1063., 1307., 1362., 1411.,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan],
       [   2.,  472., 1617., 1818., 1820.,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan],
       [   3.,  597., 1092., 1221.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan],
       [   4.,  583., 1212.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan],
       [  21.,  422.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan],
       [  13.,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,
          np.nan,   np.nan,   np.nan]])
    np.testing.assert_equal(actual, expected)
