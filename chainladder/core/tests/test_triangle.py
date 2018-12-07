import chainladder as cl
import copy

tri = cl.load_dataset('casresearch')


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
