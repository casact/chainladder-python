import chainladder as cl
import copy

tri = cl.load_dataset('casresearch')


# Test Triangle slicing
def test1():
    assert tri[tri['LOB'] == 'ppauto'].loc['Wolverine Mut Ins Co']['CumPaidLoss'] == \
                            tri.loc['Wolverine Mut Ins Co'].loc['ppauto']['CumPaidLoss']


def test2():
    assert tri.loc['Aegis Grp'].loc['comauto'].keys.iloc[0,0] == 'comauto'


def test3():
    assert tri.groupby('LOB').sum().loc['comauto'].keys.iloc[0,0] == 'comauto'


def test4():
    tri2 = copy.deepcopy(tri)
    tri2['lr'] = (tri2['CumPaidLoss']/tri2['EarnedPremDIR'])
    assert (tri.shape[0],tri.shape[1]+1,tri.shape[2],tri.shape[3]) == tri2.shape


def test5():
    assert tri.groupby('LOB').sum().sum() == tri.sum()


def test6():
    assert tri.groupby('GRNAME').sum().sum() == tri.groupby('LOB').sum().sum()


def test7():
    assert tri[tri['LOB']=='ppauto'].sum() == tri.groupby('LOB').sum().loc['ppauto']


def test8():
    assert tri.latest_diagonal.sum()['BulkLoss'] == tri.sum().latest_diagonal['BulkLoss']


def test9():
    assert (tri['BulkLoss']-tri['CumPaidLoss']).latest_diagonal == \
           (tri.latest_diagonal['BulkLoss'] - tri.latest_diagonal['CumPaidLoss'])


def test10():
    q = cl.load_dataset('quarterly')
    assert cl.Development(avg_type='volume').fit_transform(q) == \
        cl.Development(avg_type='volume', by=-1).fit_transform(q)
