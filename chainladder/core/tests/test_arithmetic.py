import pytest


def test_arithmetic_union(raa):
    assert raa.shape == (raa - raa[raa.valuation < "1987"]).shape
    assert raa[raa.valuation<'1986'] + raa[raa.valuation>='1986'] == raa


def test_arithmetic_across_keys(qtr):
    assert (qtr.sum(1) - qtr.iloc[:, 0]) == qtr.iloc[:, 1]


def test_arithmetic_1(raa):
    x = raa
    assert -(((x / x) + 0) * x) == -(+x)
    assert 1 - (x / x) ==  0 * x * 0


def test_rtruediv(raa):
    xp = raa.get_array_module()
    assert xp.nansum(abs(((1 / raa) * raa).values[0, 0] - raa.nan_triangle)) < 0.00001


def test_vector_division(raa):
    raa.latest_diagonal / raa


def test_multiindex_broadcast(clrd):
    clrd = clrd["CumPaidLoss"]
    clrd / clrd.groupby("LOB").sum()


def test_index_broadcasting(clrd):
    """ Basic broadcasting where b is a subset of a """
    assert ((clrd / clrd.sum()) - ((1 / clrd.sum()) * clrd)).sum().sum().sum() < 1e-4

def test_index_broadcasting2(clrd):
    """ b.key_labels are a subset of a.key_labels and b is missing some elements """
    a = clrd['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum().iloc[:-1]
    c = a + b
    assert (a.index == c.index).all().all()


def test_index_broadcasting3(clrd):
    """ b.key_labels are a subset of a.key_labels and a is missing some elements """
    a = clrd[~clrd['LOB'].isin(['wkcomp', 'medmal'])]['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum()
    c = a + b
    assert (a.index == c[~c['LOB'].isin(['wkcomp', 'medmal'])].index).all().all()
    assert len(c) - len(a) == 2


def test_index_broadcasting4(clrd):
    """ b should broadcast to a if b only has one index element """
    a = clrd['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum().iloc[0]
    c = a + b
    assert (a.index == c.index).all().all()


@pytest.mark.xfail
def test_index_broadacsting4(clrd):
    """ If one triangle has key_labels that are not a subset of the other, then fail """
    a = clrd['CumPaidLoss']
    b = clrd['CumPaidLoss'].groupby('LOB').sum()
    idx = b.index
    idx['New Field'] = 'New'
    b.index = idx
    c = a + b

def test_index_broadcasting5(clrd):
    """ If a and b have shared key labels but no matching levels, then they will stack """
    a = clrd['CumPaidLoss'].iloc[:300]
    b = clrd['CumPaidLoss'].iloc[300:]
    c = a + b
    assert c.sort_index() == clrd['CumPaidLoss'].sort_index()


def test_index_broadacsting6(clrd):
    a = clrd['CumPaidLoss'].iloc[:100]
    b = clrd['CumPaidLoss'].iloc[50:150]
    c = clrd['CumPaidLoss'].iloc[50:100]
    d = a + b - c
    assert d.sort_index() == clrd['CumPaidLoss'].iloc[:150].sort_index()
