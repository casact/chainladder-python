import chainladder as cl

def test_basic_case_outstanding():
    tri = cl.load_sample('usauto')
    m = cl.CaseOutstanding(paid_to_incurred=('paid', 'incurred')).fit(tri)
    out = cl.Chainladder().fit(m.fit_transform(tri))
    a = (out.full_triangle_['incurred']-out.full_triangle_['paid']).iloc[..., -1, :9]*m.paid_ldf_.values
    b = (out.full_triangle_['paid'].cum_to_incr().iloc[..., -1, 1:10]).values
    assert (a-b).max() < 1e-6
