import chainladder as cl
import numpy as np

def test_preserve_diagonal():
    triangle = cl.load_sample('berqsherm').loc['Auto']
    berq = cl.BerquistSherman(
        paid_amount='Paid', incurred_amount='Incurred',
        reported_count='Reported', closed_count='Closed')
    berq_triangle = berq.fit_transform(triangle)
    assert np.nansum((berq_triangle.latest_diagonal-triangle.latest_diagonal).values) == 0
