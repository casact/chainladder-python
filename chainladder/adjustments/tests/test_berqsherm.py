import chainladder as cl
import numpy as np


def test_preserve_diagonal():
    triangle = cl.load_sample("berqsherm").loc["Auto"]
    xp = triangle.get_array_module()
    berq = cl.BerquistSherman(
        paid_amount="Paid",
        incurred_amount="Incurred",
        reported_count="Reported",
        closed_count="Closed",
    )
    berq_triangle = berq.fit_transform(triangle)
    assert (
        xp.nansum((berq_triangle.latest_diagonal - triangle.latest_diagonal).values)
        == 0
    )
    assert berq_triangle != triangle

def test_adjusted_values():
    triangle = cl.load_sample("berqsherm").loc["MedMal"]
    xp = triangle.get_array_module()
    berq = cl.BerquistSherman(
        paid_amount="Paid",
        incurred_amount="Incurred",
        reported_count="Reported",
        closed_count="Closed",
        trend=0.15,
    )
    berq_triangle = berq.fit_transform(triangle)

    assert np.allclose(
        triangle["Reported"].values, berq_triangle["Reported"].values, equal_nan=True
    )

    # Ensure that the incurred, paid, and closed count columns are as expected
    berq_triangle.values[np.isnan(berq_triangle.values)] = 0
    assert np.isclose(
        berq_triangle["Incurred"].values.sum(), 1126985253.661, atol=1e-2
    )
    assert np.isclose(berq_triangle["Paid"].values.sum(), 182046766.054, atol=1e-2)
    assert np.isclose(berq_triangle["Closed"].values.sum(), 8798.982, atol=1e-2)