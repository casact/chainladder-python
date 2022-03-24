import chainladder as cl
import numpy as np


def test_basic_case_outstanding():
    tri = cl.load_sample("usauto")
    m = cl.CaseOutstanding(paid_to_incurred=("paid", "incurred")).fit(tri)
    out = cl.Chainladder().fit(m.fit_transform(tri))
    a = (out.full_triangle_["incurred"] - out.full_triangle_["paid"]).iloc[
        ..., -1, :9
    ] * m.paid_ldf_.values
    b = (out.full_triangle_["paid"].cum_to_incr().iloc[..., -1, 1:10]).values
    assert (a - b).max() < 1e-6


def test_outstanding_friedland_example():
    usauto = cl.load_sample("usauto")
    model = cl.CaseOutstanding(
        paid_to_incurred=("paid", "incurred"), paid_n_periods=3, case_n_periods=3
    ).fit(usauto)

    expected_paid_ldf = np.array(
        [
            [
                0.833,
                0.701,
                0.714,
                0.714,
                0.653,
                0.631,
                0.553,
                0.437,
                0.524,
            ]
        ]
    )
    assert (
        model.paid_ldf_.to_frame(origin_as_datetime=False).values - expected_paid_ldf
        < 0.001
    ).all()

    expected_case_ldf = np.array(
        [
            [
                0.526,
                0.566,
                0.528,
                0.486,
                0.511,
                0.555,
                0.652,
                0.674,
                0.580,
            ]
        ]
    )
    assert (
        model.case_ldf_.to_frame(origin_as_datetime=False).values - expected_case_ldf
        < 0.001
    ).all()
