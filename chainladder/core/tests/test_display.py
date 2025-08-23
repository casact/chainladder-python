import pytest
import chainladder as cl


def test_heatmap_render(raa):
    """The heatmap method should render correctly given the sample."""
    try:
        raa.heatmap()

    except:
        assert False


# def test_empty_triangle():
#     assert cl.Triangle()


def test_to_frame(raa):
    try:
        cl.Chainladder().fit(raa).cdf_.to_frame()
        cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=False)
        cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=True)
        cl.Chainladder().fit(raa).ultimate_.to_frame()
        cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=False)
        cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=True)

    except:
        assert False


def test_labels(xyz):
    assert (
        xyz.valuation_date.strftime("%Y-%m-%d %H:%M:%S.%f")
        == "2008-12-31 23:59:59.999999"
    )
    assert xyz.origin_grain == "Y"
    assert xyz.development_grain == "Y"
    assert xyz.shape == (1, 5, 11, 11)
    assert xyz.index_label == ["Total"]
    assert xyz.columns_label == ["Incurred", "Paid", "Reported", "Closed", "Premium"]
    assert xyz.origin_label == ["AccidentYear"]
