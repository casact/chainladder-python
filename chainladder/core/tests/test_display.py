import pytest
import chainladder as cl


def test_heatmap_render(raa):
    """The heatmap method should render correctly given the sample."""
    assert raa.heatmap()


def test_empty_triangle():
    assert cl.Triangle()


def test_to_frame(raa):
    assert cl.Chainladder().fit(raa).cdf_.to_frame()
    assert cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=False)
    assert cl.Chainladder().fit(raa).cdf_.to_frame(origin_as_datetime=True)
    assert cl.Chainladder().fit(raa).ultimate_.to_frame()
    assert cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=False)
    assert cl.Chainladder().fit(raa).ultimate_.to_frame(origin_as_datetime=True)
