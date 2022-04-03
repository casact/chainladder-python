import pytest
import chainladder as cl


def test_heatmap_render(raa):
    """ The heatmap method should render correctly given the sample."""
    return raa.heatmap()


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
