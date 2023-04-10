import chainladder as cl
import pytest


def test_fit_period():
    tri = cl.load_sample("tail_sample")
    dev = cl.Development(average="simple").fit_transform(tri)
    assert (
        round(
            cl.TailCurve(fit_period=(tri.ddims[-7], None), extrap_periods=10)
            .fit(dev)
            .cdf_["paid"]
            .set_backend("numpy", inplace=True)
            .values[0, 0, 0, -2],
            3,
        )
        == 1.044
    )


def test_curve_validation():
    """
    Test validation of the curve parameter. Should raise a value error if an incorrect argument is supplied.
    """

    with pytest.raises(ValueError):
        tri = cl.load_sample('tail_sample')
        cl.TailCurve(
            curve='Exponential'
        ).fit_transform(tri)


def test_errors_validation():
    """
    Test validation of the errors parameter. Should raise a value error if an incorrect argument is supplied.
    """
    with pytest.raises(ValueError):
        tri = cl.load_sample('tail_sample')
        cl.TailCurve(
            errors='Ignore'
        ).fit_transform(tri)
