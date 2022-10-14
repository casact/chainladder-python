import chainladder as cl


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
