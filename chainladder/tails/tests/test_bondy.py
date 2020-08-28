import chainladder as cl


def test_bondy1():
    tri = cl.load_sample("tail_sample")["paid"]
    dev = cl.Development(average="simple").fit_transform(tri)
    assert round(float(cl.TailBondy().fit(dev).cdf_.values[0, 0, 0, -2]), 3) == 1.028
