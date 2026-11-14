import chainladder as cl
from chainladder._config import __grain_to_months__


def test_grain_to_months_values():
    assert __grain_to_months__ == {"Y": 12, "S": 6, "Q": 3, "M": 1}


def test_grain_to_months_matches_triangle_dstep():
    raa = cl.load_sample("raa")
    assert __grain_to_months__ == raa._dstep()["M"]
