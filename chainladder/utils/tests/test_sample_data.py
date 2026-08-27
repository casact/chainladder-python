import numpy as np

import chainladder as cl


def test_prism_dense_samples() -> None:
    """
    The bundled ``prism_o?d?`` samples are ``prism`` summed across every claim
    and aggregated to a single grain, so each one must equal that aggregation
    computed on the fly.
    """
    dense = cl.load_sample("prism").sum()
    for name, grain in [
        ("prism_omdm", "OMDM"),
        ("prism_oqdq", "OQDQ"),
        ("prism_osds", "OSDS"),
        ("prism_oydy", "OYDY"),
    ]:
        tri = cl.load_sample(name)
        expected = dense.grain(grain)
        assert tri.shape == expected.shape
        assert list(tri.columns) == [
            "reportedCount",
            "closedPaidCount",
            "Paid",
            "Incurred",
        ]
        assert tri.origin_grain == grain[1]
        assert tri.development_grain == grain[3]
        assert not tri.is_cumulative
        assert np.allclose(np.nan_to_num(tri.values), np.nan_to_num(expected.values))
