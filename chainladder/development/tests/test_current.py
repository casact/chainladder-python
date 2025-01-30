import chainladder as cl
import numpy as np
import pytest


def test_simple_incremental_pattern():
    # Create a simple triangle with known incremental pattern
    triangle = cl.load_sample("GenIns")
    
    dev = cl.CurrentDevelopment(n_developments=1).fit(triangle)
    expected_ldf = 2.4906065479322863  # We expect a 2:1 ratio
    np.testing.assert_almost_equal(dev.ldf_.values[0, 0, 0, 0], expected_ldf)
