import chainladder as cl
import numpy as np
import pytest


def test_simple_currend_development_pattern():
    # Create a simple triangle with known incremental pattern
    triangle = cl.load_sample("GenIns")
    
    dev = cl.CurrentDevelopment(n_developments=1).fit(triangle)
    expected_ldf = 2.4906065479322863
    np.testing.assert_almost_equal(dev.ldf_.values[0, 0, 0, 0], expected_ldf)

def test_currend_development_pattern_2_periods():
    # Create a simple triangle with known incremental pattern
    triangle = cl.load_sample("GenIns")
    
    dev = cl.CurrentDevelopment(n_developments=2).fit(triangle)
    expected_ldf = 2.4906065479322863
    expected_ldf2 = 0.7473326421
    np.testing.assert_almost_equal(dev.ldf_.values[0, 0, 0, 0], expected_ldf)
    np.testing.assert_almost_equal(dev.ldf_.values[0, 0, 0, 1], expected_ldf2)

    dev = cl.CurrentDevelopment(n_periods=4, n_developments=2).fit(triangle)
    expected_ldf = 2.43652590
    expected_ldf2 = 0.8523101632
    np.testing.assert_almost_equal(dev.ldf_.values[0, 0, 0, 0], expected_ldf)
    np.testing.assert_almost_equal(dev.ldf_.values[0, 0, 0, 1], expected_ldf2)