"""
Tests for the triangle class
"""
import chainladder as cl
from chainladder import Triangle
import pandas as pd
import numpy as np
import os

PATH = os.path.join(os.path.split(cl.__file__)[0], 'tests')
DATA = pd.read_csv(os.path.join(PATH, 'simple_triangle.csv'))


def test_triangle_form():
    data = DATA.copy()
    triangle = Triangle(data.set_index('AccidentDate'))
    import pdb; pdb.set_trace()
    assert 1 == 1
