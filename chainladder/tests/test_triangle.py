"""
Tests for the triangle class
"""
import chainladder as cl
from chainladder import Triangle
import pandas as pd
from pandas.testing import assert_series_equal
# from numpy.testing import assert_allclose
import numpy as np
import os

PATH = os.path.join(os.path.split(cl.__file__)[0], 'tests')
simple_triangle = pd.read_csv(os.path.join(PATH, 'simple_triangle.csv'))
simple_triangle.set_index('AccidentDate', inplace=True)
simple_triangle = simple_triangle.astype(np.float_)
simple_tabular = simple_triangle.copy().stack()


def test_triangle_form():
    data = simple_triangle.copy()
    triangle = Triangle(data)
    assert_series_equal(triangle.data, simple_tabular)


def test_tabular_form():
    data = simple_tabular.copy()
    triangle = Triangle(data, dataform='tabular', origin='AccidentDate',
                        development='DevPeriod')
    assert_series_equal(triangle.data, simple_tabular)
    data2 = triangle.data_as_table()
    data2 = data2.astype(np.float_)
    assert_series_equal(data, data2)


def test_triangle_tabular_equal():
    tri_data = simple_triangle.copy()
    tab_data = simple_tabular.copy()
    tri_triangle = Triangle(tri_data)
    tab_triangle = Triangle(tab_data, dataform='tabular',
                            origin='AccidentDate',
                            development='DevPeriod')
    assert_series_equal(tri_triangle.data, tab_triangle.data)


def test_triangle_math():
    data1 = simple_tabular.copy()
    data2 = simple_tabular.copy() + 30

    tri1 = Triangle(data1, dataform='tabular')
    tri2 = Triangle(data2, dataform='tabular')

    tri3 = tri1 + tri2
    assert_series_equal(tri3.data, data1 + data2)

    tri3 = tri1 - tri2
    assert_series_equal(tri3.data, data1 - data2)

    tri3 = tri1 * tri2
    assert_series_equal(tri3.data, data1 * data2)

    tri3 = tri1 / tri2
    assert_series_equal(tri3.data, data1 / data2)


def test_incremental_to_cumulative():
    data = simple_tabular.copy()
    triangle = Triangle(data=data, dataform='tabular', cumulative=False)
    assert_series_equal(triangle.data, data.groupby(level=[0]).cumsum())
    assert_series_equal(triangle.incr_to_cum(),
                        data.groupby(level=[0]).cumsum())
    assert_series_equal(triangle.cum_to_incr(), data)
