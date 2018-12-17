import pandas as pd
import numpy as np
import copy
from chainladder.core.base import TriangleBase, check_triangle_postcondition


class Triangle(TriangleBase):
    """Four-dimensional tabular data structure with labeled axes (keys, values,
    origin, development). Triangle is the primary data structure in
    chainladder. The notation to manipulate the triangle object borrows heavily
    from pandas and the experience should feel familar to a practicioner versed
    in using pandas.

    The core data structure at the heart of the Triangle class is a 4D numpy
    array with dimensions defined as:

    Dimension 0 (Key Dimension):
        represents key dimensions or the lowest grain(s) at which you
        want to manage the triangle, e.g State, Company, etc. The
        grain supports multiple key dimensions that will behave like a
        pandas.multiIndex

    Dimension 1 (Value Dimension):
        represents value dimensions or numeric data to be represented
        in each triangle, e.g. Paid, Incurred, etc.

    Dimension 2 (Origin Dimension):
        represents the origin dimension which will be stored as a date
        e.g. Accident Month, Report Year, Policy Quarter, etc.

    Dimension 3 (Development Dimension):
        represents the development dimension which will be store
        e.g. Valuation Month, Valuation Year, Valuation Quarter, etc.

    Dimensions 0 and 1 are accessed like a pandas Dataframe.  You can think of
    the 4d structure as a 2D Dataframe where each element (row, col) is its
    own 2D triangle datatype.


    Parameters
    ----------
    data : DataFrame
        A single dataframe that contains columns represeting all other
        arguments to the Triangle constructor
    origin : str or list
         A representation of the accident, reporting or more generally the
         origin period of the triangle that will map to the Origin dimension
    development : str or list
        A representation of the development/valuation periods of the triangle
        that will map to the Development dimension
    values : str or list
        A representation of the keys of the triangle that will map to the
        Keys dimension.  If None, then a single 'Total' key will be generated.
    keys : str or list or None
        A representation of the keys of the triangle that will map to the
        Keys dimension.  If None, then a single 'Total' key will be generated.

    Attributes
    ----------
    keys : DataFrame
        Represents all available levels of the key dimension.
    values : DataFrame
        Represents all available levels of the value dimension.
    origin : DataFrame
        Represents all available levels of the origin dimension.
    development : DataFrame
        Represents all available levels of the development dimension.
    link_ratio: Triangle
        Set of age-to-age ratios for the triangle.
    age_to_age: Triangle
        See link_ratio

    """
    pass
