"""
"""

import pandas as pd
import numpy as np
import copy
from chainladder.core.base import TriangleBase, check_triangle_postcondition


class Triangle(TriangleBase):
    """
    The core data structure of the chainladder package
    
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
    keys
        Represents all available levels of the key dimension.
    values
        Represents all available levels of the value dimension.
    origin
        Represents all available levels of the origin dimension.
    development
        Represents all available levels of the development dimension.
    link_ratio, age_to_age
        Set of age-to-age ratios for the triangle.
    valuation_date
        The latest valuation date of the data
    loc
        pandas-style `loc` accessor
    iloc
        pandas-style `iloc` accessor

    """
    pass
