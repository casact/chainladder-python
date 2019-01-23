.. _triangle:

.. currentmodule:: chainladder

The Triangle
==============


The :class:`Triangle` is the core data structure of the chainladder package.
Just as scikit-learn requires all datasets to be numpy arrays, the chainladder
package requires all data to be instances of the Triangle class.

Four-dimensional tabular data structure with labeled axes (index, columns,
origin, development). Triangle is the primary data structure in
chainladder. The notation to manipulate the triangle object borrows heavily
from pandas and the experience should feel familar to a practitioner versed
in using pandas.

The core data structure at the heart of the Triangle class is a 4D numpy
array with dimensions defined as:

Dimension 0 (``index`` dimension):
    represents key dimensions or the lowest grain(s) at which you
    want to manage the triangle, e.g State, Company, etc. The
    grain supports multiple key dimensions that will behave like a
    pandas.multiIndex

Dimension 1 (``columns`` dimension):
    represents value dimensions or numeric data to be represented
    in each triangle, e.g. Paid, Incurred, etc.

Dimension 2 (``origin`` dimension):
    represents the origin dimension which will be stored as a date
    e.g. Accident Month, Report Year, Policy Quarter, etc.

Dimension 3 (``development`` dimension):
    represents the development dimension which will be store
    e.g. Valuation Month, Valuation Year, Valuation Quarter, etc.

``index`` and ``columns`` are accessed like a pandas Dataframe.  You can think of
the 4d structure as a 2D Dataframe where each cell (row, col) is its
own 2D triangle.
