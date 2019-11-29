.. _triangle:

.. currentmodule:: chainladder

The Triangle
==============

Structure
---------

The :class:`Triangle` is the core data structure of the chainladder package.
Just as scikit-learn requires all datasets to be numpy arrays, the chainladder
package requires all data to be instances of the Triangle class.
The Triangle is a four-dimensional tabular data structure with labeled axes
(index, columns, origin, development). Triangle is the primary data structure
in chainladder. The notation to manipulate the triangle object borrows heavily
from pandas and the experience should feel familiar to a practitioner versed
in using pandas.

The core data structure at the heart of the Triangle class is a 4D numpy
array with dimensions defined as:

Dimension 0 (``index`` dimension):
    represents index dimensions or the lowest grain(s) at which you
    want to manage the triangle, e.g State, Company, etc. The
    grain supports multiple key dimensions that will behave like a
    pandas.multiIndex

Dimension 1 (``columns`` dimension):
    represents columns dimensions or numeric data to be represented
    in each triangle, e.g. Paid, Incurred, etc.

Dimension 2 (``origin`` dimension):
    represents the origin dimension which will be stored as a date
    e.g. Accident Month, Report Year, Policy Quarter, etc.

Dimension 3 (``development`` dimension):
    represents the development dimension
    e.g. Valuation Month, Valuation Year, Valuation Quarter, etc.

``index`` and ``columns`` are accessed like a pandas Dataframe.  You can think of
the 4d structure as a pandas Dataframe where each cell (row, col) is its
own triangle.  see :ref:`Slicing<slicing>`

You can access the ``values`` property of a triangle to get its numpy
representation, however the Triangle class provides many helper methods to
keep the shape of the numpy representation in sync with the other Triangle
properties.

Inferring Dates when creating an instance
-----------------------------------------
When instantiating a :class:`Triangle`, the ``origin`` and ``development``
arguments can take a ``str`` representing the column name in your pandas DataFrame
that contains the relevant information.  Alternatively, the arguments can also
take a ``list`` in the case where your DataFrame includes multiple columns that
represent the dimension, e.g. ``['accident_year','accident_quarter']`` can be
supplied to create an ``origin`` dimension at the accident quarter grain.

**Example:**
   >>> import chainladder as cl
   >>> cl.Triangle(data, origin='Acc Year', development=['Cal Year', 'Cal Month'], values=['Paid Loss'])

 Triangle relies heavily on pandas date inference, and while pretty good it does
 not infer all dates correctly.  When initializing a Triangle you can always
 use the ``origin_format`` and/or ``development_format`` arguments to force
 the inference, e.g. ``origin_format='%Y/%m/%d'

.. _slicing:
Slicing and Boolean Indexing
----------------------------
With a Triangle is created, individual triangles can be sliced out of the object
using pandas-style ``loc``/``iloc`` or boolean indexing.

**Example:**
   >>> import chainladder as cl
   >>> clrd = cl.load_dataset('clrd')
   >>> clrd.iloc[0,1]
   >>> clrd[clrd['LOB']=='othliab']
   >>> clrd['EarnedPremDIR']

.. note::
   Boolean indexing on non-index columns in pandas feels natural.  We've exposed
   the same syntax specifically for the index column(s) of the Triangle without the
   need for ``reset_index()`` or trying to boolean-index on a ``MultiIndex``. This is
   a divergence from the pandas API.

Changing Triangle Granularity
-----------------------------
If your triangle has origin and development grains that are more frequent then
yearly, you can easily swap to a higher grain using the `grain` method of the
:class:`Triangle`. The `grain` method recognizes Yearly (Y), Quarterly (Q), and
Monthly (M) grains for both the origin period and development period.

**Example:**
   >>> import chainladder as cl
   >>> cl.load_dataset('quarterly')
   Valuation: 2006-03
   Grain:     OYDQ
   Shape:     (1, 2, 12, 45)
   index:      ['Total']
   columns:    ['incurred', 'paid']
   >>> cl.load_dataset('quarterly').grain('OYDY')
   Valuation: 2006-03
   Grain:     OYDY
   Shape:     (1, 2, 12, 12)
   index:      ['Total']
   columns:    ['incurred', 'paid']

Deriving New Columns
--------------------
Most arithmetic operations can be used to create new triangles within your
triangle instance. Like with pandas, these can automatically be added as new
columns to your :class:`Triangle`.

**Example:**
   >>> clrd = cl.load_dataset('clrd')
   >>> clrd['CaseIncur'] = clrd['IncurLoss']-clrd['BulkLoss']
   >>> clrd
   Valuation: 1997-12
   Grain:     OYDY
   Shape:     (775, 7, 10, 10)
   index:      ['GRNAME', 'LOB']
   columns:    ['BulkLoss', 'CumPaidLoss', 'EarnedPremCeded', 'EarnedPremDIR', 'EarnedPremNet', 'IncurLoss', 'CaseIncur']


Aggregating Data
----------------
Much like in pandas, you can aggregate multiple triangles within a :class:`Triangle`
by using ``sum()`` which can optionally be coupled with ``groupby()``.

**Example:**
   >>> clrd = cl.load_dataset('clrd')
   >>> clrd.sum()
   Valuation: 1997-12
   Grain:     OYDY
   Shape:     (1, 6, 10, 10)
   index:      ['All']
   columns:    ['BulkLoss', 'CumPaidLoss', 'EarnedPremCeded', 'EarnedPremDIR', 'EarnedPremNet', 'IncurLoss']
   >>> clrd.groupby('LOB').sum()
   Valuation: 1997-12
   Grain:     OYDY
   Shape:     (6, 6, 10, 10)
   index:      ['LOB']
   columns:    ['BulkLoss', 'CumPaidLoss', 'EarnedPremCeded', 'EarnedPremDIR', 'EarnedPremNet', 'IncurLoss']

Aggregation functions can also be used without ``groupby()``.  By default,
the aggregation will apply to the first axis with a length greater than 1.
Alternatively, you can specify the axis using the ``axis`` argument of the
aggregate method.

Accessor methods
----------------
Like pandas ``.str`` and ``.dt`` accessor functions, you can also perform operations
on the ``origin``, ``development`` or ``valuation`` of a triangle. For example, all
of these operations are legal.

**Example:**
   >>> raa = cl.load_dataset('raa')
   >>> x = raa[raa.origin=='1986']
   >>> x = raa[(raa.development>=24)&(raa.development<=48)]
   >>> x = raa[raa.origin<='1985-JUN']
   >>> x = raa[raa.origin>'1987-01-01'][raa.development<=36]
   >>> x = raa[raa.valuation<raa.valuation_date]


Valuation or development
------------------------
While most Estimators that use triangles expect the development period to be
expressed as an origin age, it is possible to transform a trianle into a valuation
triangle where the development periods are converted to valuation periods.  Expressing
triangles this way may provide a more convenient view of valuation slices.
Switching between a development triangle and a valuation triangle can be 
accomplished with the method `dev_to_val()` and its inverse `'val_to_dev()`.  For
example, slicing the calendar period incurred for the last two years of a 
triangle in a more compact tablular format can be done as follows:

**Example:**
   >>> import chainladder as cl
   >>> raa = cl.load_dataset('raa').dev_to_val()
   >>> raa.cum_to_incr()[raa.valuation>='1989']
           1989    1990
   1981    54.0   172.0
   1982   673.0   535.0
   1983   649.0   603.0
   1984  2658.0   984.0
   1985  3786.0   225.0
   1986  1233.0  2917.0
   1987  6926.0  1368.0
   1988  5596.0  6165.0
   1989  3133.0  2262.0
   1990     NaN  2063.0

   
Commutative methods
-------------------
Where possible, the triangle methods are designed to be commutative.  For example,
each of these operations is functionally equivalent..\


**Example:**
   >>> import chainladder as cl
   >>> full = cl.Chainladder().fit(cl.load_dataset('quarterly')).full_expectation_
   >>> # Functionally equivalent transformations
   >>> full.grain('OYDY').val_to_dev() == full.val_to_dev().grain('OYDY')
   >>> full.cum_to_incr().grain('OYDY').val_to_dev() == full.val_to_dev().cum_to_incr().grain('OYDY')
   >>> full.grain('OYDY').cum_to_incr().val_to_dev().incr_to_cum() == full.val_to_dev().grain('OYDY')



There are many more methods available to manipulate triangles.  The complete 
list of methods is available under the :class:`Triangle` docstrings.
