.. _triangle:

.. currentmodule:: chainladder

The Triangle
==============

Structure
---------
The :class:`Triangle` is the data structure of the chainladder package. Just as
Scikit-learn likes to only consume numpy arrays, Chainladder only likes
Triangles.  Triangle is the primary data structure in chainladder. It is a 4D
data structure with labeled axes.  These axes are its index, columns, origin,
development.

``index`` (axis 0):
    The index  is the lowest grain at which you want to manage the triangle.
    These can be things like state or company.  Like a pandas.multiIndex, you
    can throw more than one column into the index.

``columns`` (axis 1):
    Columns are where you would want to store the different numeric values of your
    data. Paid, Incurred, Counts are all reasonable choices for the columns of your
    triangle.

``origin`` (axis 2):
    The origin is the period of time from which your columns originate.  It can
    be an Accident Month, Report Year, Policy Quarter or any other period-like vector.

``development`` (axis 3):
    Development represents the development age or date of your triangle.
    Valuation Month, Valuation Year, Valuation Quarter in a are good choices.

Despite this structure, you interact with it in the style of pandas. You would
use ``index`` and ``columns`` in the same way you would for a pandas Dataframe.
You can think of the 4D structure as a pandas Dataframe where each cell (row,
column) is its own triangle.  see :ref:`Slicing<slicing>`

You can access the ``values`` property of a triangle to get its numpy
representation, however the Triangle class provides many helper methods to
keep the shape of the numpy representation in sync with the other Triangle
properties.

Inferring dates when creating an instance
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
Slicing and boolean indexing
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

Changing triangle granularity
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

Deriving new columns
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


Aggregating data
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

Converting to dataframes
------------------------
When a triangle is presented with a single index level and single column, it
becomes a 2D object.  As such, its display format changes to that similar to a
dataframe.  These 2D triangles can easily be converted to a pandas dataframe
using the `to_frame()` method.

**Example:**
  >>> import chainladder as cl
  >>> clrd = cl.load_dataset('clrd')
  >>> clrd
  Valuation: 1997-12
  Grain:     OYDY
  Shape:     (775, 6, 10, 10)
  Index:      ['GRNAME', 'LOB']
  Columns:    ['BulkLoss', 'CumPaidLoss', 'EarnedPremCeded', 'EarnedPremDIR', 'EarnedPremNet', 'IncurLoss']
  >>> clrd[clrd['LOB']=='ppauto']['CumPaidLoss'].sum().to_frame()
              12          24          36          48          60          72          84          96         108        120
  1988  3092818.0   5942711.0   7239089.0   7930109.0   8318795.0   8518201.0   8610355.0   8655509.0  8682451.0  8690036.0
  1989  3556683.0   6753435.0   8219551.0   9018288.0   9441842.0   9647917.0   9753014.0   9800477.0  9823747.0        NaN
  1990  4015052.0   7478257.0   9094949.0   9945288.0  10371175.0  10575467.0  10671988.0  10728411.0        NaN        NaN
  1991  4065571.0   7564284.0   9161104.0  10006407.0  10419901.0  10612083.0  10713621.0         NaN        NaN        NaN
  1992  4551591.0   8344021.0  10047179.0  10901995.0  11336777.0  11555121.0         NaN         NaN        NaN        NaN
  1993  5020277.0   9125734.0  10890282.0  11782219.0  12249826.0         NaN         NaN         NaN        NaN        NaN
  1994  5569355.0   9871002.0  11641397.0  12600432.0         NaN         NaN         NaN         NaN        NaN        NaN
  1995  5803124.0  10008734.0  11807279.0         NaN         NaN         NaN         NaN         NaN        NaN        NaN
  1996  5835368.0   9900842.0         NaN         NaN         NaN         NaN         NaN         NaN        NaN        NaN
  1997  5754249.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN        NaN        NaN

From this point the results can be operated on directly in pandas.  The
`to_frame()` functionality works when a Triangle is sliced down to any two axes
and is not limited to just the ``index`` and ``column``.

**Example:**
  >>> # 2D Triangle expressed as a Triangle
  >>> clrd['CumPaidLoss'].groupby('LOB').sum().latest_diagonal
  Valuation: 1997-12
  Grain:     OYDY
  Shape:     (6, 1, 10, 1)
  Index:      ['LOB']
  Columns:    ['CumPaidLoss']
  >>> # 2D Triangle expressed as a DataFrame
  >>> clrd['CumPaidLoss'].groupby('LOB').sum().latest_diagonal.to_frame()
  origin         1988       1989        1990        1991        1992        1993        1994        1995       1996       1997
  comauto    626097.0   674441.0    718396.0    711762.0    731033.0    762039.0    768095.0    675166.0   510191.0   272342.0
  medmal     217239.0   222707.0    235717.0    275923.0    267007.0    276235.0    252449.0    209222.0   107474.0    20361.0
  othliab    317889.0   350684.0    361103.0    426085.0    389250.0    434995.0    402244.0    294332.0   191258.0    54130.0
  ppauto    8690036.0  9823747.0  10728411.0  10713621.0  11555121.0  12249826.0  12600432.0  11807279.0  9900842.0  5754249.0
  prodliab   110973.0   112614.0    121255.0    100276.0     76059.0     94462.0    111264.0     62018.0    28107.0    10682.0
  wkcomp    1241715.0  1308706.0   1394675.0   1414747.0   1328801.0   1187581.0   1114842.0    962081.0   736040.0   340132.0


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
expressed as an origin age, it is possible to transform a triangle into a valuation
triangle where the development periods are converted to valuation periods.  Expressing
triangles this way may provide a more convenient view of valuation slices.
Switching between a development triangle and a valuation triangle can be
accomplished with the method `dev_to_val()` and its inverse `val_to_dev()`.  For
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
