.. _utils:

.. currentmodule:: chainladder

=========
Utilities
=========
Utilities contains example datasets and extra functionality to facilitate a
reserving workflow.

.. _exhibits:

Writing to Excel
================

Example output of Exhibits
--------------------------
The following example highlights the general look of the outputs when using
chainladder to output to Excel.

.. image:: ../_static/images/exhibits.PNG

Dataframe
---------

Chainladder comes with a :class:`DataFrame` class that has a supercharged version of
`pd.DataFrame.to_excel`.  This version allows for the export of any number of
DataFrames in any layout desired with formats of your choosing.  Simply wrap a
pandas dataframe or a 2D tringle representation in the Dataframe to get access
to this supercharged version.

**Example:**
   >>> import chainladder as cl
   >>> raa = cl.load_dataset('raa')
   >>> cl.DataFrame(raa).to_excel('workbook.xlsx')

This class is used exclusively for exporting to Excel and the normal
``pd.DataFrame`` should be used for any other purpose.  There are commonalities
between ``cl.DataFrame.to_excel()`` and ``pd.DataFrame.to_excel()``.  For example,
both have arguments for ``header``, ``index``, ``index_label`` that behave
identically, however the argument placement happens at object initialization
for the :class:`DataFrame`:

**Example:**
   >>> # chainladder
   >>> cl.DataFrame(raa, header=False, index=True, index_label='Origin').to_excel('workbook.xlsx')
   >>> # vs
   >>> # pandas
   >>> raa.to_frame().to_excel('workbook.xlsx', header=False, index=True, index_label='Origin')

By placing the arguments at object initialization allows for the construction
of composite objects as we will see later. ``pd.DataFrame.to_excel`` provides
additional ``startrow`` and ``startcol`` arguments in the case where you want
the dataframe exported anywhere other than cell A1.  `cl.DataFrame.to_excel`
replaces these with a single `margin` argument that behaves similarly to a
Cascading Stylesheet margin.  The ``margin`` option can be expressed as an
integer e.g. ``margin=1`` which will place empty cells around the dataframe.
The ``margin`` can also be expressed as a tuple. ``margin=(1,0)`` puts a top
and bottom set of cells, but not left and right. ``margin=(2,0,0,1)`` puts two
empty rows above the dataframe and one to the left. Example comparison to
``pd.DataFrame.to_excel``

**Example:**
   >>> # chainladder
   >>> cl.DataFrame(raa, margin=(3,0,0,1)).to_excel('workbook.xlsx')
   >>> # vs
   >>> # pandas
   >>> raa.to_frame().to_excel('workbook.xlsx', startrow=3, startcol=1)

**Formatting**

Formatting output is key to having a polished looking spreadsheet, but
unfortunately pandas does not help much.  Chainladder, uses xlsxwriter to apply
formats to the data cells in a dataframe.  Formats are expressed as
dictionaries.  You can specify a single set of formats for the entire dataframe:

**Example:**
   >>> formats={'num_format':'#,#', 'font_color':'red'}
   >>> cl.DataFrame(raa, formats=formats).to_excel('workbook.xlsx')

Alternatively, you can specify formats for each column individually using a
nested dictionary.

**Example:**
   >>> formats={'Ultimate':{'num_format':'#,#', 'font_color':'red'},
   ...          'Latest':  {'num_format':'#,0.00', 'bold':True}}

For more information on available formats refer to
https://xlsxwriter.readthedocs.io/format.html

**Other Features**
`cl.DataFrame` also allows for adding a title and column numbering. Titles are
expressed as a list:

**Example:**
   >>> title=['Sample Accident Year Triangle',
   ...        'Sourced from Mack',
   ...        'Evaluated as of Dec-1990']
   >>> cl.DataFrame(raa, title=title).to_excel('workbook.xlsx')

Many actuarial exhibits include column numbering for easier reference.
This can be turned on using the ``col_nums=True``.

Laying out composite objects
============================

While the addition of formats, titles and column numbering provide a little more
flexibility that can be obtained from ``pd.DataFrame.to_excel``, chainladder
provides a lot more flexibility with its layout objects.  There are three
layout objects `Tabs`, `Row`, and `Column`.

.. note::
   The layout API borrows from the bokeh/holoviz API and should be familiar to
   the practitioner who uses those for visualization.

Rows and Columns
----------------

:class:`Column` takes multiple objects and displays them vertically.

**Example:**
   >>> col = cl.Column(
   ...     cl.DataFrame(raa, margin=(0,0,1,0)),
   ...     cl.DataFrame(raa.link_ratio, formats={'italic': True})
   ... )
   >>> col.to_excel('workbook.xlsx')

:class:`Row` takes multiple objects and displays them horizontally.

**Example:**
   >>> cl.Row(
   ...     cl.DataFrame(raa, margin=(0,0,1,0)),
   ...     cl.DataFrame(raa.link_ratio)
   ... ).to_excel('workbook.xlsx')

You can also nest ``Row`` and ``Column``  within rows and columns.  Nesting can
be a deep as you want allowing for a highly customized layout.
**Example:**
   >>> cl.Row(col, col).to_excel('workbook.xlsx')

``Row`` and ``Column`` optionally take `title` and `margin` keywords that
function the same as those in ``cl.DataFrame``.

**Example:**
   >>> composite = cl.Row(
   ...     col, col,
   ...     title=['This title spans both Column Objects'],
   ...     margin=(0,1,0,0)
   ... )
   >>> composite.to_excel('workbook.xlsx')

Tabs
----

:class:`Tabs` are the sheet representation of these objects.

**Example:**
   >>> cl.Tabs(
   ...    ('a_sheet', composite),
   ...    ('another_sheet', composite)
   ... ).to_excel('workbook.xlsx')


Datasets
========
A variety of datasets can be loaded using :func:`load_dataset()`.  These are
sample datasets that are used in a variety of examples within this
documentation.

========= =======================================================
Dataset   Description
========= =======================================================
abc       ABC Data
auto      Auto Data
cc_sample Sample Insurance Data for Cape Cod Method in Struhuss
clrd      CAS Loss Reserving DataBase
genins    General Insurance Data used in Clark
ia_sample Sample data for Incremental Additive Method in Schmidt
bs_sample Sample data for Bootstrap sampling in Shapland
liab      more data
m3ir5     more data
mcl       Sample insurance data for Munich Adjustment in Quarg
mortgage  more data
mw2008    more data
mw2014    more data
quarterly Sample data to demonstrate changing Triangle grain
raa       Sample data used in Mack Chainladder
ukmotor   more data
usaa      more data
usauto    more data
========= =======================================================
