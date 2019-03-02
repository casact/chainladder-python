.. _utils:

.. currentmodule:: chainladder

=========
Utilities
=========
Utilities contains example datasets and extra functionality to facilitate a
reserving workflow.

.. _exhibits:

Exhibits
========
:class:`Exhibits` can be used to output highly-customizable exhibits to Microsoft
Excel spreadsheets.  Think about it like a turbocharged ``pandas.to_excel`` method.
``Exhibits`` allows you to define multiple exhibits (excel sheets) within one
instance.

Example output of Exhibits
--------------------------
The following example highlights the general usage of the ``Exhibits``
class.  The image is the output of this :ref:`example<exhibit_example>`.

.. image:: ../_static/images/exhibits.PNG

Formats
-------
The Exhibits class comes with class level attributes representing the formats to
be used in the your Excel exhibits.  These formats are defined using Excel's
formatting convention.

============== =============================================================
Format         Description
============== =============================================================
title1_format  Title format of exhibit
title2_format  Title format of exhibit
title3_format  Title format of exhibit
title4_format  Title format of exhibit
index_format   Column name format of exhibit
header_format  Header format of exhibit
money_format   Format to be applied for currencies, default set to '#,##'
percent_format Format to be applied for percentages, default set to '0.0%'
decimal_format Format to be applied for decimals, default set to '0.000'
date_format    Format to be applied for decimals, default set to 'm/d/yyyy'
int_format     Format to be applied for integers, default set to 'General'
text_format    Format to be applied to text cells.
============== =============================================================

You can override any of these formats by modifying the class directly.

**Example:**
   >>> import chainladder as cl
   >>> cl.Exhibits.date_format['num_format'] = 'dd/mm/yyyy'

.. note::
   Overrides will only persist through the end of your python session and must
   be reapplied each time you import chainlader.

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
