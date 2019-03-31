"""
==================================
Sample Excel Exhibit functionality
==================================

This example demonstrates some of the flexibility of the ``Exhibits`` class. It
creates an Excel file called 'clrd.xlsx' that includes various statistics on
industry development patterns for each line of business in the CAS loss reserve
database.

Output can be viewed online in `Google Sheets <https://docs.google.com/spreadsheets/d/1fwHK1Sys6aHDhEhFO6stVJtmZVKEcXXBsmJLSLIBLJY/edit#gid=1190415861>`_.

See :ref:`Exhibits<exhibits>` for more detail.

.. _exhibit_example:
"""
import chainladder as cl
import pandas as pd

# Grab industry Paid Triangles
clrd = cl.load_dataset('clrd').groupby('LOB').sum()['CumPaidLoss']

# Create instance of Exhibits
exhibits = cl.Exhibits()

# Line of Business Dictionary for looping
lobs = dict(comauto='Commercial Auto',
            medmal='Medical Malpractice',
            othliab='Other Liability',
            ppauto='Private Passenger Auto',
            prodliab='Product Liability',
            wkcomp='Workers\' Compensation')

# Loop through each LOB
for key, value in lobs.items():
    title = ['CAS Loss Reserve Database',
             value, 'Cumulative Paid Loss',
             'Evaluated as of 31-December-1997']
    # Show Raw Triangle
    exhibits.add_exhibit(key, clrd.loc[key],
                         header=True, formats='money',
                         title=title, col_nums=False,
                         index_label='Accident Year')
    # Show Link Ratios
    exhibits.add_exhibit(key, clrd.loc[key].link_ratio,
                         header=True, formats='decimal',
                         col_nums=False,
                         index_label='Accident Year',
                         start_row=clrd.shape[2]+6)
    # Show various Various Averages
    df = pd.concat(
        (cl.Development(n_periods=2).fit(clrd.loc[key]).ldf_.drop_duplicates(),
         cl.Development(n_periods=3).fit(clrd.loc[key]).ldf_.drop_duplicates(),
         cl.Development(n_periods=7).fit(clrd.loc[key]).ldf_.drop_duplicates(),
         cl.Development().fit(clrd.loc[key]).ldf_.drop_duplicates(),
         cl.Development().fit(clrd.loc[key]).ldf_.drop_duplicates()),
        axis=0)
    df.index = ['2 Yr Wtd', '3 Yr Wtd', '7 Yr Wtd', '10 Yr Wtd', 'Selected']
    exhibits.add_exhibit(key, df, col_nums=False, formats='decimal',
                         index_label='Averages', start_row=clrd.shape[2]*2+7)

# Create Excel File
exhibits.to_excel('clrd.xlsx')
