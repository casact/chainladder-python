import numpy as np
import pandas as pd
import chainladder as cl


def test_simple_exhibit():
    exhibits = cl.Exhibits()
    exhibits.add_exhibit(data=cl.load_dataset('raa'),
                         col_nums=False,
                         sheet_name='Sheet1')
    exhibits.add_exhibit(data=cl.load_dataset('raa'),
                         col_nums=False,
                         sheet_name='Sheet2')
    exhibits.del_exhibit('Sheet1')
    exhibits.to_excel('test_excel.xlsx')
    np.testing.assert_equal(pd.read_excel('test_excel.xlsx', index_col=0).values,
                            cl.load_dataset('raa').to_frame().values)
