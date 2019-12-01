import numpy as np
import pandas as pd
import chainladder as cl


def test_simple_exhibit():
    raa = cl.load_dataset('raa')

    col = cl.Column(
        cl.DataFrame(raa, margin=(0,0,1,0)),
        cl.DataFrame(raa.link_ratio, formats={'italic': True})
    )
    composite = cl.Row(
        col, col,
        title=['This title spans both Column Objects'],
        margin=(0,1,0,0)
    )
    x = cl.Tabs(
       ('a_sheet', composite),
       ('another_sheet', composite)
    ).to_excel('workbook.xlsx')
