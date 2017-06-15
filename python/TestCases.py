# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 19:50:13 2017

@author: jboga
"""

from Triangles import triangle
import os.path
import pandas as pd

path = os.path.abspath('')[:-6] + 'data\\'
df_list = [pd.read_pickle(path + item) for item in os.listdir(path)] 

def test_tri_to_table_convert(df):
    try:
        a = triangle(df)
        data1 = a.data
        a.dataAsTable(inplace=True)
        a.dataAsTriangle(inplace=True)
        test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
        return test.shape[0]*test.shape[1] == sum(sum(test))
    except:
        return False

test1 = pd.Series([item.shape != (0,0) for item in df_list], index=os.listdir(path), name='test1')
test2 = pd.Series([test_tri_to_table_convert(item) for item in df_list], index=os.listdir(path), name='test2')
tests = pd.concat([test1, test2],axis=1)

