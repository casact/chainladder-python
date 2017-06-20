# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:58:03 2017

@author: jboga
"""
import chainladder as cl
import pytest
import os.path

@pytest.mark.parametrize('df',[cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))])
def test_dataAsTable(df):
        a = cl.Triangle(df)
        data1 = a.data
        a.dataAsTable(inplace=True)
        a.dataAsTriangle(inplace=True)
        test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
        assert test.shape[0]*test.shape[1] == sum(sum(test))
        
@pytest.mark.parametrize('df',[cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))])
def test_cum2incr_convert(df):
        a = cl.Triangle(df)
        data1 = a.data
        a.cum2incr(inplace=True)
        a.incr2cum(inplace=True)
        test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
        assert test.shape[0]*test.shape[1] == sum(sum(test))
        
@pytest.mark.parametrize('df',[cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))])
def test_incr2cum_convert(df):
        a = cl.Triangle(df)
        data1 = a.data
        a.incr2cum(inplace=True)
        a.cum2incr(inplace=True)
        test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
        assert test.shape[0]*test.shape[1] == sum(sum(test))
    