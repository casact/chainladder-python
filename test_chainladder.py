# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:58:03 2017

@author: jboga
"""
import chainladder as cl
import pytest
import os.path
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
pandas2ri.activate()
import numpy as np
CL =importr('ChainLadder')
d = r('data(package=\"ChainLadder\")')

@pytest.mark.parametrize('df',[cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))])
def test_dataAsTable(df):
    a = cl.Triangle(df)
    if a.dataform == 'tabular':
        assert True
    else:
        
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


l1 = [cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))]
l2 = [item for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))]
l3 = list(zip(l1,l2))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainLadder_f(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainLadder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainLadder(cl.Triangle(python_data),alpha=2).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[4])
        assert np.all(r_lang.round(5)==python.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainLadder_fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainLadder(cl.Triangle(python_data),alpha=2).fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[5])
        assert np.all(r_lang.round(5)==python.round(5))
