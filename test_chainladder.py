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
def test_data_as_table(df):
    a = cl.Triangle(df)
    if a.dataform == 'tabular':
        assert True
    else:
        
        data1 = a.data
        a.data_as_table(inplace=True)
        a.data_as_triangle(inplace=True)
        test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
        assert test.shape[0]*test.shape[1] == sum(sum(test))
       
@pytest.mark.parametrize('df',[cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))])
def test_cum_to_incr_convert(df):
        a = cl.Triangle(df)
        data1 = a.data
        a.cum_to_incr(inplace=True)
        a.incr_to_cum(inplace=True)
        test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
        assert test.shape[0]*test.shape[1] == sum(sum(test))
        
@pytest.mark.parametrize('df',[cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))])
def test_incr_to_cum_convert(df):
        a = cl.Triangle(df)
        data1 = a.data
        a.incr_to_cum(inplace=True)
        a.cum_to_incr(inplace=True)
        test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
        assert test.shape[0]*test.shape[1] == sum(sum(test))


l1 = [cl.load_dataset(item) for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))]
l2 = [item for item in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'chainladder','data'))]
l3 = list(zip(l1,l2))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_f(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=2).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[4])
        assert np.all(r_lang.round(5)==python.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=2).fse[:-1]
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[5])
        assert np.all(r_lang.round(5)==python.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_Fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=2).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[6])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
        
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_process_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=2).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[8])
        assert np.all(np.array(python).round(5)==r_lang.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_parameter_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=2).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[9])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
############## alpha = 1
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_f_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=1).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[4])
        assert np.all(r_lang.round(5)==python.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=1).fse[:-1]
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[5])
        assert np.all(r_lang.round(5)==python.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_Fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=1).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[6])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
        
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_process_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=1).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[8])
        assert np.all(np.array(python).round(5)==r_lang.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_parameter_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=1).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[9])
        assert np.all(np.array(python).round(5)==r_lang.round(5))

        
############## Alpha = 0
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_f_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=0).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[4])
        assert np.all(r_lang.round(5)==python.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=0).fse[:-1]
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[5])
        assert np.all(r_lang.round(5)==python.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_Fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=0).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[6])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
        
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_process_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=0).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[8])
        assert np.all(np.array(python).round(5)==r_lang.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_parameter_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),alpha=0).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[9])
        assert np.all(np.array(python).round(5)==r_lang.round(5))

        

def test_MunichChainladder_1():
    python = cl.MunichChainladder(cl.load_dataset('MCLpaid'), cl.load_dataset('MCLincurred'))
    r_lang = np.array(r('a<-MunichChainLadder(MCLpaid,MCLincurred)'))
    assert np.all((round(python.lambdaI,5)==round(np.array(r_lang[12][0])[0],5)) and
                  (round(python.lambdaP,5)==round(np.array(r_lang[11][0])[0],5)) and
                  (python.q_f.round(5) == np.array(r_lang[15]).round(5)))
    

#################        Try with exponential tail
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_f(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong','auto$PersonalAutoIncurred']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=2).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=2)')[4])
        assert np.all(r_lang.round(5)==python.round(5))
  
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong', 'auto$PersonalAutoIncurred']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=2).fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=2)')[5])
        assert np.all(r_lang.round(5)==python.round(5))
  
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_Fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=2).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=2)')[6])
        assert np.all(np.array(python).round(5)==r_lang.round(5))

"""  

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_process_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=2).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=2)')[8])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_parameter_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=2).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=2)')[9])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    ############## alpha = 1
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_f_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1)')[4])
        assert np.all(r_lang.round(5)==python.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1).fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1)')[5])
        assert np.all(r_lang.round(5)==python.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_Fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1)')[6])
        assert np.all(np.array(python).round(5)==r_lang.round(5))

@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_process_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1)')[8])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_parameter_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1)')[9])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    
############## Alpha = 0
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_f_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=0).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=0)')[4])
        assert np.all(r_lang.round(5)==python.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=0).fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=0)')[5])
        assert np.all(r_lang.round(5)==python.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_Fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=0).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=0)')[6])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_process_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=0).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=0)')[8])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_parameter_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=0).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=0)')[9])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    
    ####### Alpha = 1.2
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_f_a1p2(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1.2).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1.2)')[4])
        assert np.all(r_lang.round(5)==python.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_fse_a1p2(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1.2).fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1.2)')[5])
        assert np.all(r_lang.round(5)==python.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_Fse_a1p2(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1.2).Fse.iloc[:,:-1]
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1.2)')[6])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_process_risk_a1p2(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1.2).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1.2)')[8])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
    
@pytest.mark.parametrize('python_data, r_data',l3)
def test_MackChainladder_tail_parameter_risk_a1p2(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        assert True
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True, alpha=1.2).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data + ' ,tail=TRUE,alpha=1.2)')[9])
        assert np.all(np.array(python).round(5)==r_lang.round(5))
"""
