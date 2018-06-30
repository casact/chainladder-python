# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:58:03 2017

@author: jboga
"""
import chainladder as cl
import chainladder.deterministic as cld
import copy
import pytest
import os.path
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
import numpy as np

from numpy.testing import assert_allclose, assert_equal

pandas2ri.activate()
CL = importr('ChainLadder')
d = r('data(package=\"ChainLadder\")')

ATOL = 1e-5

DATA_DIR = os.listdir(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
l1 = [cl.load_dataset(item) for item in DATA_DIR]
l2 = [item for item in DATA_DIR]
l3 = list(zip(l1, l2))

@pytest.mark.parametrize('df', l1)
def test_cum_to_incr_convert(df):
    if df.shape[0] > df.shape[1]:
        return
    a = cl.Triangle(df)
    data1 = a.data
    a.cum_to_incr(inplace=True)
    a.incr_to_cum(inplace=True)
    test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
    assert_equal(test.shape[0]*test.shape[1], np.sum(test))


@pytest.mark.parametrize('df', l1)
def test_incr_to_cum_convert(df):
    if df.shape[0] > df.shape[1]:
        return
    a = cl.Triangle(df)
    data1 = a.data
    a.incr_to_cum(inplace=True)
    a.cum_to_incr(inplace=True)
    test = ((data1 == a.data) | ((data1 != data1) & (a.data != a.data))).as_matrix()
    assert_equal(test.shape[0]*test.shape[1], np.sum(test))


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_f(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=2).f
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[4])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=2).fse[:-1]
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[5])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_Fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=2).Fse
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[6])
        assert_allclose(python, r_lang)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_process_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(
            cl.Triangle(python_data), alpha=2).process_risk
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[8])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_parameter_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(
                cl.Triangle(python_data), alpha=2).parameter_risk
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=2)')[9])
        np.allclose(python, r_lang, atol=ATOL)


# alpha = 1
@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_f_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=1).f
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[4])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=1).fse[:-1]
        r_lang = np.array(
            r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[5])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_Fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=1).Fse
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[6])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_process_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(
                cl.Triangle(python_data), alpha=1).process_risk
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[8])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_parameter_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),
                                    alpha=1).parameter_risk
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=1)')[9])
        assert_allclose(python, r_lang, atol=ATOL)


# Alpha = 0
@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_f_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=0).f
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[4])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=0).fse[:-1]
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[5])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_Fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), alpha=0).Fse
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[6])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_process_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),
                                    alpha=0).process_risk
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[8])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_parameter_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),
                                    alpha=0).parameter_risk
        r_lang = np.array(
                r('mack<-MackChainLadder(' + r_data + ',alpha=0)')[9])
        assert_allclose(python, r_lang, atol=ATOL)


def test_MunichChainladder_1():
    python = cl.MunichChainladder(cl.load_dataset('MCLpaid'),
                                  cl.load_dataset('MCLincurred'))
    r_lang = np.array(r('a<-MunichChainLadder(MCLpaid,MCLincurred)'))
    assert_allclose(python.lambdaI, np.array(r_lang[12][0])[0], atol=ATOL)
    assert_allclose(python.lambdaP, np.array(r_lang[11][0])[0], atol=ATOL)
    assert_allclose(python.q_f, r_lang[15], atol=ATOL)


# Try with exponential tail
@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_f(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data),
                                    tail=True, alpha=2).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=2)')[4])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=2)
        if not python.chainladder.tail:
            python = python.fse[:-1]
        else:
            python = python.fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=2)')[5])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_Fse(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=2).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=2)')[6])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_process_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=2).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=2)')[8])
        if r_data == 'GenIns':
            ATOL = 1e-3    # GenIns rounding is so close to 5, that it fails
        elif r_data in ['liab$AutoLiab', 'Mortgage']:
            ATOL = 1e-4
        else:
            ATOL = 1e-5
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_parameter_risk(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=2).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ',tail=TRUE,alpha=2)')[9])
        assert_allclose(python, r_lang, atol=ATOL)


# alpha = 1
@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_f_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=1).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=1)')[4])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=1)
        if not python.chainladder.tail:
            python = python.fse[:-1]
        else:
            python = python.fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=1)')[5])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_Fse_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=1).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=1)')[6])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_process_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=1).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=1)')[8])
        if r_data == 'GenIns':
            ATOL = 1e-3    # GenIns rounding is so close to 5, that it fails
        elif r_data in ['liab$AutoLiab', 'Mortgage']:
            ATOL = 1e-4
        else:
            ATOL = 1e-5
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_parameter_risk_a1(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=1).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=1)')[9])
        assert_allclose(python, r_lang, atol=ATOL)


# Alpha = 0
@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_f_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=0).f
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=0)')[4])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=0)
        if not python.chainladder.tail:
            python = python.fse[:-1]
        else:
            python = python.fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=0)')[5])
        assert_allclose(r_lang, python, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_Fse_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=0).Fse
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=0)')[6])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_process_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=0).process_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=0)')[8])
        assert_allclose(python, r_lang, atol=ATOL)


@pytest.mark.parametrize('python_data, r_data', l3)
def test_MackChainladder_tail_parameter_risk_a0(python_data, r_data):
    # R MackChainladder does not work on quarterly triangle, so bypassing
    # Also eliminating GenInsLong as R MackChainladder fails on tabular form
    # data
    if r_data in ['qpaid', 'qincurred', 'GenInsLong',
                  'auto$PersonalAutoIncurred', 'MCLincurred']:
        return
    else:
        python = cl.MackChainladder(cl.Triangle(python_data), tail=True,
                                    alpha=0).parameter_risk
        r_lang = np.array(r('mack<-MackChainLadder(' + r_data +
                            ' ,tail=TRUE,alpha=0)')[9])
        assert_allclose(python, r_lang, atol=ATOL)

def test_triangle_subtract():
    a = cl.Triangle(cl.load_dataset('RAA'))
    assert_equal(np.sum(np.nan_to_num(np.array((a - a).data))), 0)

def test_triangle_tabular():
    assert_equal(cl.Triangle(cl.load_dataset('GenInsLong'), origin='accyear', development='devyear', values='incurred claims').data.shape,(10,10))

def test_triangle_OYDY_grain():
    assert_equal(cl.Triangle(cl.load_dataset('qincurred')).grain('OYDY').data.shape,(12,12))

def test_DFM_deterministic():
    answer = np.array([18834, 16857, 24083, 28703, 28926, 19501, 17749, 24019, 16044, 18402])
    RAA = cl.load_dataset('RAA')
    question = np.array(cld.Chainladder(RAA).ultimates.astype(int))
    assert_equal(question, answer)

def test_BF_deterministic():
    answer = np.array([18834, 16886, 23978, 28207, 28079, 19594, 18438, 22194, 18670, 19820])
    RAA = cl.load_dataset('RAA')
    exp = answer*0+40000
    question = np.array(cld.Chainladder(RAA).born_ferg(exposure = exp, apriori = 0.5).ultimates.astype(int))
    assert_equal(question, answer)

def test_benktander_deterministic_1():
    RAA = cl.load_dataset('RAA')
    exp = np.array([40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000, 40000])
    answer = np.array(cld.Chainladder(RAA).born_ferg(exposure = exp, apriori = 0.5).ultimates.astype(int))
    question = np.array(cld.Chainladder(RAA).benktander(exposure = exp, apriori = 0.5, n_iters=1).ultimates.astype(int))
    assert_equal(question, answer)

def test_benktander_deterministic_999():
    RAA = cl.load_dataset('RAA')
    answer = np.array(cld.Chainladder(RAA).ultimates.astype(int))
    exp = answer*0+40000
    question = np.array(cld.Chainladder(RAA).benktander(exposure = exp, apriori = 0.5, n_iters=999).ultimates.astype(int))
    assert_equal(question, answer)

def test_benktander_deterministic_3():
    answer = np.array([18834, 16857, 24083, 28701, 28919, 19504, 17813, 23642, 17201, 19520])
    RAA = cl.load_dataset('RAA')
    exp = answer*0+40000
    question = np.array(cld.Chainladder(RAA).benktander(exposure = exp, apriori = 0.5, n_iters=3).ultimates.astype(int))
    assert_equal(question, answer)

def test_benktander_deterministic_3_vol5():
    answer = np.array([18834, 16857, 24083, 28701, 28919, 19504, 17866, 23287, 17937, 21562])
    RAA = cl.load_dataset('RAA')
    exp = answer*0+40000
    question = np.array(cld.Chainladder(RAA).select_average(num_periods=5).benktander(exposure = exp, apriori = 0.5, n_iters=3).ultimates.astype(int))
    assert_equal(question, answer)

def test_benktander_deterministic_3_custom():
    answer = np.array([18834, 16857, 24083, 28701, 28919, 19504, 17866, 23626, 19141, 21952])
    avg_type = ['volume','simple','volume','volume', 'volume','volume','volume','volume','volume']
    RAA = cl.load_dataset('RAA')
    nper = [5, 5, 4, 5, 5, 5, 5, 5, 5]
    exp = answer*0+40000
    question = np.array(cld.Chainladder(RAA).select_average(LDF_average=avg_type, num_periods=nper).exclude_link_ratios(otype=[(7,1)]).benktander(exposure = exp, apriori = 0.5, n_iters=3).ultimates.astype(int))
    assert_equal(question, answer)

def test_cape_cod_1():
    # trend = 0 and decay = 1 -> apriori to be used in BF, where BF and cape cod will be equal
    RAA = cl.Triangle(cl.load_dataset('RAA'))
    exp = copy.deepcopy(RAA.data.iloc[:,0])*0+40000
    question = cld.Chainladder(RAA).cape_cod(trend=0,decay=1,exposure=exp).ultimates
    apriori = sum(question)/sum(exp)
    answer = cld.Chainladder(RAA).born_ferg(exposure=exp, apriori=apriori).ultimates.astype(int)
    assert_equal(np.array(question.astype(int)) , np.array(answer))

def test_triangle_mult_div():
    RAA = cl.load_dataset('RAA')
    assert_equal(np.nan_to_num(np.array((cl.Triangle(RAA)/cl.Triangle(RAA)*cl.Triangle(RAA)).data)),
    np.nan_to_num(np.array(cl.Triangle(RAA).data)))

def test_triangle_add_subtract():
    RAA = cl.load_dataset('RAA')
    assert_equal(np.nan_to_num(np.array((cl.Triangle(RAA)+cl.Triangle(RAA)-cl.Triangle(RAA)).data)),
    np.nan_to_num(np.array(cl.Triangle(RAA).data)))
