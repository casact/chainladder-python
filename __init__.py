# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 07:08:08 2017

@author: jboga
"""

from pandas import DataFrame, Series, concat, pivot_table, read_pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
import os
print('Welcome to chainladder')
__all__ = ['TestCases', 'Triangles', 'ChainLadder', 'MackChainLadderFunctions', 'ata']

sns.set_style("whitegrid")

from chainladder.TestCases import df_list
from chainladder.Triangles import triangle 
from chainladder.ChainLadder import chainladder 
from chainladder.MackChainLadderFunctions import MackChainLadder
