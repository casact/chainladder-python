# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 07:08:08 2017

@author: jboga
"""

from pandas import DataFrame, Series, concat, pivot_table, read_pickle
import matplotlib.pyplot as plt
#import seaborn as sns
import os


print('Welcome to chainladder')
__all__ = ['Triangles', 'ChainLadder', 'MackChainLadderFunctions']

#sns.set_style("whitegrid")

#from chainladder.TestCases import df_list
#from chainladder.classes import Triangle 
#from chainladder.classes import ChainLadder 
#from chainladder.classes import MackChainLadder
