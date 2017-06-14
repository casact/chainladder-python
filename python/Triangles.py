# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os.path


path = os.path.abspath('')[:-6] + 'data\\'
RAA = pd.read_pickle(path + 'RAA')



def cum2incr(triangle):
    incr = triangle.iloc[:,0]
    for val in range(1, len(triangle.T.index)):
        incr = pd.concat([incr,triangle.iloc[:,val]-triangle.iloc[:,val-1]],axis=1)
    incr = incr.rename_axis('dev', axis='columns')
    incr.columns = triangle.T.index
    return incr

        
def incr2cum(triangle):
    incr = pd.DataFrame(triangle.iloc[:,0])
    for val in range(1, len(triangle.T.index)):
        incr = pd.concat([incr,triangle.iloc[:,val]+incr.iloc[:,-1]],axis=1)
    incr = incr.rename_axis('dev', axis='columns')
    incr.columns = triangle.T.index
    return incr 

def aslongtriangle(triangle):
    # will need to create triangle class that has origin and dev
    lx = pd.DataFrame()
    for val in range(len(triangle.T.index)):
        df = pd.DataFrame(triangle.iloc[:,val].rename('value'))
        df['dev']= int(triangle.iloc[:,val].name)
        lx = lx.append(df)
    lx.dropna(inplace=True)
    return lx[['dev','value']]

def astriangle(lx):
    triangle = pd.pivot_table(lx,values='value',index='origin', columns='dev')
    triangle.columns = [str(item) for item in triangle.columns]
    return triangle
        
      
  
  
  