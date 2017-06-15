# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:09:27 2017

@author: jboga
"""
import pandas as pd
#class ChainLadder:

    
def get_ata_tri(tri):
    incr = pd.DataFrame(tri.data.iloc[:,1]/tri.data.iloc[:,0])
    for i in range(1, len(tri.data.columns)-1):
        incr = pd.concat([incr,tri.data.iloc[:,i+1]/tri.data.iloc[:,i]],axis=1)
    incr.columns = [item + '-' + tri.data.columns.values[num+1] for num, item in enumerate(tri.data.columns.values[:-1])]
    return incr.iloc[:-1]

def get_tail(tri):
    return pd.Series([1], index = [tri.data.iloc[:,-1].name + '-Ult'])

def get_ldf(tri):
    ata = get_ata_tri(tri)
    ldf = ata.describe().iloc[1]
    ldf = ldf.append(get_tail(tri))
    ldf.name = 'ldf'
    return ldf
   
def get_cdf(tri):
    ldf = get_ldf(tri)
    cdf = ldf[::-1].cumprod().iloc[::-1]
    return cdf

def get_latest_diag(tri):
    latest = pd.DataFrame([[int(tri.data.iloc[0].dropna().index[-1]), tri.data.iloc[0].dropna()[-1]]], columns=['dev','values'])
    for i in range(len(tri.data)-1):
        latest = latest.append(pd.DataFrame([[int(tri.data.iloc[i+1].dropna().index[-1]), tri.data.iloc[i+1].dropna()[-1]]], columns=['dev','values']))
    latest.index=[tri.data.index]
    return latest
        
def get_ult(tri):
    cdf_iloc = [int(item[:item.find('-')]) for item in get_cdf(tri).index.values]
    latest = get_latest_diag(tri)
    cdf = get_cdf(tri)
    relevant_cdf = cdf.iloc[[cdf_iloc.index(item) for item in latest['dev']]]
    latest['CDF'] = relevant_cdf.values
    latest['Ultimate'] = latest['values'].values * relevant_cdf.values
    latest.rename(columns={'values':'Latest'}, inplace=True)
    return latest.drop(['dev'], axis=1)

