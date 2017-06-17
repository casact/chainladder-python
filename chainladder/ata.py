# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 08:30:04 2017

@author: jboga
"""

def ata(tri, colname_sep = '-'):
    incr = DataFrame(tri.data.iloc[:,1]/tri.data.iloc[:,0])
    for i in range(1, len(tri.data.columns)-1):
        incr = concat([incr,tri.data.iloc[:,i+1]/tri.data.iloc[:,i]],axis=1)
    incr.columns = [item + colname_sep + tri.data.columns.values[num+1] for num, item in enumerate(tri.data.columns.values[:-1])]
    incr = incr.iloc[:-1]
    ldf = [item.coef_ for item in chainladder(tri, delta=2).models]
    incr.loc['smpl']=ldf
    ldf = [item.coef_ for item in chainladder(tri, delta=1).models]
    incr.loc['vwtd']=ldf
    return incr.round(3)