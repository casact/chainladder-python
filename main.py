# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:19:58 2017

@author: jboga
"""

import chainladder as cl

RAA = cl.load_dataset('RAA')
ABC = cl.load_dataset('ABC')
UKMotor = cl.load_dataset('UKMotor')
GenIns = cl.load_dataset('GenIns')

RAA_mack = cl.MackChainladder(cl.Triangle(RAA))
ABC_mack = cl.MackChainladder(cl.Triangle(ABC), alpha=2, tail=True)
UKMotor_mack = cl.MackChainladder(cl.Triangle(UKMotor))
GenIns_mack = cl.MackChainladder(cl.Triangle(GenIns), alpha=2, tail=True)

#print(RAA_mack.summary().round(3))
#print(ABC_mack.summary().round(3))
#cl.MackChainladder(M3IR5, tail=True)


MCL_inc = cl.load_dataset('MCLincurred')
MCL_paid = cl.load_dataset('MCLpaid')

MCL = cl.MunichChainladder(MCL_paid, MCL_inc)
BS = cl.BootChainladder(cl.Triangle(RAA),n_sims=1000, process_distr="od poisson")

#CAS = pd.read_csv(r'http://www.casact.org/research/reserve_data/wkcomp_pos.csv')
#CAS['INC_LOSS'] = CAS['IncurLoss_D']-CAS['BulkLoss_D']
#large = (pd.pivot_table(CAS, values=['EarnedPremNet_D'],index=['GRCODE'])>1000)
#large = list(large[large['EarnedPremNet_D']== True].index)


def MCLArray(a):
    import pandas as pd
    import numpy as np
    b = pd.pivot_table(a,values=['CumPaidLoss_D'],index=['AccidentYear'],columns=['DevelopmentLag'], aggfunc=np.sum)
    b.columns = b.columns.droplevel(0)
    b.columns = [str(item) for item in b.columns]
    Paid=np.array(b)*np.array([np.append(np.array([1]*(len(b)-i)),np.array([np.nan]*i)) for i in range(len(b))])*b
    
    b = pd.pivot_table(a,values=['INC_LOSS'],index=['AccidentYear'],columns=['DevelopmentLag'], aggfunc=np.sum)
    b.columns = b.columns.droplevel(0)
    Incurred=np.array(b)*np.array([np.append(np.array([1]*(len(b)-i)),np.array([np.nan]*i)) for i in range(len(b))])*b
    return cl.MunichChainladder(Paid,Incurred)