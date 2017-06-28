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

RAA_mack = cl.MackChainladder(cl.Triangle(RAA), tail=True)
ABC_mack = cl.MackChainladder(cl.Triangle(ABC), alpha=2)
UKMotor_mack = cl.MackChainladder(cl.Triangle(UKMotor))
GenIns_mack = cl.MackChainladder(cl.Triangle(GenIns), alpha=2)

#print(RAA_mack.summary().round(3))
#print(ABC_mack.summary().round(3))
#cl.MackChainladder(M3IR5, tail=True)


MCL_inc = cl.load_dataset('MCLincurred')
MCL_paid = cl.load_dataset('MCLpaid')