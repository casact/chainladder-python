# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
sys.path.append(r'C:\Users\jboga\OneDrive\Documents\GitHub\chainladder-python\chainladder')
print(os.getcwd())
import chainladder as cl

RAA = cl.Triangle(cl.df_list[16])
mack = cl.MackChainLadder(RAA, alpha=2)
