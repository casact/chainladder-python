# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:19:58 2017

@author: jboga
"""

import pandas as pd
import numpy as np

import chainladder as cl

RAA = cl.datasets.load_dataset('RAA')
ABC = cl.datasets.load_dataset('ABC')

RAA_mack = cl.MackChainLadder(cl.Triangle(RAA), alpha=2)
ABC_mack = cl.MackChainLadder(cl.Triangle(ABC), alpha=2)

print(RAA_mack.summary().round(3))
print(ABC_mack.summary().round(3))
