# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 06:35:12 2017

@author: jboga
"""
import numpy as np
import scipy
from chainladder.Triangle import Triangle
import copy

class ClarkLDF:
    def __init__(self, triangle, cumulative = True, maxage = np.inf, adol = True, adol_age = None, origin_width = None, G = "loglogistic"):
        self.triangle = triangle.data_as_triangle()
        self.origin = self.triangle.data.index
        self.cols = self.triangle.data.index
        self.nr = len(self.origin)
        self.dev = np.array([int(item) for item in self.cols])
        self.age_to = self.dev
        self.age_diff = np.array(self.age_to)[1:]-np.array(self.age_to)[:-1]
        self.origin_width = np.mean(self.age_diff)
        if adol_age == None:
            self.adol_age = self.origin_width / 2
        self.early_age = self.age_to < self.origin_width
        self.age_to = np.invert(self.early_age)*(self.age_to-self.adol_age)
        self.triangle.data.columns = self.age_to
        self.maxage_used = maxage - self.adol_age
        self.age_from = np.append([0],self.age_to[:-1])
        self.CurrentValue = triangle.get_latest_diagonal()
        self.Uinit = self.CurrentValue
        self.magscale = max(self.CurrentValue)
        self.triangle.data = self.triangle.data/self.magscale
        self.CurrentValue = self.CurrentValue / self.magscale
        self.Uinit = self.Uinit / self.magscale
        self.CurrentAge = np.array(self.triangle.data.columns[::-1])
        self.CurrentAge_from = self.age_from[::-1]
        self.CurrentAge_to = self.age_to[::-1]
        temp = copy.deepcopy(triangle)
        temp.index = [item+ 1 for item in range(self.nr)]
        temp.columns = [item+1 for item in range(temp.ncol)]
        af_dict = dict(zip(temp.columns, self.age_from))
        at_dict = dict(zip(temp.columns, self.age_to))
        self.table_one_one = temp.data_as_table().data
        self.table_one_one['Age_from'] =self.table_one_one['dev_lag'].map(af_dict)
        self.table_one_one['Age_to'] = self.table_one_one['dev_lag'].map(at_dict)
        self.theta = copy.deepcopy(Uinit)
        self.theta.loc['omega']=2
        self.theta.loc['theta']=np.median(self.table_one_one['Age_to'])

class loglogistic:
    def __init__(self):
        from sympy.abc import x, theta, omega
        self.cdf = (1 + (x/theta)**(-omega))**-1
        self.G = 1 / self.cdf
        self.pdf = sympy.simplify(sympy.diff(self.cdf,x))
        self.dGd
        
        