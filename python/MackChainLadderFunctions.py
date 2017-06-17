# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:24:42 2017

@author: jboga
"""
import pandas as pd

class MackChainLadder:
    def __init__(self, tri,  weights=1,  alpha=1,  est_sigma="log-linear",  
                 tail=False,  tail_se=None,  tail_sigma=None,  mse_method = "Mack"):
        delta = 2-alpha
        cl = chainladder(tri, weights=weights,  delta=delta)
        alpha = [2 - item for item in cl.delta]
        
        self.triangle = tri
        self.weights = cl.weights
        self.fullTriangle = cl.predict()
        self.alpha = alpha
        self.f = np.array([item.coef_ for item in cl.models])
        self.fse = np.array([item.se for item in cl.models])
        self.sigma = np.array([item.sigma for item in cl.models])
        self.Fse = self.Fse()

        


            
        
# "Mack.S.E"]] <- sqrt(StdErr$FullTriangle.procrisk^2 + StdErr$FullTriangle.paramrisk^2)

    def process_risk(self):
        procrisk = pd.DataFrame([0 for item in range(len(self.fullTriangle))],index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1,len(self.fullTriangle.columns)):
            print(i)
            temp = pd.DataFrame(np.sqrt((self.fullTriangle.iloc[:,i-1]*self.Fse.iloc[:,i-1])**2 + (self.f[i-1]*procrisk.iloc[:,i-1])**2)*self.triangle.data.iloc[:,i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            procrisk = pd.concat([procrisk, temp],axis=1)
        return procrisk

    def parameter_risk(self):
        paramrisk = pd.DataFrame([0 for item in range(len(self.fullTriangle))],index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1,len(self.fullTriangle.columns)):
            print(i)
            temp = pd.DataFrame(np.sqrt((self.fullTriangle.iloc[:,i-1]*self.fse[i-1])**2 + (self.f[i-1]*paramrisk.iloc[:,i-1])**2)*self.triangle.data.iloc[:,i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            paramrisk = pd.concat([paramrisk, temp],axis=1)
        return paramrisk
    
    def Fse(self):
        # This is sloppy, and I don't know that it works for all cases.  Need to
        # understand weights better.
        fulltriangleweightconst = self.weights.data.mode().T.mode().iloc[0,0]
        fulltriangleweight = self.fullTriangle*0 + fulltriangleweightconst
        Fse = pd.DataFrame()
        for i in range(fullTriangle.shape[1]-2):
            Fse = pd.concat([Fse, pd.DataFrame(self.sigma[i]/np.sqrt(fulltriangleweight.iloc[:,i]*self.fullTriangle.iloc[:,i]**alpha[i]))],axis=1)
            #print(self.sigma[i]/np.sqrt(fulltriangleweight.iloc[:,i]*self.fullTriangle.iloc[:,i]**alpha[i]))
            
        #Fse = pd.DataFrame(np.array(Fse).T)
        Fse.set_index(self.fullTriangle.index, inplace = True)
        #Fse.columns=fullTriangle.columns[:-1].tolist()
        return Fse
    