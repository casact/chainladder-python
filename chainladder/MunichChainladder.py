# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:21:15 2017

@author: jboga
"""
import chainladder as cl
import numpy as np
import pandas as pd
import statsmodels.api as sm

class MunichChainladder:
    def __init__(self, Paid, Incurred, tailP=False, tailI=False):
        self.Paid = Paid
        self.Incurred = Incurred
        self.tailP = tailP
        self.tailI = tailI
        self.MackPaid = cl.MackChainladder(self.Paid, tail=self.tailP)
        self.MackIncurred = cl.MackChainladder(self.Incurred, tail=self.tailI)
        MCL_model = self.__get_MCL_model()
        self.rhoI_sigma = MCL_model[0]
        self.rhoP_sigma = MCL_model[1]
        self.q_f = MCL_model[2]
        self.qinverse_f = MCL_model[3]
        MCL_residuals = self.__get_MCL_residuals()
        self.paid_residuals =  MCL_residuals[0]
        self.incurred_residuals = MCL_residuals[1]
        self.Q_inverse_residuals = MCL_residuals[2]
        self.Q_residuals =  MCL_residuals[3]
        MCL_lambda = self.__get_MCL_lambda()
        self.lambdaP = MCL_lambda[0]
        self.lambdaI = MCL_lambda[1]
        MCL_full_triangle = self.__get_MCL_full_triangle()
        self.MCL_paid = MCL_full_triangle[0]
        self.MCL_incurred = MCL_full_triangle[1]

#if self.Incurred.shape != self.Paid.shape: print('Paid and Incurred triangle must have same dimension.')
#if len(self.Incurred) > len(self.Incurred.columns): print('MunichChainLadder does not support triangles with fewer development periods than origin periods.')

    def __MCL_vector(self, df, n):
        for i in range(n - len(df)):
            df = df.append([np.nan],ignore_index=True)
        df = df.T
        for i in range(n - len(df)):
            df = df.append([np.nan], ignore_index=True)
        for i in range(len(df)):
            df.iloc[len(df)-1-i,i] = np.nan
        return np.array(df).reshape(n*n,)        

    def __get_MCL_model(self): 
        modelsI=[]
        modelsP=[]
        for i in range(len(self.Incurred.columns)):
                modelsI.append(cl.WRTO(self.Incurred.iloc[:,i].dropna(), self.Paid.iloc[:,i].dropna(), w=1/self.Incurred.iloc[:,i].dropna()))
                modelsP.append(cl.WRTO(self.Paid.iloc[:,i].dropna(), self.Incurred.iloc[:,i].dropna(), w=1/self.Paid.iloc[:,i].dropna()))       
        q_f = np.array([item.coefficient for item in modelsI])
        qinverse_f = np.array([item.coefficient for item in modelsP])
        rhoI_sigma = np.array([item.sigma for item in modelsI])
        rhoP_sigma = np.array([item.sigma for item in modelsP])
        #y = np.log(rhoI_sigma[:-1])
        #x = np.array([i + 1 for i in range(len(y))])
        #x = sm.add_constant(x)
        #OLS = sm.OLS(y,x).fit()
        #tailsigma = np.exp((x[:,1][-1]+ 1) * OLS.params[1] + OLS.params[0])
        return rhoI_sigma, rhoP_sigma, q_f, qinverse_f
        
    def __get_MCL_residuals(self):    
        ## Estimate the residuals
        paid_residual = pd.DataFrame((np.array(self.MackPaid.age_to_age().iloc[:-3,:-1]) 
                                    - np.array(pd.DataFrame([self.MackPaid.f[:-1]]*(len(self.Paid)-1)))) \
                                    / np.array(pd.DataFrame([self.MackPaid.sigma[:-1]]*(len(self.Paid)-1))) 
                                    * np.array(np.sqrt(self.Paid.iloc[:-1,:-1])))
        incurred_residual = pd.DataFrame((np.array(self.MackIncurred.age_to_age().iloc[:-3,:-1]) 
                                    - np.array(pd.DataFrame([self.MackIncurred.f[:-1]]*(len(self.Incurred)-1)))) \
                                    / np.array(pd.DataFrame([self.MackIncurred.sigma[:-1]]*(len(self.Incurred)-1))) 
                                    * np.array(np.sqrt(self.Incurred.iloc[:-1,:-1])))
        
        Q_ratios = np.array((self.Paid/self.Incurred).iloc[:,:-1])
        Q_f = np.array(pd.DataFrame([self.q_f[:-1]]*(len(self.Paid))))       
        Q_residual = pd.DataFrame((Q_ratios - Q_f) \
                      / np.array(pd.DataFrame([self.rhoI_sigma[:-1]]*(len(self.Incurred)))) 
                      * np.array(np.sqrt(self.Incurred.iloc[:,:-1])))
        Q_inverse_residual = pd.DataFrame((1/Q_ratios - 1/Q_f) \
                                / np.array(pd.DataFrame([self.rhoP_sigma[:-1]]*(len(self.Paid)))) 
                                * np.array(np.sqrt(self.Paid.iloc[:,:-1])))

        paid_residual = self.__MCL_vector(paid_residual.iloc[:,:-1],len(self.Paid.columns))
        incurred_residual = self.__MCL_vector(incurred_residual.iloc[:,:-1],len(self.Paid.columns))    
        Q_inverse_residual = self.__MCL_vector(Q_inverse_residual.iloc[:,:-1],len(self.Paid.columns))
        Q_residual = self.__MCL_vector(Q_residual.iloc[:,:-1],len(self.Paid.columns))
        return paid_residual, incurred_residual, Q_inverse_residual, Q_residual

    def __get_MCL_lambda(self):  
        inc_res_model = cl.WRTO(pd.Series(self.Q_residuals).dropna(), pd.Series(self.incurred_residuals).dropna(), np.array([1 for item in pd.Series(self.Q_residuals).dropna()]))
        lambdaI = inc_res_model.coefficient
        paid_res_model = cl.WRTO(pd.Series(self.Q_inverse_residuals).dropna(), pd.Series(self.paid_residuals).dropna(), np.array([1 for item in pd.Series(self.Q_inverse_residuals).dropna()]))
        lambdaP = paid_res_model.coefficient
        return lambdaP, lambdaI

    def __get_MCL_full_triangle(self):    
        full_paid = pd.DataFrame(self.Paid.iloc[:,0])
        full_incurred = pd.DataFrame(self.Incurred.iloc[:,0])
        for i in range(len(self.Paid.columns)-1):
            paid = (self.MackPaid.f[i] + self.lambdaP*self.MackPaid.sigma[i] /self.rhoP_sigma[i]*(full_incurred.iloc[:,-1]/full_paid.iloc[:,-1] - self.qinverse_f[i]))*full_paid.iloc[:,-1]
            inc = (self.MackIncurred.f[i] + self.lambdaI*self.MackIncurred.sigma[i] /self.rhoI_sigma[i]*(full_paid.iloc[:,-1]/full_incurred.iloc[:,-1] - self.q_f[i]))*full_incurred.iloc[:,-1]
            full_incurred[str(i+2)] = self.Incurred.iloc[:,i+1].fillna(inc)
            full_paid[str(i+2)] = self.Paid.iloc[:,i+1].fillna(paid)
        return full_paid, full_incurred
