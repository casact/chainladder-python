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
    pass
       

MCL_inc = cl.load_dataset('MCLincurred')
MCL_paid = cl.load_dataset('MCLpaid')


def left_tri(df):
    return df.notnull()

if MCL_inc.shape != MCL_paid.shape: print('Paid and Incurred triangle must have same dimension.')
if len(MCL_inc) > len(MCL_inc.columns): print('MunichChainLadder does not support triangles with fewer development periods than origin periods.')
        

#def MCL(Paid, Incurred, est.sigmaP="log-linear", est.sigmaI="log-linear", tailP=FALSE, tailI=FALSE):
MackPaid = cl.MackChainladder(MCL_paid, tail=False)
MackIncurred = cl.MackChainladder(MCL_inc, tail=False)


    
modelsI=[]
modelsP=[]
for i in range(len(MCL_inc.columns)):
        modelsI.append(cl.WRTO(MCL_inc.iloc[:,i].dropna(), MCL_paid.iloc[:,i].dropna(), w=1/MCL_inc.iloc[:,i].dropna()))
        modelsP.append(cl.WRTO(MCL_paid.iloc[:,i].dropna(), MCL_inc.iloc[:,i].dropna(), w=1/MCL_paid.iloc[:,i].dropna()))

q_f = np.array([item.coefficient for item in modelsI])
qinverse_f = np.array([item.coefficient for item in modelsP])
rhoI_sigma = np.array([item.sigma for item in modelsI])
rhoP_sigma = np.array([item.sigma for item in modelsP])

y = np.log(rhoI_sigma[:-1])
x = np.array([i + 1 for i in range(len(y))])
x = sm.add_constant(x)
OLS = sm.OLS(y,x).fit()
tailsigma = np.exp((x[:,1][-1]+ 1) * OLS.params[1] + OLS.params[0])


## Estimate the residuals
paid_residual = pd.DataFrame((np.array(MackPaid.age_to_age().iloc[:-3,:-1]) 
                            - np.array(pd.DataFrame([MackPaid.f[:-1]]*(len(MCL_paid)-1)))) \
                            / np.array(pd.DataFrame([MackPaid.sigma[:-1]]*(len(MCL_paid)-1))) 
                            * np.array(np.sqrt(MCL_paid.iloc[:-1,:-1])))
incurred_residual = pd.DataFrame((np.array(MackIncurred.age_to_age().iloc[:-3,:-1]) 
                            - np.array(pd.DataFrame([MackIncurred.f[:-1]]*(len(MCL_inc)-1)))) \
                            / np.array(pd.DataFrame([MackIncurred.sigma[:-1]]*(len(MCL_inc)-1))) 
                            * np.array(np.sqrt(MCL_inc.iloc[:-1,:-1])))

q_ratios = np.array((MCL_paid/MCL_inc).iloc[:,:-1])
Q_f = np.array(pd.DataFrame([q_f[:-1]]*(len(MCL_paid))))       
q_residual = pd.DataFrame((q_ratios - Q_f) \
              / np.array(pd.DataFrame([rhoI_sigma[:-1]]*(len(MCL_inc)))) 
              * np.array(np.sqrt(MCL_inc.iloc[:,:-1])))
q_inverse_residual = pd.DataFrame((1/q_ratios - 1/Q_f) \
                        / np.array(pd.DataFrame([rhoP_sigma[:-1]]*(len(MCL_paid)))) 
                        * np.array(np.sqrt(MCL_paid.iloc[:,:-1])))


def MCL_vector(df, n):
    for i in range(n - len(df)):
        df = df.append([np.nan],ignore_index=True)
    df = df.T
    for i in range(n - len(df)):
        df = df.append([np.nan], ignore_index=True)
    for i in range(len(df)):
        df.iloc[len(df)-1-i,i] = np.nan
    return np.array(df).reshape(n*n,)

paid_residual = MCL_vector(paid_residual.iloc[:,:-1],len(MCL_paid.columns))
incurred_residual = MCL_vector(incurred_residual.iloc[:,:-1],len(MCL_paid.columns))    
q_inverse_residual = MCL_vector(q_inverse_residual.iloc[:,:-1],len(MCL_paid.columns))
q_residual = MCL_vector(q_residual.iloc[:,:-1],len(MCL_paid.columns))

  
inc_res_model = cl.WRTO(pd.Series(q_residual).dropna(), pd.Series(incurred_residual).dropna(), np.array([1 for item in pd.Series(q_residual).dropna()]))
lambdaI = inc_res_model.coefficient



paid_res_model = cl.WRTO(pd.Series(q_inverse_residual).dropna(), pd.Series(paid_residual).dropna(), np.array([1 for item in pd.Series(q_inverse_residual).dropna()]))
lambdaP = paid_res_model.coefficient
    
#Create ful triangle
full_paid = pd.DataFrame(MCL_paid.iloc[:,0])
full_incurred = pd.DataFrame(MCL_inc.iloc[:,0])
for i in range(len(MCL_paid.columns)-1):
    paid = (MackPaid.f[i] + lambdaP*MackPaid.sigma[i] /rhoP_sigma[i]*(full_incurred.iloc[:,-1]/full_paid.iloc[:,-1] - qinverse_f[i]))*full_paid.iloc[:,-1]
    inc = (MackIncurred.f[i] + lambdaI*MackIncurred.sigma[i] /rhoI_sigma[i]*(full_paid.iloc[:,-1]/full_incurred.iloc[:,-1] - q_f[i]))*full_incurred.iloc[:,-1]
    full_incurred[str(i+2)] = MCL_inc.iloc[:,i+1].fillna(inc)
    full_paid[str(i+2)] = MCL_paid.iloc[:,i+1].fillna(paid)
