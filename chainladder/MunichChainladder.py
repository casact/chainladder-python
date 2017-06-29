# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:21:15 2017

@author: jboga
"""
import chainladder as cl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from chainladder.UtilityFunctions import Plot

class MunichChainladder:
    def __init__(self, Paid, Incurred, tailP=False, tailI=False):
        self.tailP = tailP
        self.tailI = tailI
        self.MackPaid = cl.MackChainladder(Paid, tail=self.tailP)
        self.MackIncurred = cl.MackChainladder(Incurred, tail=self.tailI)
        self.Paid = self.MackPaid.triangle
        self.Incurred = self.MackIncurred.triangle
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

    def __repr__(self):   
        return str(self.summary())
    
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
        for i in range(len(self.Incurred.data.columns)):
                modelsI.append(cl.WRTO(self.Incurred.data.iloc[:,i].dropna(), self.Paid.data.iloc[:,i].dropna(), w=1/self.Incurred.data.iloc[:,i].dropna()))
                modelsP.append(cl.WRTO(self.Paid.data.iloc[:,i].dropna(), self.Incurred.data.iloc[:,i].dropna(), w=1/self.Paid.data.iloc[:,i].dropna()))       
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
                                    - np.array(pd.DataFrame([self.MackPaid.f[:-1]]*(len(self.Paid.data)-1)))) \
                                    / np.array(pd.DataFrame([self.MackPaid.sigma[:-1]]*(len(self.Paid.data)-1))) 
                                    * np.array(np.sqrt(self.Paid.data.iloc[:-1,:-1])))
        incurred_residual = pd.DataFrame((np.array(self.MackIncurred.age_to_age().iloc[:-3,:-1]) 
                                    - np.array(pd.DataFrame([self.MackIncurred.f[:-1]]*(len(self.Incurred.data)-1)))) \
                                    / np.array(pd.DataFrame([self.MackIncurred.sigma[:-1]]*(len(self.Incurred.data)-1))) 
                                    * np.array(np.sqrt(self.Incurred.data.iloc[:-1,:-1])))
        
        Q_ratios = np.array((self.Paid.data/self.Incurred.data).iloc[:,:-1])
        Q_f = np.array(pd.DataFrame([self.q_f[:-1]]*(len(self.Paid.data))))       
        Q_residual = pd.DataFrame((Q_ratios - Q_f) \
                      / np.array(pd.DataFrame([self.rhoI_sigma[:-1]]*(len(self.Incurred.data)))) 
                      * np.array(np.sqrt(self.Incurred.data.iloc[:,:-1])))
        Q_inverse_residual = pd.DataFrame((1/Q_ratios - 1/Q_f) \
                                / np.array(pd.DataFrame([self.rhoP_sigma[:-1]]*(len(self.Paid.data)))) 
                                * np.array(np.sqrt(self.Paid.data.iloc[:,:-1])))

        paid_residual = self.__MCL_vector(paid_residual.iloc[:,:-1],len(self.Paid.data.columns))
        incurred_residual = self.__MCL_vector(incurred_residual.iloc[:,:-1],len(self.Paid.data.columns))    
        Q_inverse_residual = self.__MCL_vector(Q_inverse_residual.iloc[:,:-1],len(self.Paid.data.columns))
        Q_residual = self.__MCL_vector(Q_residual.iloc[:,:-1],len(self.Paid.data.columns))
        return paid_residual, incurred_residual, Q_inverse_residual, Q_residual

    def __get_MCL_lambda(self):  
        inc_res_model = cl.WRTO(pd.Series(self.Q_residuals).dropna(), pd.Series(self.incurred_residuals).dropna(), np.array([1 for item in pd.Series(self.Q_residuals).dropna()]))
        lambdaI = inc_res_model.coefficient
        paid_res_model = cl.WRTO(pd.Series(self.Q_inverse_residuals).dropna(), pd.Series(self.paid_residuals).dropna(), np.array([1 for item in pd.Series(self.Q_inverse_residuals).dropna()]))
        lambdaP = paid_res_model.coefficient
        return lambdaP, lambdaI

    def __get_MCL_full_triangle(self):    
        full_paid = pd.DataFrame(self.Paid.data.iloc[:,0])
        full_incurred = pd.DataFrame(self.Incurred.data.iloc[:,0])
        for i in range(len(self.Paid.data.columns)-1):
            paid = (self.MackPaid.f[i] + self.lambdaP*self.MackPaid.sigma[i] /self.rhoP_sigma[i]*(full_incurred.iloc[:,-1]/full_paid.iloc[:,-1] - self.qinverse_f[i]))*full_paid.iloc[:,-1]
            inc = (self.MackIncurred.f[i] + self.lambdaI*self.MackIncurred.sigma[i] /self.rhoI_sigma[i]*(full_paid.iloc[:,-1]/full_incurred.iloc[:,-1] - self.q_f[i]))*full_incurred.iloc[:,-1]
            full_incurred[str(i+2)] = self.Incurred.data.iloc[:,i+1].fillna(inc)
            full_paid[str(i+2)] = self.Paid.data.iloc[:,i+1].fillna(paid)
        return full_paid, full_incurred
    
    def summary(self):
        """ Method to produce a summary table of of the Munich Chainladder 
        model.

        Returns:
            This calculation is consistent with the R calculation 
            MunichChainLadder$summary
        """
        summary = pd.DataFrame()
        summary['Latest Paid'] = self.Paid.get_latest_diagonal()
        summary['Latest Incurred'] = self.Incurred.get_latest_diagonal()
        summary['Latest P/I Ratio'] = summary['Latest Paid']/summary['Latest Incurred'] 
        summary['Ult. Paid'] = self.MCL_paid.iloc[:,-1]
        summary['Ult. Incurred'] = self.MCL_incurred.iloc[:,-1]
        summary['P/I Ratio'] = summary['Ult. Paid']/summary['Ult. Incurred'] 
        return summary
    
    def plot(self, plots=['summary', 'MCLvsMack', 'resid1', 'resid2']):
        """ Method that is designed to configure the matplotlib graphs for the
        MunichChainLadder class using a dictionary and then calls on the Plot 
        class from UtilityFunctions.
        
        Returns:
            Returns a dictionary containing the configuration of the selected plot.
        """
    
        my_dict = []
        plot_dict = self.__get_plot_dict()
        for item in plots:
            my_dict.append(plot_dict[item])
        Plot(my_dict)
    
    def __get_plot_dict(self):
        plot_dict = {'summary':{'Title':'Munich Chainladder Results',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Ultimates',
                                     'chart_type_dict':{'type':['bar','bar'],
                                                       'height':[self.summary()['Ult. Paid'],self.summary()['Ult. Incurred']],
                                                       'left':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0]
                                                       }},
                     'MCLvsMack':{'Title':'Munich Chainladder vs. Standard Chainladder',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Ultimates',
                                     'chart_type_dict':{'type':['bar','bar'],
                                                       'height':[self.summary()['Ult. Paid']/self.summary()['Ult. Incurred'],self.MackPaid.summary()['Ultimate']/self.MackIncurred.summary()['Ultimate']],
                                                       'left':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0]
                                                       }},
                      'resid1':{'Title':'Paid Residual Plot',
                                     'XLabel':'Incurred/Paid Residuals',
                                     'YLabel':'Paid Residuals',
                                     'chart_type_dict':{'type':['plot','plot','plot','plot'],
                                                       'x':[self.Q_inverse_residuals,[min(self.Q_inverse_residuals),max(self.Q_inverse_residuals)],[0,0],[min(self.Q_inverse_residuals),max(self.Q_inverse_residuals)]],
                                                       'y':[self.paid_residuals, [min(self.Q_inverse_residuals)*self.lambdaP,max(self.Q_inverse_residuals)*self.lambdaP],[min(self.paid_residuals),max(self.paid_residuals)],[0,0]],
                                                       'marker':['o','','',''],
                                                       'linestyle':['','-','-','-'],
                                                       'color':['blue','red','grey','grey']
                                                       }},
                      'resid2':{'Title':'Incurred Residual Plot',
                                     'XLabel':'Paid/Incurred Residuals',
                                     'YLabel':'Incurred Residuals',
                                     'chart_type_dict':{'type':['plot','plot','plot', 'plot'],
                                                       'x':[self.Q_residuals,[min(self.Q_residuals),max(self.Q_residuals)],[0,0],[min(self.Q_residuals),max(self.Q_residuals)]],
                                                       'y':[self.incurred_residuals, [min(self.Q_residuals)*self.lambdaI,max(self.Q_residuals)*self.lambdaI],[min(self.incurred_residuals),max(self.incurred_residuals)],[0,0]],
                                                       'marker':['o','','',''],
                                                       'linestyle':['','-','-','-'],
                                                       'color':['blue','red','grey','grey']
                                                       }},
                      'MCLvsMackpaid':{'Title':'Paid Munich vs Paid Mack',
                                     'XLabel':'Origin',
                                     'YLabel':'Ultimate',
                                     'chart_type_dict':{'type':['bar','bar'],
                                                       'height':[self.summary()['Ult. Paid'], self.MackPaid.summary()['Ultimate']],
                                                       'left':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0]
                                                       }},
                      'MCLvsMackinc':{'Title':'Incurred Munich vs Paid Mack',
                                     'XLabel':'Origin',
                                     'YLabel':'Ultimate',
                                     'chart_type_dict':{'type':['bar','bar'],
                                                       'height':[self.summary()['Ult. Incurred'], self.MackIncurred.summary()['Ultimate']],
                                                       'left':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0]
                                                       }}
                    }
        return plot_dict


    