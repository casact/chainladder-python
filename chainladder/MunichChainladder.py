# -*- coding: utf-8 -*-
"""
The Munich chainladder approach considers the correlations between paid and
incurred loss data so as to converge chainladder paid ultimates and chainladder
incurred ultimates `[Quarg99] <citations.html>`_

"""
import chainladder as cl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from warnings import warn
from chainladder.UtilityFunctions import Plot
from chainladder.Triangle import Triangle
from statsmodels.nonparametric.smoothers_lowess import lowess
from bokeh.palettes import Spectral10

class MunichChainladder:
    """ This is the Munich Chain Ladder Class.
    
    Parameters:
        Paid: `Triangle <Triangle.html>`_
            The raw paid triangle
        Incurred: `Triangle <Triangle.html>`_
            The raw incurred triangle
        tailP: bool
            Represent whether a tail factor should be applied to the paid data. 
            Value of False sets tail factor to 1.0
        tailI: bool
            Represent whether a tail factor should be applied to the incurred data. 
            Value of False sets tail factor to 1.0
        
    Attributes:
        tailP: bool
            Represent whether a tail factor should be applied to the paid data. 
            Value of False sets tail factor to 1.0
        tailI: bool
            Represent whether a tail factor should be applied to the incurred data. 
            Value of False sets tail factor to 1.0
        MackPaid: `MackChainladder <MackChainLadder.html>`_
            A Mack chainladder model of paid triangle
        MackIncurred: `MackChainladder <MackChainLadder.html>`_
            A Mack chainladder model of incurred triangle
        Paid: `Triangle <Triangle.html>`_
            The raw paid triangle
        Incurred: `Triangle <Triangle.html>`_
            The raw incurred triangle
        rhoI_sigma: np.array
            The rho_sigma of incurred:
            
            :math:`\\widehat{\\rho_s^I}=\\sqrt{\\frac{1}{n-s}\\cdot \\sum_{j=1}^{n-s+1}I_{j,s}\\cdot \\left ( Q_{j,s}-\\widehat{q_s} \\right )^2}`
        rhoP_sigma: np.array
            The rho_sigma of paid:
            
            :math:`\\widehat{\\rho_s^P}=\\sqrt{\\frac{1}{n-s}\\cdot \\sum_{j=1}^{n-s+1}P_{j,s}\\cdot \\left ( Q_{j,s}^{-1}-\\widehat{q_s}^{-1} \\right )^2}`
        q_f: np.array
            The volume weighted Paid/Incurred link-ratios
            
            :math:`\\widehat{q_s}=\\frac{\\sum_{j=1}^{n-s+1}P_{j,s}}{\\sum_{j=1}^{n-s+1}I_{j,s}}`
        qinverse_f: np.array
            The volume weighted Incurred/Paid link-ratios
            
            :math:`\\widehat{q_s^{-1}}=\\frac{1}{\\widehat{q_s}}`
        paid_residuals: np.array
            Paid residuals:
            
            :math:`\\mathbf{\\widehat{Res}}(P_{i,t})=\\frac{\\frac{P_{i,t}}{P_{i,s}}-\\widehat{f}_{s\\rightarrow t}^{P}}{\\widehat{\\sigma_{s\\rightarrow t}^P}}\\cdot \\sqrt{P_{i,s}}`,  where
            
            :math:`\\widehat{(\\sigma_{s\\rightarrow t}^{P})}:=\\sqrt{\\frac{1}{n-s-1}\\cdot \\sum_{i=1}^{n-s}P_{i,s}\\cdot \\left (\\frac{P_{i,t}}{P_{i,s}}-\\widehat{f^{_{s\\rightarrow t}^{P}}}  \\right )^2}`
        incurred_residuals: np.array
            Incurred residuals:
            
            :math:`\\mathbf{\\widehat{Res}}(I_{i,t})=\\frac{\\frac{I_{i,t}}{I_{i,s}}-\\widehat{f}_{s\\rightarrow t}^{I}}{\\widehat{\\sigma_{s\\rightarrow t}^I}}\\cdot \\sqrt{I_{i,s}}`,  where
            
            :math:`\\widehat{(\\sigma_{s\\rightarrow t}^{I})}:=\\sqrt{\\frac{1}{n-s-1}\\cdot \\sum_{i=1}^{n-s}I_{i,s}\\cdot \\left (\\frac{I_{i,t}}{I_{i,s}}-\\widehat{f^{_{s\\rightarrow t}^{I}}}  \\right )^2}`
        Q_inverse_residuals: numpy.array
            The Qinverse residuals:
            
            :math:`\\mathbf{\\widehat{Res}}(Q_{i,s}^{-1})=\\frac{Q_{i,s}^{-1}-\\widehat{q}_s^{-1}}{\\widehat{\\rho_s^P}}\\cdot \\sqrt{P_{i,s}}`
        Q_residuals: numpy.array
            The Q residuals:
            
            :math:`\\mathbf{\\widehat{Res}}(Q_{i,s})=\\frac{Q_{i,s}-\\widehat{q}_s}{\\widehat{\\rho_s^I}}\\cdot \\sqrt{I_{i,s}}`
        lambdaP: float32
            The Paid correlation parameter:
            
            :math:`\\widehat{\\lambda^{P}}:=\\frac{\\sum_{i,s}\\mathbf{\\widehat{Res}}(Q_{i,s}^{-1})\\cdot \\mathbf{\\widehat{Res}}(P_{i,t})}{\\sum_{i,s}\\mathbf{\\widehat{Res}}(Q_{i,s}^{-1})^{2}}`
        lambdaI: float32
            The Incurred correlation parameter:
            
            :math:`\\widehat{\\lambda^{I}}:=\\frac{\\sum_{i,s}\\mathbf{\\widehat{Res}}(Q_{i,s})\\cdot \\mathbf{\\widehat{Res}}(I_{i,t})}{\\sum_{i,s}\\mathbf{\\widehat{Res}}(Q_{i,s})^{2}}`
        MCL_paid: pandas.DataFrame
            Fully developed paid triangle using the Munich chainladder model
        MCL_incurred: pandas.DataFrame
            Fully developed paid triangle using the Munich chainladder model
            
    
    """
    def __init__(self, Paid, Incurred, tailP=False, tailI=False):
        if Incurred.shape != Paid.shape:
            warn('Paid and Incurred triangle must have same dimension.')
            return
        if len(Incurred) > len(Incurred.columns): 
            warn('MunichChainLadder does not support triangles with fewer development periods than origin periods.')
            return
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
    
    def __get_PI_ratios(self):
        """ Used for plotting
        """
        actuals = MCL.MackPaid.triangle.data_as_table().data['values']/MCL.MackIncurred.triangle.data_as_table().data['values']
        
    def summary(self):
        """ Method to produce a summary table of of the Munich Chainladder 
        model.

        Returns:
            This calculation is consistent with the R calculation 
            MunichChainLadder$summary
        """
        summary = pd.DataFrame()
        summary['Latest Paid'] = self.Paid.get_latest_diagonal().iloc[:,-1]
        summary['Latest Incurred'] = self.Incurred.get_latest_diagonal().iloc[:,-1]
        summary['Latest P/I Ratio'] = summary['Latest Paid']/summary['Latest Incurred'] 
        summary['Ult. Paid'] = self.MCL_paid.iloc[:,-1]
        summary['Ult. Incurred'] = self.MCL_incurred.iloc[:,-1]
        summary['P/I Ratio'] = summary['Ult. Paid']/summary['Ult. Incurred'] 
        return summary
    
    def plot(self, ctype='m', plots=['summary', 'MCLvsMack', 'resid1', 'resid2'], plot_width=450, plot_height=275):
        """ Method, callable by end-user that renders the matplotlib plots.
        
        Arguments:
            plots: list[str]
                A list of strings representing the charts the end user would like
                to see.  If ommitted, all plots are displayed.  Available plots include:
                    ============== ==========================================================
                    Str            Description
                    ============== ==========================================================
                    summary        Bar chart with Ultimate paid and incurred
                    MCLvsMack      Bar chart comparing P/I of Munich vs. Mack Chainladder 
                    resid1         Paid residual vs I/P residual
                    resid2         Incurred residual vs P/I residual
                    MCLvsMackpaid  Bar chart of paid Munich vs Paid Mack Ultimates
                    MCLvsMackinc   Bar chart of incurred Munich vs incurred Mack Ultimates
                    PI1            P/I ratios by development period using Munich chainladder
                    PI2            P/I ratios by development period using Mack chainladder
                    ============== ==========================================================
                    
        Returns:
            Renders the matplotlib plots.
            
        """   
    
        my_dict = []
        plot_dict = self.__get_plot_dict()
        for item in plots:
            my_dict.append(plot_dict[item])
        return Plot(ctype, my_dict, plot_width=plot_width, plot_height=plot_height).grid
    
    def __get_plot_dict(self):
        plot_dict = {'summary':{'Title':'Munich Chainladder Results',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Ultimates',
                                     'chart_type_dict':{'mtype':['bar','bar'],
                                                       'height':[self.summary()['Ult. Paid'],self.summary()['Ult. Incurred']],
                                                       'x':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0],
                                                       'type':['vbar','vbar'],
                                                       'top':[self.summary()['Ult. Paid'],self.summary()['Ult. Incurred']],
                                                       'color':[Spectral10[1],Spectral10[2]],
                                                       'label':['Paid','Incurred']
                                                       }},
                     'MCLvsMack':{'Title':'Munich Chainladder vs. Standard Chainladder (P/I) Ratio',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Ultimates',
                                     'chart_type_dict':{'mtype':['bar','bar'],
                                                       'height':[self.summary()['Ult. Paid']/self.summary()['Ult. Incurred'],self.MackPaid.summary()['Ultimate']/self.MackIncurred.summary()['Ultimate']],
                                                       'x':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0],
                                                       'type':['vbar','vbar'],
                                                       'top':[self.summary()['Ult. Paid']/self.summary()['Ult. Incurred'],self.MackPaid.summary()['Ultimate']/self.MackIncurred.summary()['Ultimate']],
                                                       'color':[Spectral10[1],Spectral10[2]],
                                                       'label':['Munich','Mack']
                                                       }},
                      'resid1':{'Title':'Paid Residual Plot',
                                     'XLabel':'Incurred/Paid Residuals',
                                     'YLabel':'Paid Residuals',
                                     'chart_type_dict':{'mtype':['plot','plot','plot','plot'],
                                                       'x':[self.Q_inverse_residuals,[min(self.Q_inverse_residuals),max(self.Q_inverse_residuals)],[0,0],[min(self.Q_inverse_residuals),max(self.Q_inverse_residuals)]],
                                                       'yM':[self.paid_residuals, [min(self.Q_inverse_residuals)*self.lambdaP,max(self.Q_inverse_residuals)*self.lambdaP],[min(self.paid_residuals),max(self.paid_residuals)],[0,0]],
                                                       'markerM':['o','','',''],
                                                       'linestyle':['','-','-','-'],
                                                       'colorM':['blue','red','grey','grey'],
                                                       'type':['scatter','line','line','line'],
                                                       'y':[self.paid_residuals, pd.DataFrame([min(self.Q_inverse_residuals)*self.lambdaP,max(self.Q_inverse_residuals)*self.lambdaP]).T,pd.DataFrame([min(self.paid_residuals),max(self.paid_residuals)]).T,pd.DataFrame([0,0]).T],
                                                       'marker':['circle','','',''],
                                                       'line_width':[None,2,2,2],
                                                       'line_cap':[None,'round','round','round'],
                                                       'line_join':[None,'round','round','round'],
                                                       'line_dash':[None,'solid','solid','solid'],
                                                       'alpha':[.5,.5,.5,.5],
                                                       'label':['residual',['fit'],[''],['']],
                                                       'rows':[None,1,1,1],
                                                       'color':['blue',['red'],['grey'],['grey']]
                                                       }},
                      'resid2':{'Title':'Incurred Residual Plot',
                                     'XLabel':'Paid/Incurred Residuals',
                                     'YLabel':'Incurred Residuals',
                                     'chart_type_dict':{'mtype':['plot','plot','plot', 'plot'],
                                                       'x':[self.Q_residuals,[min(self.Q_residuals),max(self.Q_residuals)],[0,0],[min(self.Q_residuals),max(self.Q_residuals)]],
                                                       'yM':[self.incurred_residuals, [min(self.Q_residuals)*self.lambdaI,max(self.Q_residuals)*self.lambdaI],[min(self.incurred_residuals),max(self.incurred_residuals)],[0,0]],
                                                       'markerM':['o','','',''],
                                                       'linestyle':['','-','-','-'],
                                                       'colorM':['blue','red','grey','grey'],
                                                       'type':['scatter','line','line','line'],
                                                       'y':[self.incurred_residuals, pd.DataFrame([min(self.Q_residuals)*self.lambdaI,max(self.Q_residuals)*self.lambdaI]).T,pd.DataFrame([min(self.incurred_residuals),max(self.incurred_residuals)]).T,pd.DataFrame([0,0]).T],
                                                       'marker':['circle','','',''],
                                                       'line_width':[None,2,2,2],
                                                       'line_cap':[None,'round','round','round'],
                                                       'line_join':[None,'round','round','round'],
                                                       'line_dash':[None,'solid','solid','solid'],
                                                       'alpha':[.5,.5,.5,.5],
                                                       'label':['residual',['fit'],[''],['']],
                                                       'rows':[None,1,1,1],
                                                       'marker':['circle','','',''],
                                                       'color':['blue',['red'],['grey'],['grey']]
                                                       }},
                      'MCLvsMackpaid':{'Title':'Paid Munich vs Paid Mack',
                                     'XLabel':'Origin',
                                     'YLabel':'Ultimate',
                                     'chart_type_dict':{'mtype':['bar','bar'],
                                                       'height':[self.summary()['Ult. Paid'], self.MackPaid.summary()['Ultimate']],
                                                       'x':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0],
                                                       'type':['vbar','vbar'],
                                                       'top':[self.summary()['Ult. Paid'], self.MackPaid.summary()['Ultimate']],
                                                       'color':[Spectral10[1],Spectral10[2]],
                                                       'label':['Munich','Mack']
                                                       }},
                      'MCLvsMackinc':{'Title':'Incurred Munich vs Incurred Mack',
                                     'XLabel':'Origin',
                                     'YLabel':'Ultimate',
                                     'chart_type_dict':{'mtype':['bar','bar'],
                                                       'height':[self.summary()['Ult. Incurred'], self.MackIncurred.summary()['Ultimate']],
                                                       'x':[self.summary().index-.35/2,self.summary().index+.35/2],
                                                       'width':[0.35,0.35],
                                                       'bottom':[0,0],
                                                       'yerr':[0,0],
                                                       'type':['vbar','vbar'],
                                                       'top':[self.summary()['Ult. Incurred'], self.MackIncurred.summary()['Ultimate']],
                                                       'color':[Spectral10[1],Spectral10[2]],
                                                       'label':['Munich','Mack']
                                                       }},
                      'PI1':{'Title':'(P/I) Triangle using Basic Chainladder',
                                     'XLabel':'Development Period',
                                     'YLabel':'Paid/Incurred Ratio',
                                     'chart_type_dict':{'mtype':['plot', 'plot', 'plot'],
                                                       'x':[Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag'], self.MackPaid.triangle.data_as_table().data['dev_lag'],
                                                            lowess((Triangle(self.MackPaid.full_triangle.iloc[:,:-1]/self.MackIncurred.full_triangle.iloc[:,:-1]).data_as_table().data)['values'],
                                                                   Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag'],frac=1 if len(np.unique(Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag']))<=6 else 0.666).T[0]],
                                                       'yM':[(Triangle(self.MackPaid.full_triangle.iloc[:,:-1]/self.MackIncurred.full_triangle.iloc[:,:-1]).data_as_table().data)['values'], 
                                                            (self.MackPaid.triangle.data_as_table().data['values']/self.MackIncurred.triangle.data_as_table().data['values']),
                                                            lowess((Triangle(self.MackPaid.full_triangle.iloc[:,:-1]/self.MackIncurred.full_triangle.iloc[:,:-1]).data_as_table().data)['values'],
                                                                   Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag'],frac=1 if len(np.unique(Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag']))<=6 else 0.666).T[1]],
                                                       'markerM':['o','o',''],
                                                       'linestyle':['','','-'],
                                                       'colorM':['grey','blue', 'red'],
                                                       'type':['scatter', 'scatter', 'line'],
                                                       'y':[(Triangle(self.MackPaid.full_triangle.iloc[:,:-1]/self.MackIncurred.full_triangle.iloc[:,:-1]).data_as_table().data)['values'], 
                                                            (self.MackPaid.triangle.data_as_table().data['values']/self.MackIncurred.triangle.data_as_table().data['values']),
                                                            pd.DataFrame(lowess((Triangle(self.MackPaid.full_triangle.iloc[:,:-1]/self.MackIncurred.full_triangle.iloc[:,:-1]).data_as_table().data)['values'],
                                                                   Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag'],frac=1 if len(np.unique(Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag']))<=6 else 0.666).T[1]).T],
                                                       'line_width':[None,None,2],
                                                       'line_cap':[None,None,'round'],
                                                       'line_join':[None,None,'round'],
                                                       'line_dash':[None,None,'solid'],
                                                       'alpha':[.5,.5,.5],
                                                       'label':['projected','actual',['loess']],
                                                       'rows':[None,1,1,1],
                                                       'marker':['circle','circle',''],
                                                       'color':['grey','blue',['red']]
                                                       }},
                      'PI2':{'Title':'(P/I) Triangle using Munich Chainladder',
                                     'XLabel':'Development Period',
                                     'YLabel':'Paid/Incurred Ratio',
                                     'chart_type_dict':{'mtype':['plot', 'plot','plot'],
                                                       'x':[Triangle(self.MCL_paid).data_as_table().data['dev_lag'], self.MackPaid.triangle.data_as_table().data['dev_lag'],
                                                            lowess((Triangle(self.MackPaid.full_triangle.iloc[:,:-1]/self.MackIncurred.full_triangle.iloc[:,:-1]).data_as_table().data)['values'],
                                                                   Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag'],frac=1 if len(np.unique(Triangle(self.MackPaid.full_triangle.iloc[:,:-1]).data_as_table().data['dev_lag']))<=6 else 0.666).T[0]],
                                                       'yM':[(Triangle(self.MCL_paid/self.MCL_incurred).data_as_table().data)['values'], 
                                                            (self.MackPaid.triangle.data_as_table().data['values']/self.MackIncurred.triangle.data_as_table().data['values']),
                                                            lowess((Triangle(self.MCL_paid/self.MCL_incurred).data_as_table().data)['values'],
                                                                   Triangle(self.MCL_paid).data_as_table().data['dev_lag'],frac=1 if len(np.unique(Triangle(self.MCL_paid).data_as_table().data['dev_lag']))<=6 else 0.666).T[1]],
                                                       'markerM':['o','o',''],
                                                       'linestyle':['','','-'],
                                                       'colorM':['grey','blue', 'red'],
                                                       'type':['scatter', 'scatter', 'line'],
                                                       'y':[(Triangle(self.MCL_paid/self.MCL_incurred).data_as_table().data)['values'], 
                                                            (self.MackPaid.triangle.data_as_table().data['values']/self.MackIncurred.triangle.data_as_table().data['values']),
                                                            pd.DataFrame(lowess((Triangle(self.MCL_paid/self.MCL_incurred).data_as_table().data)['values'],
                                                                   Triangle(self.MCL_paid).data_as_table().data['dev_lag'],frac=1 if len(np.unique(Triangle(self.MCL_paid).data_as_table().data['dev_lag']))<=6 else 0.666).T[1]).T],
                                                       'line_width':[None,None,2],
                                                       'line_cap':[None,None,'round'],
                                                       'line_join':[None,None,'round'],
                                                       'line_dash':[None,None,'solid'],
                                                       'alpha':[.5,.5,.5],
                                                       'label':['projected','actual',['loess']],
                                                       'rows':[None,1,1,1],
                                                       'marker':['circle','circle',''],
                                                       'color':['grey','blue',['red']]
                                                       }},      
                    }
        return plot_dict


    