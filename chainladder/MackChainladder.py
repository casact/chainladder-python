# -*- coding: utf-8 -*-   

"""The MackChainladder module and class establishes the statistical framework
of the chainladder method.  This is based *The Standard Error of Chain Ladder 
Reserve Estimates: Recursive Calculation and Inclusion of a Tail Factor* by
Thomas Mack `[Mack99] <citations.html>`_.

"""
import numpy as np
from pandas import DataFrame, concat, Series
from scipy import stats
from chainladder.Chainladder import Chainladder
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from chainladder.UtilityFunctions import Plot


class MackChainladder:
    """ MackChainLadder class specifies the Mack chainladder model.

    Thomas Mack published in 1993 [Mac93] a method which estimates the standard
    errors of the chain-ladder forecast without assuming a distribution under 
    three conditions.
    `Proper Citation Needed... <https://github.com/mages/ChainLadder>`_

    Parameters:    
        tri : `Triangle <Triangle.html>`_
            A triangle object. Refer to :class:`Triangle`
        weights : int
            A value representing an input into the weights of the WRTO class.
        alpha : int
            A value :math:`\\alpha \\in \\{0;1;2\\}` where: 
                | :math:`\\alpha = 0` corresponds with straight average LDFs.
                | :math:`\\alpha = 1` corresponds with volume weighted average LDFs.          
                | :math:`\\alpha = 2` corresponds with ordinary regression with intercept 0.  
        tail : bool
            Represent whether a tail factor should be applied to the data. 
            Value of False sets tail factor to 1.0

    Attributes:
        tri : `Triangle <Triangle.html>`_
            The raw triangle
        weights : pandas.DataFrame
            A value representing an input into the weights of the WRTO class.
        full_triangle : pandas.DataFrame
            A completed triangle using Mack Chainladder assumptions.
        f : numpy.array
            An array representing the loss development (age-to-age) factors defined as: 
            :math:`\\widehat{f}_{k}=\\frac{\\sum_{i=1}^{n-k}w_{ik}C_{ik}^{\\alpha}
            F_{ik}}{\\sum_{i=1}^{n-k}w_{ik}C_{ik}^{\\alpha}}` 
            
            where: 
            :math:`F_{ik}=C_{i,k+1}/C_{ik}, 1\\leq i\\leq n, 1\\leq k \\leq n-1`  
            
            are the individual development factors and where 
            :math:`w_{ik}\\in [0;1]`  
        chainladder: `Chainladder <ChainLadder.html>`_
        sigma: numpy.array
            An array representing the standard error of :math:`\\widehat{f}`:
            :math:`\\widehat{\\sigma}_{k}=\\sqrt{\\frac{1}{n-k-1}
            \\sum_{i=1}^{n-k}w_{ik}C_{ik}^{\\alpha}\\left (F_{ik}-
            \\widehat{f}_{k}  \\right )^{2}}, 1\\leq k\\leq n - 2`   
        fse : numpy.array
            An array representing the (col-1) standard errors of loss 
            development factors.
        Fse : pandas.DataFrame
            Needs documentation
        parameter_risk: pandas.DataFrame
            Parameter risk
        process_risk: pandas.DataFrame
            Process Risk
        total_parameter_risk: numpy.array
            Needs documentation
        total_process_risk: numpy.array
            Needs documentation
        mack_se: pandas.DataFrame
            Needs documentation
        total_mack_se: float32  
            Needs documentation

    """

    def __init__(self, tri,  weights=1,  alpha=1,
                 tail=False):
        # Determine whether LDFs can eb extrapolated with exponential tail
        if tail == True:
            tail = self.is_exponential_tail_appropriate(tri,  weights,  alpha)       
        self.chainladder = Chainladder(tri, weights=weights,  delta=2 - alpha, tail=tail)
        self.triangle = self.chainladder.triangle
        self.alpha = [2 - item for item in self.chainladder.delta]
        self.weights = self.chainladder.weights
        self.full_triangle = self.chainladder.full_triangle
        self.f = self.chainladder.LDF
        self.sigma = np.array([item.sigma for item in self.chainladder.models])[:-1]
        self.sigma = np.append(self.sigma,self.__get_tail_sigma())
        self.fse = np.array([item.standard_error for item in self.chainladder.models])[:-1]
        self.fse = np.append(self.fse,self.__get_tail_se())
        self.Fse = self.__get_Fse()
        self.parameter_risk = self.__get_parameter_risk()
        self.process_risk = self.__get_process_risk()
        self.total_process_risk = np.array(np.sqrt((self.__get_process_risk()**2).sum()))
        self.total_parameter_risk = self.__get_total_parameter_risk()
        self.mack_se = np.sqrt(self.__get_process_risk()**2 +
                               self.__get_parameter_risk()**2)
        self.total_mack_se = np.sqrt(
            self.total_process_risk[-1]**2 + self.total_parameter_risk[-1]**2)

    def __repr__(self):   
        return str(self.summary())
    
    def __get_process_risk(self):
        """ Method to return the process risk of the Mack Chainladder model.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$Mack.ProcessRisk.
        """

        procrisk = DataFrame([0 for item in range(len(self.full_triangle))],
                             index=self.full_triangle.index, columns=[self.full_triangle.columns[0]])
        bool_df = self.triangle.data.isnull()
        bool_df['Ult'] = True
        ind = 0 if self.chainladder.tail == False else 1
        for i in range(1, self.triangle.ncol + ind):
            temp = DataFrame(np.sqrt((self.full_triangle.iloc[:, i - 1] * self.Fse.iloc[:, i - 1])**2 + (
                self.f[i - 1] * procrisk.iloc[:, i - 1])**2) * bool_df.iloc[:, i])
            temp.columns = [self.full_triangle.columns[i]]
            procrisk = concat([procrisk, temp], axis=1)
        return procrisk

    def __get_parameter_risk(self):
        """ Method to return the parameter risk of the Mack Chainladder model.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$Mack.ParameterRisk.
        """

        paramrisk = DataFrame([0 for item in range(len(self.full_triangle))],
                              index=self.full_triangle.index, columns=[self.full_triangle.columns[0]])
        bool_df = self.triangle.data.isnull()
        bool_df['Ult'] = True
        ind = 0 if self.chainladder.tail == False else 1
        for i in range(1, self.triangle.ncol+ind):
            temp = DataFrame(np.sqrt((self.full_triangle.iloc[:, i - 1] * self.fse[i - 1])**2 + (
                self.f[i - 1] * paramrisk.iloc[:, i - 1])**2) * bool_df.iloc[:, i])          
            temp.columns = [self.full_triangle.columns[i]]
            paramrisk = concat([paramrisk, temp], axis=1)
        return paramrisk

    def __get_total_parameter_risk(self):
        """ Method to produce the parameter risk of the Mack Chainladder model
        for each development column.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$Total.ProcessRisk.

        """
        M = np.empty(0)
        tpr = [0]
        ind = 1 if self.chainladder.tail == False else 0
        for i in range(len(self.full_triangle.columns)-ind):
            M = np.append(M, np.array(
                sum(self.full_triangle.iloc[:, i].iloc[-(i + 1):])))
        for i in range(len(self.full_triangle.columns) - 1 - ind):
            tpr.append(np.sqrt((M[i] * self.fse[i]) **
                               2 + (tpr[-1] * self.f[i])**2))
        return np.array(tpr)

    def __get_Fse(self):
        """ Method to produce the Full Triangle standard error of the Mack Chainladder 
        model.


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$F.se
        """

        # This is sloppy, and I don't know that it works for non-uniform
        # weights.
        full_triangle = self.full_triangle.iloc[:,:-1]
        full_triangleweightconst = self.weights.data.mode().T.mode().iloc[0, 0]
        full_triangleweight = self.full_triangle * 0 + full_triangleweightconst
        Fse = DataFrame()
        for i in range(self.full_triangle.shape[1] - 1):
            Fse = concat([Fse, DataFrame(self.sigma[i] /
                                         np.sqrt(np.array(full_triangleweight.iloc[:, i]).astype(float)
                                                 * np.array(self.full_triangle.iloc[:, i]).astype(float)**self.alpha[i]))], axis=1)
        Fse.set_index(self.full_triangle.index, inplace=True)
        Fse.columns = self.triangle.data.columns
        if self.chainladder.tail == True:
            Fse.iloc[:,-1] = self.sigma[-1]/np.sqrt(self.full_triangle.iloc[:,-2])
        else: 
            Fse = Fse.iloc[:,:-1]
        return Fse

    def __get_tail_se(self):
        """ Method to produce the standard error of the Mack Chainladder 
        model tail factor


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$tail.se
        """
        
        tailse = np.array(self.sigma[-2] / \
            np.sqrt(self.full_triangle.iloc[0, -3]**self.alpha[-1]))
        if self.chainladder.tail == True:
            time_pd = self.__get_tail_weighted_time_period()
            fse = np.append(self.fse, tailse)
            x = np.array([i + 1 for i in range(len(fse))])
            fse_reg = stats.linregress(x, np.log(fse))
            tailse = np.append(tailse, np.exp(time_pd * fse_reg[0] + fse_reg[1]))
        else:
            tailse = np.append(tailse,0)
        return tailse

    def __get_tail_sigma(self):
        """ Method to produce the sigma of the Mack Chainladder 
        model tail factor


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$tail.sigma
        """
        y = np.log(self.sigma[:len(self.triangle.data.columns[:-2])])
        x = np.array([i + 1 for i in range(len(y))])
        model = stats.linregress(x, y)
        tailsigma = np.exp((x[-1] + 1) * model[0] + model[1])
        if model[3] > 0.05:  # p-vale of slope parameter
            y = self.sigma
            tailsigma = np.sqrt(
                abs(min((y[-1]**4 / y[-2]**2), min(y[-2]**2, y[-1]**2))))
        if self.chainladder.tail == True: 
            time_pd = self.__get_tail_weighted_time_period()
            y = np.log(np.append(self.sigma,tailsigma))
            x = np.array([i + 1 for i in range(len(y))])
            sigma_reg = stats.linregress(x, y)
            tailsigma = np.append(tailsigma, np.exp(time_pd * sigma_reg[0] + sigma_reg[1]))
        else:
            tailsigma = np.append(tailsigma,0)
        return tailsigma

    def __get_tail_weighted_time_period(self):
        """ Method to approximate the weighted-average development age assuming
        exponential tail fit.
        
        Returns: float32
        """
        #n = self.triangle.ncol-1
        #y = self.f[:n]
        #x = np.array([i + 1 for i in range(len(y))]) 
        #ldf_reg = stats.linregress(x, np.log(y - 1))
        #time_pd = (np.log(self.f[n] - 1) - ldf_reg[1]) / ldf_reg[0]
        
        n = self.triangle.ncol-1
        y = Series(self.f[:n])
        x = [num+1 for num, item in enumerate(y)]
        y.index = x
        x = sm.add_constant((y.index)[y>1])
        y = y[y>1]
        ldf_reg = sm.OLS(np.log(y-1),x).fit()
        time_pd = (np.log(self.f[n] - 1) - ldf_reg.params[0]) / ldf_reg.params[1]
        #tail_factor = np.exp(tail_model.params[0] + np.array([i+2 for i in range(n,n+100)]) * tail_model.params[1]).astype(float) + 1)
        
        return time_pd
        
    def summary(self):
        """ Method to produce a summary table of of the Mack Chainladder 
        model.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$summary
            
        """
        summary = DataFrame()
        summary['Latest'] = self.triangle.get_latest_diagonal()
        summary['Dev to Date'] = Series([row.dropna().iloc[-1] for index, row in self.triangle.data.iterrows(
        )], index=self.triangle.data.index) / self.full_triangle.iloc[:, -1]
        summary['Ultimate'] = self.full_triangle.iloc[:, -1]
        summary['IBNR'] = summary['Ultimate'] - summary['Latest']
        summary['Mack S.E.'] = self.mack_se.iloc[:, -1]
        summary['CV(IBNR)'] = summary['Mack S.E.'] / summary['IBNR']
        return summary
    
    def age_to_age(self, colname_sep='-'):
        """ Simple method that calls on the Chainladder class method of the 
        same name.  See age_to_age() in Chainladder module.

        Parameters:    
            colname_sep : str
                text to join the names of two adjacent columns representing the
                age-to-age factor column name.

        Returns:
            Pandas.DataFrame of the age-to-age triangle.
            
        """
        return self.chainladder.age_to_age(colname_sep='-')
    
    def is_exponential_tail_appropriate(self, tri,  weights,  alpha):
        """ Method to determine whether an exponential tail fit is appropriate
        based purely on the p-value of the slope parameter.  This method
        currently uses scipy.stats, but this dependency can be removed as 
        statsmodels.api has a more robust regression framework and is used
        elsewhere in the package.
        
        Arguments:
            tri: Triangle 
                An object of the Triangle class that will be used to test the 
                exponential tail fit.
            weights: Triangle 
                An object representing the weights of the exponential regression
                model.
            alpha: pandas.Series
                An object representing the alpha parameter of the mackChainLadder
                model.
        
        Returns:
            Boolean value is returned as True if exponential tail is appropriate
            otherwise a value of False is returned.
                
        
        """
        if Chainladder(tri,weights=weights, delta=2-alpha, tail=True).get_tail_factor()[0] == 1:
            return False
        else:
            return True
    
    def plot(self, plots=['summary', 'full_triangle', 'resid1', 'resid2','resid3','resid4']): 
        """ Method, callable by end-user that renders the matplotlib plots.
        
        Arguments:
            plots: list[str]
                A list of strings representing the charts the end user would like
                to see.  If ommitted, all plots are displayed.  Available plots include:
                    ============== =================================================
                    Str            Description
                    ============== =================================================
                    summary        Bar chart with Ultimates and std. Errors
                    full_triangle  Line chart of origin period x development period
                    resid1         Studentized residuals x fitted Value
                    resid2         Studentized residuals x origin period
                    resid3         Studentized residuals x calendar period
                    resid4         Studentized residuals x development period
                    ============== =================================================
                    
        Returns:
            Renders the matplotlib plots.
            
        """   
        my_dict = []
        plot_dict = self.__get_plot_dict()
        for item in plots:
            my_dict.append(plot_dict[item])
        Plot(my_dict)
        
    def __get_plot_dict(self):
        resid_df = self.chainladder.get_residuals()  
        xlabs = self.full_triangle.columns
        xvals = [i+1 for i in range(len(self.full_triangle.columns))]
        summary = self.summary()
        means = list(summary['Ultimate'])
        ci = list(zip(summary['Ultimate']+summary['Mack S.E.'], summary['Ultimate']-summary['Mack S.E.']))
        y_r = [list(summary['Ultimate'])[i] - ci[i][1] for i in range(len(ci))]
        plot_dict = {'resid1':{'Title':'Studentized Residuals by Fitted Value',
                             'XLabel':'Fitted Value',
                             'YLabel':'Studentized Residual',
                             'chart_type_dict':{'type':['plot','plot'],
                                               'x':[resid_df['fitted_value'],lowess(resid_df['standard_residuals'],resid_df['fitted_value'],frac=1 if len(np.unique(resid_df['fitted_value'].values))<=6 else 0.666).T[0]],
                                               'y':[resid_df['standard_residuals'],lowess(resid_df['standard_residuals'],resid_df['fitted_value'],frac=1 if len(np.unique(resid_df['fitted_value'].values))<=6 else 0.666).T[1]],
                                               'marker':['o',''],
                                               'linestyle':['','-'],
                                               'color':['blue','red']
                                               }},
                    'resid4':{'Title':'Studentized Residuals by Development Period',
                                     'XLabel':'Development Period',
                                     'YLabel':'Studentized Residual',
                                     'chart_type_dict':{'type':['plot','plot'],
                                                       'x':[resid_df['dev'],lowess(resid_df['standard_residuals'],resid_df['dev'],frac=1 if len(np.unique(resid_df['dev'].values))<=6 else 0.666).T[0]],
                                                       'y':[resid_df['standard_residuals'],lowess(resid_df['standard_residuals'],resid_df['dev'],frac=1 if len(np.unique(resid_df['dev'].values))<=6 else 0.666).T[1]],
                                                       'marker':['o',''],
                                                       'linestyle':['','-'],
                                                       'color':['blue','red']
                                                       }},
                    'resid2':{'Title':'Studentized Residuals by Origin Period',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Studentized Residual',
                                     'chart_type_dict':{'type':['plot','plot'],
                                                       'x':[resid_df.index,lowess(resid_df['standard_residuals'],resid_df.index, frac=1 if len(np.unique(resid_df.index.values))<=6 else 0.666).T[0]],
                                                       'y':[resid_df['standard_residuals'],lowess(resid_df['standard_residuals'],resid_df.index, frac=1 if len(np.unique(resid_df.index.values))<=6 else 0.666).T[1]],
                                                       'marker':['o',''],
                                                       'linestyle':['','-'],
                                                       'color':['blue','red']
                                                       }},
                    'resid3':{'Title':'Studentized Residuals by Calendar Period',
                                     'XLabel':'Calendar Period',
                                     'YLabel':'Studentized Residual',
                                     'chart_type_dict':{'type':['plot','plot'],
                                                       'x':[resid_df['cal_period'],lowess(resid_df['standard_residuals'],resid_df['cal_period'],frac=1 if len(np.unique(resid_df['cal_period'].values))<=6 else 0.666).T[0]],
                                                       'y':[resid_df['standard_residuals'],lowess(resid_df['standard_residuals'],resid_df['cal_period'],frac=1 if len(np.unique(resid_df['cal_period'].values))<=6 else 0.666).T[1]],
                                                       'marker':['o',''],
                                                       'linestyle':['','-'],
                                                       'color':['blue','red']
                                                       }},      
                    'summary':{'Title':'Ultimates by origin period +/- 1 std. err.',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Ultimates',
                                     'chart_type_dict':{'type':['bar','bar'],
                                                       'height':[summary['Latest'],summary['IBNR']],
                                                       'left':[summary.index,summary.index],
                                                       'width':[0.8,0.8],
                                                       'bottom':[0,summary['Latest']],
                                                       'yerr':[0,y_r]
                                                       }},
                    'full_triangle':{'Title':'Fully Developed Triangle',
                                     'XLabel':'Development Period',
                                     'YLabel':'Values',
                                     'chart_type_dict':{'type':['line','line'],
                                                       'rows':[len(self.full_triangle), len(self.triangle.data)],
                                                       'x':[xvals, xvals[:-1]],
                                                       'y':[self.full_triangle, self.triangle.data],
                                                       'linestyle':['--', '-'],
                                                       'linewidth':[5, 5],
                                                       'alpha':[0.5, 1.0]
                                                       
                                                       }} ,
                    'triangle':{'Title':'Latest Triangle Data',
                                     'XLabel':'Development Period',
                                     'YLabel':'Values',
                                     'chart_type_dict':{'type':['line'],
                                                       'rows':[len(self.triangle.data)],
                                                       'x':[xvals[:-1]],
                                                       'y':[self.triangle.data],
                                                       'linestyle':['-'],
                                                       'linewidth':[5],
                                                       'alpha':[1]
                                                       }} 
                    }
        return plot_dict


    
