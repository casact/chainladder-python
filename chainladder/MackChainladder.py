# -*- coding: utf-8 -*-

"""The MackChainladder module and class establishes the statistical framework
of the chainladder method.  With MackChainLadder, various standard errors can
be computed to ultimately develop confidence intervals on IBNR estimates.   
"""
import numpy as np
from pandas import DataFrame, concat, Series, pivot_table
from scipy import stats
from chainladder.Chainladder import Chainladder
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess


class MackChainladder:
    """ MackChainLadder class specifies the Mack chainladder model.

    Thomas Mack published in 1993 [Mac93] a method which estimates the standard
    errors of the chain-ladder forecast without assuming a distribution under 
    three conditions.
    `Proper Citation Needed... <https://github.com/mages/ChainLadder>`_

    Parameters:    
        tri : Triangle
            A triangle object. Refer to :class:`Classes.Triangle`
        weights : int
            A value representing an input into the weights of the WRTO class.
        alpha : int
            A value representing an input into the weights of the WRTO class.
        tail : bool
            Represent whether a tail factor should be applied to the data. 
            Value of False sets tail factor to 1.0

    Attributes:
        tri : Triangle
            A triangle object. Refer to :class:`Classes.Triangle`
        weights : pandas.DataFrame
            A value representing an input into the weights of the WRTO class.
        full_triangle : pandas.DataFrame
            A completed triangle using Mack Chainladder assumptions.
        f : numpy.array
            An array representing the (col-1) loss development factors, 
            f-notation borrowed from Mack
        fse : numpy.array
            An array representing the (col-1) standard errors of loss 
            development factors.

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
        self.sigma = np.append(self.sigma,self.get_tail_sigma())
        self.fse = np.array([item.standard_error for item in self.chainladder.models])[:-1]
        self.fse = np.append(self.fse,self.get_tail_se())
        self.Fse = self.get_Fse()
        self.total_process_risk = np.array(np.sqrt((self.get_process_risk()**2).sum()))
        self.total_parameter_risk = self.get_total_parameter_risk()
        self.mack_se = np.sqrt(self.get_process_risk()**2 +
                               self.get_parameter_risk()**2)
        self.total_mack_se = np.sqrt(
            self.total_process_risk[-1]**2 + self.total_parameter_risk[-1]**2)

    def __repr__(self):   
        return str(self.summary())
    
    def get_process_risk(self):
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
        for i in range(1, len(self.triangle.data.columns) + ind):
            temp = DataFrame(np.sqrt((self.full_triangle.iloc[:, i - 1] * self.Fse.iloc[:, i - 1])**2 + (
                self.f[i - 1] * procrisk.iloc[:, i - 1])**2) * bool_df.iloc[:, i])
            temp.columns = [self.full_triangle.columns[i]]
            procrisk = concat([procrisk, temp], axis=1)
        return procrisk

    def get_parameter_risk(self):
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
        for i in range(1, len(self.triangle.data.columns)+ind):
            temp = DataFrame(np.sqrt((self.full_triangle.iloc[:, i - 1] * self.fse[i - 1])**2 + (
                self.f[i - 1] * paramrisk.iloc[:, i - 1])**2) * bool_df.iloc[:, i])
            temp.columns = [self.full_triangle.columns[i]]
            paramrisk = concat([paramrisk, temp], axis=1)
        return paramrisk

    def get_total_parameter_risk(self):
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

    def get_Fse(self):
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
        return Fse

    def get_tail_se(self):
        """ Method to produce the standard error of the Mack Chainladder 
        model tail factor


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$tail.se
        """
        
        tailse = np.array(self.sigma[-2] / \
            np.sqrt(self.full_triangle.iloc[0, -3]**self.alpha[-1]))
        if self.chainladder.tail == True:
            time_pd = self.get_tail_weighted_time_period()
            fse = np.append(self.fse, tailse)
            x = np.array([i + 1 for i in range(len(fse))])
            fse_reg = stats.linregress(x, np.log(fse))
            tailse = np.append(tailse, np.exp(time_pd * fse_reg[0] + fse_reg[1]))
        else:
            tailse = np.append(tailse,0)
        return tailse

    def get_tail_sigma(self):
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
            time_pd = self.get_tail_weighted_time_period()
            y = np.log(np.append(self.sigma,tailsigma))
            x = np.array([i + 1 for i in range(len(y))])
            sigma_reg = stats.linregress(x, y)
            tailsigma = np.append(tailsigma, np.exp(time_pd * sigma_reg[0] + sigma_reg[1]))
        else:
            tailsigma = np.append(tailsigma,0)
        return tailsigma

    def get_tail_weighted_time_period(self):
        """ Method to approximate the weighted-average development age assuming
        exponential tail fit.
        
        Returns: float32
        """
        n = len(self.triangle.data.columns)-1
        y = self.f[:n]
        x = np.array([i + 1 for i in range(len(y))]) 
        ldf_reg = stats.linregress(x, np.log(y - 1))
        time_pd = (np.log(self.f[n] - 1) - ldf_reg[1]) / ldf_reg[0]
        return time_pd
        
    def summary(self):
        """ Method to produce a summary table of of the Mack Chainladder 
        model.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$summary
        """
        summary = DataFrame()
        summary['Latest'] = Series([row.dropna(
        ).iloc[-1] for index, row in self.triangle.data.iterrows()], index=self.triangle.data.index)
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
        y = Series(Chainladder(tri, weights=weights,  delta=2 - alpha).LDF)
        y.index = [i+1 for i in range(len(y))]
        y = y[y>1]
        model = stats.linregress(y.index, np.log(y-1))
        if model[3]>0.05:
            warn('Unable to generate tail from LDFs, setting tail factor to 1.0')        
            return False
        else:
            return True
    
    def plot(self, plots=['summary', 'full_triangle', 'resid1', 'resid2','resid3','resid4']): 
        """ Method, callable by end-user that renders the matplotlib plots 
        based on the configurations in __get_plot_dict().
        
        Arguments:
            plots: list[str]
                A list of strings representing the charts the end user would like
                to see.  Of ommitted, all plots are displayed.

        Returns:
            Renders the matplotlib plots.
        """        
        
        sns.set_style("whitegrid")
        _ = plt.figure()
        grid_x = 1 if len(plots) == 1 else round(len(plots) / 2,0)
        grid_y = 1 if len(plots) == 1 else 2
        fig, ax = plt.subplots(figsize=(grid_y*15, grid_x*10))
        for i in range(len(plots)):
            _ = plt.subplot(grid_x,grid_y,i+1)
            self.dict_plot(plots[i]) 

    def dict_plot(self, chart):
        """ Method that renders the matplotlib plots based on the configurations
        in __get_plot_dict().  This method should probably be private and may be
        so in a future release.
        
        """
        my_dict = self.__get_plot_dict()[chart]
        for i in range(len(my_dict['chart_type_dict']['type'])):
            if my_dict['chart_type_dict']['type'][i] == 'plot':
                _ = plt.plot(my_dict['chart_type_dict']['x'][i], 
                             my_dict['chart_type_dict']['y'][i],
                             linestyle=my_dict['chart_type_dict']['linestyle'][i], 
                             marker=my_dict['chart_type_dict']['marker'][i], 
                             color=my_dict['chart_type_dict']['color'][i])
            if my_dict['chart_type_dict']['type'][i] == 'bar':
                _ =plt.bar(height = my_dict['chart_type_dict']['height'][i], 
                           left = my_dict['chart_type_dict']['left'][i],
                           yerr = my_dict['chart_type_dict']['yerr'][i], 
                           bottom = my_dict['chart_type_dict']['bottom'][i])
            if my_dict['chart_type_dict']['type'][i] == 'line':
                _ = plt.gca().set_prop_cycle(None)
                for j in range(my_dict['chart_type_dict']['rows'][i]):
                    _ = plt.plot(my_dict['chart_type_dict']['x'][i], 
                                 my_dict['chart_type_dict']['y'][i].iloc[j], 
                                 linestyle=my_dict['chart_type_dict']['linestyle'][i], 
                                 linewidth=my_dict['chart_type_dict']['linewidth'][i],
                                 alpha=my_dict['chart_type_dict']['alpha'][i])
            _ = plt.title(my_dict['Title'], fontsize=30)

    def __get_plot_dict(self):
        """ Private method that is designed to configure the matplotlib graphs.
        
        Returns:
            Returns a dictionary containing the configuration of the selected plot.
        """
        resid_df = self.chainladder.get_residuals()  
        # Jittering values so LOWESS can calculate
        xlabs = self.full_triangle.columns
        xvals = [i for i in range(len(self.full_triangle.columns))]
        
        summary = self.summary()
        means = list(summary['Ultimate'])
        ci = list(zip(summary['Ultimate']+summary['Mack S.E.'], summary['Ultimate']-summary['Mack S.E.']))
        y_r = [list(summary['Ultimate'])[i] - ci[i][1] for i in range(len(ci))]
        plot_dict = {'resid1':{'Title':'Studentized Residuals by fitted Value',
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
                    'resid3':{'Title':'Studentized Residuals by Origin Period',
                                     'XLabel':'Origin Period',
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
                                                       'bottom':[0,summary['Latest']],
                                                       'yerr':[0,y_r]
                                                       }},
                    'full_triangle':{'Title':'Fully Developed Triangle',
                                     'XLabel':'Origin Period',
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
                                     'XLabel':'Origin Period',
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
    
