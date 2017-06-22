# -*- coding: utf-8 -*-

"""The MackChainladder module and class establishes the statistical framework
of the chainladder method.  With MackChainLadder, various standard errors can
be computed to ultimately develop confidence intervals on IBNR estimates.   
"""
import numpy as np
from pandas import DataFrame, concat, Series, pivot_table
from scipy import stats
from chainladder.Triangle import Triangle
from chainladder.Chainladder import Chainladder


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
        return self.chainladder.age_to_age(colname_sep='-')
    
