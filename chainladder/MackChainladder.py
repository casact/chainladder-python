# -*- coding: utf-8 -*-

"""The MackChainladder module and class establishes the statistical framework
of the chainladder method.  With MackChainLadder, various standard errors can
be computed to ultimately develop confidence intervals on IBNR estimates.   
"""
import numpy as np
from pandas import DataFrame, concat, Series, pivot_table
from scipy import stats
from chainladder.Triangle import Triangle
from chainladder.Chainladder import ChainLadder


class MackChainLadder:
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
        fullTriangle : pandas.DataFrame
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
        delta = 2 - alpha
        if tail == False:
            self.tail_factor = 1.0
        else:
            self.tail_factor = 2.0  # need to do more tail test cases
        self.triangle = tri
        self.triangle.dataAsTriangle(inplace=True)
        cl = ChainLadder(self.triangle, weights=weights,  delta=delta)
        alpha = [2 - item for item in cl.delta]

        self.weights = cl.weights
        self.fullTriangle = cl.predict()
        self.alpha = alpha
        self.f = np.append(
            np.array([item.coef_ for item in cl.models]), self.tail_factor)
        self.fse = np.array([item.se for item in cl.models])[:-1]
        self.sigma = np.array([item.sigma for item in cl.models])[:-1]
        self.sigma = np.append(self.sigma, self.tail_sigma())
        self.fse = np.append(self.fse, self.tail_se())
        self.Fse = self.Fse()
        self.total_process_risk = np.sqrt((self.process_risk()**2).sum())
        self.total_parameter_risk = self.total_parameter_risk()
        self.mack_se = np.sqrt(self.process_risk()**2 +
                               self.parameter_risk()**2)
        self.total_mack_se = np.sqrt(
            self.total_process_risk[-1]**2 + self.total_parameter_risk[-1]**2)

    def process_risk(self):
        """ Method to return the process risk of the Mack Chainladder model.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$Mack.ProcessRisk.
        """

        procrisk = DataFrame([0 for item in range(len(self.fullTriangle))],
                             index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1, len(self.fullTriangle.columns)):
            temp = DataFrame(np.sqrt((self.fullTriangle.iloc[:, i - 1] * self.Fse.iloc[:, i - 1])**2 + (
                self.f[i - 1] * procrisk.iloc[:, i - 1])**2) * self.triangle.data.iloc[:, i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            procrisk = concat([procrisk, temp], axis=1)
        return procrisk

    def parameter_risk(self):
        """ Method to return the parameter risk of the Mack Chainladder model.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$Mack.ParameterRisk.
        """

        paramrisk = DataFrame([0 for item in range(len(self.fullTriangle))],
                              index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1, len(self.fullTriangle.columns)):
            temp = DataFrame(np.sqrt((self.fullTriangle.iloc[:, i - 1] * self.fse[i - 1])**2 + (
                self.f[i - 1] * paramrisk.iloc[:, i - 1])**2) * self.triangle.data.iloc[:, i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            paramrisk = concat([paramrisk, temp], axis=1)
        return paramrisk

    def total_parameter_risk(self):
        """ Method to produce the parameter risk of the Mack Chainladder model
        for each development column.

        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$Total.ProcessRisk.

        """
        M = np.empty(0)
        tpr = [0]
        for i in range(len(self.fullTriangle.columns)):
            M = np.append(M, np.array(
                sum(self.fullTriangle.iloc[:, i].iloc[-(i + 1):])))
        for i in range(len(self.fullTriangle.columns) - 1):
            tpr.append(np.sqrt((M[i] * self.fse[i]) **
                               2 + (tpr[-1] * self.f[i])**2))
        return np.array(tpr)

    def Mack_SE(self):
        """ Method to produce the Mack Standard Error of the Mack Chainladder 
        model.


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$Mack.S.E.
        """

        return DataFrame(np.sqrt(np.matrix(self.process_risk()**2)
                                 + np.matrix(self.parameter_risk()**2)), index=self.fullTriangle.index)

    def Fse(self):
        """ Method to produce the Full Triangle standard error of the Mack Chainladder 
        model.


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$F.se
        """

        # This is sloppy, and I don't know that it works for non-uniform
        # weights.
        fulltriangleweightconst = self.weights.data.mode().T.mode().iloc[0, 0]
        fulltriangleweight = self.fullTriangle * 0 + fulltriangleweightconst
        Fse = DataFrame()
        for i in range(self.fullTriangle.shape[1] - 1):
            Fse = concat([Fse, DataFrame(self.sigma[i] /
                                         np.sqrt(np.array(fulltriangleweight.iloc[:, i]).astype(float)
                                                 * np.array(self.fullTriangle.iloc[:, i]).astype(float)**self.alpha[i]))], axis=1)

        Fse.set_index(self.fullTriangle.index, inplace=True)
        return Fse

    def tail_se(self):
        """ Method to produce the standard error of the Mack Chainladder 
        model tail factor


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$tail.se
        """
        if True:
            tailse = self.sigma[-1] / \
                np.sqrt(self.fullTriangle.iloc[0, -2]**self.alpha[-1])

        else:
            # Need to do some further testing of tails to determine the
            # usefulness of this code.
            n = len(self.fullTriangle.columns)
            f = self.f[:-2]
            dev = np.array(self.fullTriangle.columns[:-2]).astype(int)
            ldf_reg = np.polyfit(dev, np.log(f - 1), 1)
            time_pd = (np.log(self.f[-2] - 1) - ldf_reg[1]) / ldf_reg[0]
            fse = self.fse
            fse_reg = np.polyfit(dev, np.log(fse), 1)
            tailse = np.exp(time_pd * fse_reg[0] + fse_reg[1])
        return tailse

    def tail_sigma(self):
        """ Method to produce the sigma of the Mack Chainladder 
        model tail factor


        Returns:
            This calculation is consistent with the R calculation 
            MackChainLadder$tail.sigma
        """
        if True:
            y = np.log(self.sigma)
            x = np.array([i + 1 for i in range(len(self.sigma))])
            model = stats.linregress(x, y)
            tailsigma = np.exp((x[-1] + 1) * model[0] + model[1])
            if model[3] > 0.05:  # p-vale of slope parameter
                y = self.sigma
                tailsigma = np.sqrt(
                    abs(min((y[-1]**4 / y[-2]**2), min(y[-2]**2, y[-1]**2))))
        else:
            # I cannot replicate R exactly!!!
            n = len(self.fullTriangle.columns)
            f = self.f[:-2]
            dev = np.array(self.fullTriangle.columns[:-2]).astype(int)
            ldf_reg = np.polyfit(dev, np.log(f - 1), 1)
            time_pd = (np.log(self.f[-2] - 1) - ldf_reg[1]) / ldf_reg[0]
            sigma = self.sigma
            sigma_reg = np.polyfit(dev, np.log(sigma), 1)
            tailsigma = np.exp(time_pd * sigma_reg[0] + sigma_reg[1])
        return tailsigma

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
        )], index=self.triangle.data.index) / self.fullTriangle.iloc[:, -1]
        summary['Ultimate'] = self.fullTriangle.iloc[:, -1]
        summary['IBNR'] = summary['Ultimate'] - summary['Latest']
        summary['Mack S.E.'] = self.mack_se.iloc[:, -1]
        summary['CV(IBNR)'] = summary['Mack S.E.'] / summary['IBNR']
        return summary
