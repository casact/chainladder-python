# -*- coding: utf-8 -*-

"""The MackChainladder module and class establishes the statistical framework
of the chainladder method.  This is based *The Standard Error of Chain Ladder
Reserve Estimates: Recursive Calculation and Inclusion of a Tail Factor* by
Thomas Mack `[Mack99] <citations.html>`_.

"""
import numpy as np
import copy
from chainladder.methods.base import MethodBase


class Mack(MethodBase):
    """ MackChainladder class specifies the Mack chainladder model.

    Thomas Mack published in 1993 [Mac93] a method which estimates the standard
    errors of the chainladder forecast without assuming a distribution under
    three conditions.
    `Proper Citation Needed... <https://github.com/mages/ChainLadder>`_

    Parameters:

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

    def fit(self, X, y=None):
        self.validate_X(X)
        self._mack_recursion('param_risk')
        self._mack_recursion('process_risk')
        self._mack_recursion('total_param_risk')
        return self

    @property
    def full_std_err_(self):
        obj = copy.deepcopy(self.X_)
        tri_array = self.full_triangle_.triangle
        weight_dict = {'regression': 0, 'volume': 1, 'simple': 2}
        val = np.array([weight_dict.get(item.lower(), 2)
                        for item in list(self.average_) + ['volume']])
        for i in [2, 1, 0]:
            val = np.repeat(np.expand_dims(val, 0), tri_array.shape[i], axis=0)
        k, v, o, d = val.shape
        weight = np.sqrt(tri_array[:, :, :, :-1]**(2-val))
        obj.triangle = self.sigma_.triangle / weight
        obj.nan_override = True
        return obj

    @property
    def total_process_risk_(self):
        obj = copy.deepcopy(self.process_risk_)
        obj.triangle = np.sqrt(np.sum(self.process_risk_.triangle**2, 2))
        obj.triangle = np.expand_dims(obj.triangle, 2)
        obj.odims = ['tot_proc_risk']
        return obj

    def _mack_recursion(self, est):
        obj = copy.deepcopy(self.X_)
        nans = np.expand_dims(np.expand_dims(self.X_.nan_triangle(), 0), 0)
        k, v, o, d = self.X_.shape
        nans = nans * np.ones((k, v, o, d))
        nans = np.concatenate((nans, np.ones((k, v, o, 1))*np.nan), 3)
        nans = 1-np.nan_to_num(nans)
        obj.ddims = np.array([str(item) for item in obj.ddims]+['Ult'])
        obj.nan_override = True
        risk_arr = np.zeros((k, v, o, 1))
        if est == 'param_risk':
            obj.triangle = self._get_risk(nans, risk_arr,
                                          self.std_err_.triangle)
            self.parameter_risk_ = obj
        elif est == 'process_risk':
            obj.triangle = self._get_risk(nans, risk_arr,
                                          self.full_std_err_.triangle)
            self.process_risk_ = obj
        else:
            risk_arr = risk_arr[:, :, 0:1, :]
            obj.triangle = self._get_tot_param_risk(risk_arr)
            obj.odims = ['Total param risk']
            self.total_parameter_risk_ = obj

    def _get_risk(self, nans, risk_arr, std_err):
        for i in range(self.full_triangle_.shape[3]-1):
            t1 = (self.full_triangle_.triangle[:, :, :, i:i+1] *
                  std_err[:, :, :, i:i+1])**2
            t2 = (self.ldf_.triangle[:, :, :, i:i+1] *
                  risk_arr[:, :, :, i:i+1])**2
            t_tot = np.sqrt(t1+t2)*nans[:, :, :, i+1:i+2]
            risk_arr = np.concatenate((risk_arr, t_tot), 3)
        return risk_arr

    def _get_tot_param_risk(self, risk_arr):
        t1 = self.full_triangle_.triangle[:, :, :, :-1] - \
             np.nan_to_num(self.X_.triangle) + \
             np.nan_to_num(self.X_.get_latest_diagonal(False).triangle)
        t1 = np.expand_dims(np.sum(t1, 2), 2) * self.std_err_.triangle
        for i in range(self.full_triangle_.shape[3]-1):
            t_tot = np.sqrt((t1[:, :, :, i:i+1])**2 +
                            (self.ldf_.triangle[:, :, :, i:i+1] *
                             risk_arr[:, :, :, -1:])**2)
            risk_arr = np.concatenate((risk_arr, t_tot), 3)
        return risk_arr

    @property
    def mack_std_err_(self):
        obj = copy.deepcopy(self.parameter_risk_)
        obj.triangle = np.sqrt(self.parameter_risk_.triangle**2 +
                               self.process_risk_.triangle**2)
        obj.odims = ['Mack Std Err']
        return obj

    @property
    def total_mack_std_err_(self):
        return np.sqrt(self.total_process_risk_.triangle**2 +
                       self.total_parameter_risk_.triangle**2)[:, :, :, -1]
