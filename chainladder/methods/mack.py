# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import copy
from chainladder.methods import Chainladder


class MackChainladder(Chainladder):
    """ Basic stochastic chainladder method popularized by Thomas Mack

    Parameters
    ----------
    None

    Attributes
    ----------
    triangle
        returns **X**
    ultimate_
        The ultimate losses per the method
    ibnr_
        The IBNR per the method
    full_expectation_
        The ultimates back-filled to each development period in **X** replacing
        the known data
    full_triangle_
        The ultimates back-filled to each development period in **X** retaining
        the known data
    summary_
        summary of the model
    full_std_err_
        The full standard error
    total_process_risk_
        The total process error
    total_parameter_risk_
        The total parameter error
    mack_std_err_
        The total prediction error by origin period
    total_mack_std_err_
        The total prediction error across all origin periods

    """

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Data to which the model will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y, sample_weight)
        self._mack_recursion('param_risk')
        self._mack_recursion('process_risk')
        self._mack_recursion('total_param_risk')
        return self

    @property
    def full_std_err_(self):
        obj = copy.copy(self.X_)
        tri_array = self.full_triangle_.values
        weight_dict = {'regression': 0, 'volume': 1, 'simple': 2}
        val = np.array([weight_dict.get(item.lower(), 2)
                        for item in list(self.average_) + ['volume']])
        val = np.broadcast_to(val, self.X_.shape)
        weight = np.sqrt(tri_array[..., :len(self.X_.ddims)]**(2-val))
        weight[weight == 0] = np.nan
        obj.values = self.X_.sigma_.values / weight
        w = np.concatenate(
            (self.X_.w_, np.ones((*val.shape[:3], 1))*np.nan), axis=3)
        w[np.isnan(w)] = 1
        obj.values = np.nan_to_num(obj.values) * w
        obj.nan_override = True
        obj._set_slicers()
        return obj

    @property
    def total_process_risk_(self):
        origin = 2
        obj = copy.copy(self.process_risk_)
        obj.values = np.sqrt(np.nansum(self.process_risk_.values**2, origin))
        obj.values = np.expand_dims(obj.values, origin)
        obj.odims = ['tot_proc_risk']
        obj._set_slicers()
        return obj

    def _mack_recursion(self, est):
        obj = copy.copy(self.X_)
        nans = self.X_._nan_triangle()[np.newaxis, np.newaxis]
        nans = nans * np.ones(self.X_.shape)
        nans = np.concatenate(
            (nans, np.ones((*self.X_.shape[:3], 1))*np.nan), 3)
        nans = 1-np.nan_to_num(nans)
        properties = self.full_triangle_
        obj.valuation = properties.valuation
        obj.ddims = np.append(
            properties.ddims[:len(self.X_.ddims)],
            properties.ddims[-1])
        obj.nan_override = True
        risk_arr = np.zeros((*self.X_.shape[:3], 1))
        if est == 'param_risk':
            obj.values = self._get_risk(nans, risk_arr,
                                        obj.std_err_.values)
            obj._set_slicers()
            self.parameter_risk_ = obj
        elif est == 'process_risk':
            obj.values = self._get_risk(nans, risk_arr,
                                        self.full_std_err_.values)
            obj._set_slicers()
            self.process_risk_ = obj
        else:
            risk_arr = risk_arr[..., 0:1, :]
            obj.values = self._get_tot_param_risk(risk_arr)
            obj.odims = ['Total param risk']
            obj._set_slicers()
            self.total_parameter_risk_ = obj

    def _get_risk(self, nans, risk_arr, std_err):
        full_tri = np.nan_to_num(self.full_triangle_.values)[..., :len(self.X_.ddims)]
        t1_t = (full_tri * std_err)**2
        extend = self.X_.ldf_.shape[-1]-self.X_.shape[-1]+1
        ldf = self.X_.ldf_.values[..., :len(self.X_.ddims)-1]
        ldf = np.concatenate(
            (ldf, np.prod(self.X_.ldf_.values[..., -extend:], -1,
             keepdims=True)), -1)
        for i in range(len(self.X_.ddims)):
            t1 = t1_t[..., i:i+1]
            t2 = (ldf[..., i:i+1] * risk_arr[..., i:i+1])**2
            t_tot = np.sqrt(t1+t2)*nans[..., i+1:i+2]
            risk_arr = np.concatenate((risk_arr, t_tot), 3)
        return risk_arr

    def _get_tot_param_risk(self, risk_arr):
        """ This assumes triangle symmertry """
        t1 = np.nan_to_num(self.full_triangle_.values)[..., :len(self.X_.ddims)] - \
            np.nan_to_num(self.X_.values) + \
            np.nan_to_num(self.X_._get_latest_diagonal(False).values)
        t1 = np.sum(t1*self.X_.std_err_.values, axis=2, keepdims=True)
        extend = self.X_.ldf_.shape[-1]-self.X_.shape[-1]+1
        ldf = self.X_.ldf_.values[..., :len(self.X_.ddims)-1]
        ldf = np.concatenate(
            (ldf, np.prod(self.X_.ldf_.values[..., -extend:], -1,
             keepdims=True)), -1)
        ldf = np.unique(ldf, axis=-2)
        for i in range(self.full_triangle_.shape[-1]-1):
            t_tot = np.sqrt((t1[..., i:i+1])**2 + (ldf[..., i:i+1] *
                            risk_arr[..., -1:])**2)
            risk_arr = np.concatenate((risk_arr, t_tot), -1)
        return risk_arr

    @property
    def mack_std_err_(self):
        obj = copy.copy(self.parameter_risk_)
        obj.values = np.sqrt(self.parameter_risk_.values**2 +
                             self.process_risk_.values**2)
        obj._set_slicers()
        return obj

    @property
    def total_mack_std_err_(self):
        obj = copy.copy(self.X_.latest_diagonal)
        obj.values = np.sqrt(self.total_process_risk_.values**2 +
                             self.total_parameter_risk_.values**2)
        obj.values = obj.values[..., -1:]
        return pd.DataFrame(
            obj.values[..., 0, 0], index=obj.kdims,
            columns=[item for item in obj.vdims])

    @property
    def summary_(self):
        # This might be better as a dataframe
        obj = copy.copy(self.X_)
        obj.values = np.concatenate(
            (self.X_.latest_diagonal.values,
             self.ibnr_.values,
             self.ultimate_.values,
             self.mack_std_err_.values[..., -1:]), 3)
        obj.ddims = ['Latest', 'IBNR', 'Ultimate', 'Mack Std Err']
        obj.nan_override = True
        obj._set_slicers()
        return obj
