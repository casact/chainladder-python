# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
import json
import joblib


class TriangleIO():
    def to_pickle(self, path, protocol=None):
        ''' Serializes triangle object to pickle.

        Parameters
        ----------
        path : str
            File path and name of pickle object.
        protocol :
            The pickle protocol to use.

        '''
        joblib.dump(self, filename=path, protocol=protocol)

    def to_json(self):
        ''' Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format
        '''
        def sparse_out(tri):
            k, v, o, d = tri.shape
            xp = self.get_array_module(tri)
            if self.array_backend == 'sparse':
                coo = xp.nan_to_num(tri.reshape((k*v*o, d))).to_scipy_sparse()
            else:
                coo = coo_matrix(np.nan_to_num(tri.reshape((k*v*o, d))))
            return json.dumps(dict(zip([str(item) for item in zip(coo.row, coo.col)], coo.data)))

        json_dict = {}
        if self.is_val_tri:
            ddims = self.ddims
            json_dict['ddims'] = {
                'dtype': str(ddims.dtype),
                'array': ddims.values.tolist()}
            attributes = ['kdims', 'vdims', 'odims']
        else:
            attributes = ['kdims', 'vdims', 'odims', 'ddims']
        for attribute in attributes:
            json_dict[attribute] = {
                'dtype': str(getattr(self, attribute).dtype),
                'array': getattr(self, attribute).tolist()}
        xp = self.get_array_module()
        if self.array_backend == 'cupy':
            out = xp.asnumpy(self.cum_to_incr().values)
            xp = np
        else:
            out = self.cum_to_incr().values
        if xp.sum(xp.nan_to_num(out)==0) / xp.prod(self.shape) > 0.40 or \
            self.array_backend == 'sparse':
            json_dict['values'] = {
                'dtype': str(out.dtype),
                'array': sparse_out(out),
                'sparse': True}
        else:
            json_dict['values'] = {
                'dtype': str(out.dtype),
                'array': out.tolist(),
                'sparse': False}
        json_dict['key_labels'] = self.key_labels
        json_dict['origin_grain'] = self.origin_grain
        json_dict['development_grain'] = self.development_grain
        json_dict['is_cumulative'] = self.is_cumulative
        json_dict['is_val_tri'] = self.is_val_tri
        json_dict['valuation_date'] = self.valuation_date.strftime('%Y-%m-%d')
        sub_tris = [k for k, v in vars(self).items() if isinstance(v, TriangleIO)]
        json_dict['sub_tris'] = {
            sub_tri: getattr(self, sub_tri).to_json() for sub_tri in sub_tris}
        dfs = [k for k, v in vars(self).items() if isinstance(v, pd.DataFrame)]
        json_dict['dfs'] = {df: getattr(self, df).to_json() for df in dfs}
        dfs = [k for k, v in vars(self).items() if isinstance(v, pd.Series)]
        json_dict['dfs'].update({df: getattr(self, df).to_frame().to_json() for df in dfs})
        return json.dumps(json_dict)


class EstimatorIO:
    ''' Class intended to allow persistence of estimator objects
    '''

    def to_pickle(self, path, protocol=None):
        ''' Serializes triangle object to pickle.

        Parameters
        ----------
        path : str
            File path and name of pickle object.
        protocol :
            The pickle protocol to use.
        '''
        joblib.dump(self, filename=path, protocol=protocol)

    def to_json(self):
        ''' Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format
        '''
        params = self.get_params(deep=False)
        params = {k: v.to_json() if isinstance(v, BaseEstimator) else v
                  for k, v in params.items()}
        return json.dumps(
            {'params': params,
             '__class__': self.__class__.__name__})

    def __contains__(self, value):
        if self.__dict__.get(value, None) is None:
            return False
        return True
