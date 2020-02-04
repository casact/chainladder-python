# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
from scipy.sparse import coo_matrix
import json
import joblib


class TriangleIO():
    def to_pickle(self, path, protocol=None):
        joblib.dump(self, filename=path, protocol=protocol)

    def to_json(self):
        ''' Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format
        '''
        def sparse_out(tri):
            k, v, o, d = tri.shape
            coo = coo_matrix(np.nan_to_num(tri.values.reshape((k*v*o, d))))
            return json.dumps(dict(zip([str(item) for item in zip(coo.row, coo.col)], coo.data)))

        json_dict = {}
        attributes = ['kdims', 'vdims', 'odims', 'ddims']
        for attribute in attributes:
            json_dict[attribute] = {
                'dtype': str(getattr(self, attribute).dtype),
                'array': getattr(self, attribute).tolist()}
        if np.sum(np.nan_to_num(self.values)==0) / np.prod(self.shape) > 0.40:
            json_dict['values'] = {
                'dtype': str(self.values.dtype),
                'array': sparse_out(self.cum_to_incr()),
                'sparse': True}
        else:
            json_dict['values'] = {
                'dtype': str(self.values.dtype),
                'array': self.values.tolist(),
                'sparse': False}
        json_dict['key_labels'] = self.key_labels
        json_dict['origin_grain'] = self.origin_grain
        json_dict['development_grain'] = self.development_grain
        json_dict['nan_override'] = self.nan_override
        json_dict['is_cumulative'] = self.is_cumulative
        json_dict['valuation_date'] = self.valuation_date.strftime('%Y-%m-%d')
        return json.dumps(json_dict)


class EstimatorIO:
    ''' Class intended to allow persistence of estimator objects
    '''

    def to_pickle(self, path, protocol=None):
        joblib.dump(self, filename=path, protocol=protocol)

    def to_json(self):
        ''' Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format
        '''
        return json.dumps(
            {'params': self.get_params(),
             '__class__': self.__class__.__name__})

    def __contains__(self, value):
        if self.__dict__.get(value, None) is None:
            return False
        return True
