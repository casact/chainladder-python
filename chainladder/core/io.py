# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
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
        json_dict = {}
        attributes = ['values', 'kdims', 'vdims', 'odims', 'ddims']
        for attribute in attributes:
            json_dict[attribute] = {
                'dtype': str(getattr(self, attribute).dtype),
                'array': getattr(self, attribute).tolist()}
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
