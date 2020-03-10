# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp

class TriangleDisplay():
    def __repr__(self):
        if (self.values.shape[0], self.values.shape[1]) == (1, 1):
            data = self._repr_format()
            return data.to_string()
        else:
            data = 'Valuation: ' + self.valuation_date.strftime('%Y-%m') + \
                   '\nGrain:     ' + 'O' + self.origin_grain + \
                                     'D' + self.development_grain + \
                   '\nShape:     ' + str(self.shape) + \
                   '\nIndex:      ' + str(self.key_labels) + \
                   '\nColumns:    ' + str(list(self.vdims))
            return data

    def _repr_html_(self):
        ''' Jupyter/Ipython HTML representation '''
        if (self.values.shape[0], self.values.shape[1]) == (1, 1):
            data = self._repr_format()
            if np.nanmean(abs(data)) < 10:
                fmt_str = '{0:,.4f}'
            elif np.nanmean(abs(data)) < 1000:
                fmt_str = '{0:,.2f}'
            else:
                fmt_str = '{:,.0f}'
            if len(self.ddims) > 1 and type(self.ddims[0]) is int:
                data.columns = [['Development Lag'] * len(self.ddims),
                                self.ddims]
            default = data.to_html(max_rows=pd.options.display.max_rows,
                                   max_cols=pd.options.display.max_columns,
                                   float_format=fmt_str.format) \
                          .replace('nan', '')
            return default.replace(
                '<th></th>\n      <th>{}</th>'.format(
                    list(data.columns)[0]),
                '<th>Origin</th>\n      <th>{}</th>'.format(
                    list(data.columns)[0]))
        else:
            data = pd.Series([self.valuation_date.strftime('%Y-%m'),
                             'O' + self.origin_grain + 'D'
                              + self.development_grain,
                              self.shape, self.key_labels, list(self.vdims)],
                             index=['Valuation:', 'Grain:', 'Shape:',
                                    'Index:', "Columns:"],
                             name='Triangle Summary').to_frame()
            pd.options.display.precision = 0
            return data.to_html(max_rows=pd.options.display.max_rows,
                                max_cols=pd.options.display.max_columns)

    def _repr_format(self):
        odims, ddims = self._repr_date_axes()
        if cp.get_array_module(self.values).__name__ == 'cupy':
            out = cp.asnumpy(self.values[0, 0])
        else:
            out = self.values[0, 0]
        out = pd.DataFrame(out, index=odims, columns=ddims)
        if str(out.columns[0]).find('-') > 0 and not \
           isinstance(out.columns, pd.PeriodIndex):
            out.columns = [item.replace('-9999', '-Ult')
                           for item in out.columns]
            if len(out.drop_duplicates()) != 1:
                return out
            else:
                return out.drop_duplicates().set_index(pd.Index(['(All)']))
        else:
            return out

    def _repr_date_axes(self):
        if type(self.odims[0]) == np.datetime64:
            odims = pd.Series(self.odims).dt.to_period(self.origin_grain)
        else:
            odims = pd.Series(self.odims)
        if len(self.ddims) == 1 and self.ddims[0] is None:
            ddims = list(self.vdims)
        elif self.is_val_tri:
            ddims = self.ddims.to_period(self.development_grain)
        else:
            ddims = self.ddims
        return odims, ddims
