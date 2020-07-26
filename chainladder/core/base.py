# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
import warnings

from chainladder.core.display import TriangleDisplay
from chainladder.core.dunders import TriangleDunders
from chainladder.core.pandas import TrianglePandas
from chainladder.core.slice import TriangleSlicer
from chainladder.core.io import TriangleIO
from chainladder.core.common import Common


class TriangleBase(TriangleIO, TriangleDisplay, TriangleSlicer,
                   TriangleDunders, TrianglePandas, Common):
    ''' This class handles the initialization of a triangle '''

    def __init__(self, data=None, origin=None, development=None,
                 columns=None, index=None, origin_format=None,
                 development_format=None, cumulative=None,
                 array_backend=None, *args, **kwargs):
        if array_backend is None:
            from chainladder import ARRAY_BACKEND
            self.array_backend = ARRAY_BACKEND
        else:
            self.array_backend = array_backend
        if data is None:
            # Instance with nothing set - Useful for piecemeal triangle creation
            return
        if columns:
            check = data[columns].dtypes
            check = [check] if check.__class__.__name__ == 'dtype' else check.to_list()
            if 'object' in check:
                raise TypeError("column attribute must be numeric.")
        # Sanitize inputs
        index, columns, origin, development = self._str_to_list(
            index, columns, origin, development)

        # Initialize origin and development dates and grains
        origin_date = TriangleBase._to_datetime(
            data, origin, format=origin_format)
        origin_date.name = 'origin'
        self.origin_grain = TriangleBase._get_grain(origin_date)
        origin_date = pd.PeriodIndex(origin_date, freq=self.origin_grain).to_timestamp()
        m_cnt = {'Y': 12, 'Q': 3, 'M': 1}
        if development:
            development_date = TriangleBase._to_datetime(
                data, development, period_end=True,
                format=development_format)
            self.development_grain = TriangleBase._get_grain(development_date)
            col = 'development'
        else:
            development_date = origin_date + \
                pd.tseries.offsets.MonthEnd(m_cnt[self.origin_grain])
            self.development_grain = self.origin_grain
            col = None
        development_date.name = 'development'

        # Aggregate data

        key_gr = [origin_date, development_date] + \
                 [data[item] for item in self._flatten(index)]
        data_agg = data[columns].groupby(key_gr).sum().reset_index().fillna(0)
        if not index:
            index = ['Total']
            data_agg[index[0]] = 'Total'
        for item in index:
            if pd.api.types.is_numeric_dtype(data_agg[item]):
                data_agg[item] = data_agg[item].astype(str)

        # Prep the data for 4D Triangle
        self.valuation_date = data_agg['development'].max()
        # Assign object properties
        date_axes = self._get_date_axes(data_agg['origin'], data_agg['development']) # cartesian product
        dev_lag_unique = TriangleBase._development_lag(date_axes['origin'], date_axes['development'])
        dev_lag = TriangleBase._development_lag(data_agg['origin'], data_agg['development'])
        dev = np.sort(dev_lag_unique.unique())
        orig = np.sort(date_axes['origin'].unique())
        key = data_agg[index].drop_duplicates().reset_index(drop=True)
        dev = dict(zip(dev, range(len(dev))))
        orig = dict(zip(orig, range(len(orig))))
        kdims = {v:k for k, v in key.sum(axis=1).to_dict().items()}
        orig_idx = data_agg['origin'].map(orig).values[None].T
        if development:
            dev_idx = dev_lag.map(dev).values[None].T
        else:
            dev_idx = (dev_lag*0).values[None].T

        data_agg = data_agg[data_agg['origin']<=data_agg['development']]
        orig_idx = orig_idx[data_agg['origin']<=data_agg['development']]
        dev_idx = dev_idx[data_agg['origin']<=data_agg['development']]
        if sum(data_agg['origin']>data_agg['development']) > 0:
            warnings.warn("Observations with development before origin start have been removed.")
        key_idx = data_agg[index].sum(axis=1).map(kdims).values[None].T
        val_idx = ((np.ones(len(data_agg))[None].T)*range(len(columns))).reshape((1,-1), order='F').T
        coords = np.concatenate(tuple([np.concatenate((orig_idx, dev_idx), axis=1)]*len(columns)),  axis=0)
        coords = np.concatenate((np.concatenate(tuple([key_idx]*len(columns)), axis=0), val_idx, coords), axis=1)
        amts = data_agg[columns].unstack()
        amts.loc[amts==0] = sp.nan
        amts = amts.values.astype('float64')
        values = sp(coords.T, amts, prune=True, fill_value=sp.nan,
                    shape=(len(key), len(columns), len(orig),
                           len(dev) if development else 1))
        self.kdims = np.array(key)
        self.key_labels = index
        for num, item in enumerate(index):
            if item in data.columns:
                if pd.api.types.is_numeric_dtype(data[item]):
                    self.kdims[:, num] = self.kdims[:, num].astype(data[item].dtype)
        self.odims = np.sort(date_axes['origin'].unique())
        if development:
            self.ddims = np.sort(dev_lag_unique.unique())
            self.ddims = self.ddims*(m_cnt[self.development_grain])
        else:
            self.ddims = np.array([None])
        self.vdims = np.array(columns)
        self._set_slicers()
        # Create 4D Triangle
        if self.array_backend == 'numpy':
            self.values = np.array(values.todense(), dtype=kwargs.get('dtype', None))
        elif self.array_backend == 'sparse':
            self.values=values
        else:
            xp = cp
            if cp == np:
                warnings.warn('Unable to load CuPY.  Using numpy instead.')
                self.array_backend = 'numpy'
            self.values = xp.array(values, dtype=kwargs.get('dtype', None))
        self.is_cumulative = cumulative

    def _len_check(self, x, y):
        if len(x) != len(y):
            raise ValueError(
                'Length mismatch: Expected axis has ',
                '{} elements, new values have'.format(len(x)),
                ' {} elements'.format(len(y)))

    def _get_date_axes(self, origin_date, development_date):
        ''' Function to find any missing origin dates or development dates that
            would otherwise mess up the origin/development dimensions.
        '''
        def complete_date_range(origin_date, development_date,
                                origin_grain, development_grain):
            ''' Determines origin/development combinations in full.  Useful for
                when the triangle has holes in it. '''
            origin_unique = pd.period_range(
                start=origin_date.min(),
                end=max(origin_date.max(), self.valuation_date),
                freq=origin_grain).to_timestamp()
            development_unique = pd.period_range(
                start=origin_date.min(),
                end=development_date.max(),
                freq=development_grain).to_timestamp(how='e')
            # Let's get rid of any development periods before origin periods
            cart_prod = TriangleBase._cartesian_product(
                origin_unique, development_unique)
            cart_prod = cart_prod[cart_prod[:, 0] <= cart_prod[:, 1], :]
            return pd.DataFrame(cart_prod, columns=['origin', 'development'])
        cart_prod_o = complete_date_range(
            pd.Series(origin_date.min()), development_date,
            self.origin_grain, self.development_grain)
        cart_prod_d = complete_date_range(
            origin_date, pd.Series(origin_date.max()),
            self.origin_grain, self.development_grain)
        cart_prod_t = pd.DataFrame({'origin': origin_date,
                                   'development': development_date})
        cart_prod = cart_prod_o.append(cart_prod_d, sort=True) \
                               .append(cart_prod_t, sort=True) \
                               .drop_duplicates()
        cart_prod = cart_prod[cart_prod['development'] >= cart_prod['origin']]
        return cart_prod

    @property
    def nan_triangle(self):
        '''Given the current triangle shape and valuation, it determines the
           appropriate placement of NANs in the triangle for future valuations.
           This becomes useful when managing array arithmetic.
        '''
        xp = cp.get_array_module(self.values)
        if min(self.values.shape[2:]) == 1:
            return xp.ones(self.values.shape[2:], dtype='float16')
        val_array = np.array(self.valuation).reshape(self.shape[-2:], order='f')
        nan_triangle = np.array(
            pd.DataFrame(val_array) > self.valuation_date)
        nan_triangle = xp.array(np.where(nan_triangle, xp.nan, 1), dtype='float16')
        return nan_triangle

    def _lowest_grain(self):
        my_list = ['M', 'Q', 'Y']
        my_dict = {item: num for num, item in enumerate(my_list)}
        lowest_grain = my_list[min(my_dict[self.origin_grain],
                                   my_dict[self.development_grain])]
        return lowest_grain

    def _expand_dims(self, tri_2d):
        '''Expands from one 2D triangle to full 4D object'''
        xp = cp.get_array_module(tri_2d)
        return xp.broadcast_to(
            tri_2d, (len(self.kdims), len(self.vdims), *tri_2d.shape))

    @staticmethod
    def _to_datetime(data, fields, period_end=False, format=None):
        '''For tabular form, this will take a set of data
        column(s) and return a single date array.  This function heavily
        relies on pandas, but does two additional things:
        1. It extends the automatic inference using date_inference_list
        2. it allows pd_to_datetime on a set of columns
        '''
        # Concat everything into one field
        if len(fields) > 1:
            target_field = data[fields].astype(str).apply(
                lambda x: '-'.join(x), axis=1)
        else:
            target_field = data[fields].iloc[:, 0]
        if hasattr(target_field, 'dt'):
            target = target_field
            if type(target.iloc[0]) == pd.Period:
                if period_end:
                    return target.dt.to_timestamp(how='e')
                else:
                    return target.dt.to_timestamp(how='s')
        else:
            datetime_arg = target_field.unique()
            date_inference_list = \
                [{'arg': datetime_arg, 'format': '%Y%m'},
                 {'arg': datetime_arg, 'format': '%Y'},
                 {'arg': datetime_arg, 'infer_datetime_format': True}]
            if format is not None:
                date_inference_list = [{'arg': datetime_arg, 'format': format}] + \
                                      date_inference_list
            for item in date_inference_list:
                try:
                    arr = dict(zip(datetime_arg, pd.to_datetime(**item)))
                    break
                except:
                    pass
            target = target_field.map(arr)
        if period_end:
                target = target.dt.to_period(
                    TriangleBase._get_grain(target)
                ).dt.to_timestamp(how='e')
        return target

    @staticmethod
    def _development_lag(origin, development):
        ''' For tabular format, this will convert the origin/development
            difference to a development lag '''
        year_diff = development.dt.year - origin.dt.year
        quarter_diff = development.dt.quarter - origin.dt.quarter
        month_diff = development.dt.month - origin.dt.month
        if np.all(origin != development):
            development_grain = TriangleBase._get_grain(development)
        else:
            development_grain = 'M'
        diffs = dict(Y=year_diff + 1,
                     Q=year_diff * 4 + quarter_diff + 1,
                     M=year_diff * 12 + month_diff + 1)
        return diffs[development_grain]

    @staticmethod
    def _get_grain(array):
        months = set(array.dt.month)
        grain = {**{1: 'Y', 4: 'Q'}, **{item: 'M' for item in range(5,13)}}
        return grain[len(months)]

    @staticmethod
    def _cartesian_product(*arrays):
        '''A fast implementation of cartesian product, used for filling in gaps
        in triangles (if any)'''
        arr = np.empty([len(a) for a in arrays]+[len(arrays)],
                       dtype=np.result_type(*arrays))
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        arr = arr.reshape(-1, len(arrays))
        return arr

    def _str_to_list(self, *args):
        return tuple([arg] if type(arg) in [str, pd.Period] else arg for arg in args)

    def _flatten(self, *args):
        return_list = []
        for item in args:
            if item:
                return_list = return_list + item
        return return_list

    def num_to_nan(self):
        xp = cp.get_array_module(self.values)
        if xp == sp:
            self.values.fill_value = sp.nan
            self.values = sp(self.values)
        else:
            self.values[self.values == 0] = xp.nan
