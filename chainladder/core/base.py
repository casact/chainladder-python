# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
import sparse
from chainladder.utils.cupy import cp
import warnings

from chainladder.core.display import TriangleDisplay
from chainladder.core.dunders import TriangleDunders
from chainladder.core.pandas import TrianglePandas
from chainladder.core.slice import TriangleSlicer
from chainladder.core.io import TriangleIO


class TriangleBase(TriangleIO, TriangleDisplay, TriangleSlicer,
                   TriangleDunders, TrianglePandas):
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
            ' Instance with nothing set'
            return
        # Sanitize inputs
        index, columns, origin, development = self._str_to_list(
            index, columns, origin, development)
        key_gr = origin + self._flatten(development, index)

        # Aggregate data
        data_agg = data.groupby(key_gr).sum().reset_index().fillna(0)
        if not index:
            index = ['Total']
            data_agg[index[0]] = 'Total'
        # Initialize origin and development dates and grains
        origin_date = TriangleBase._to_datetime(
            data_agg, origin, format=origin_format)
        self.origin_grain = TriangleBase._get_grain(origin_date)
        m_cnt = {'Y': 12, 'Q': 3, 'M': 1}
        if development:
            development_date = TriangleBase._to_datetime(
                data_agg, development, period_end=True,
                format=development_format)
            self.development_grain = TriangleBase._get_grain(development_date)
            col = 'development'
        else:
            development_date = origin_date + \
                pd.tseries.offsets.MonthEnd(m_cnt[self.origin_grain])
            self.development_grain = self.origin_grain
            col = None
        # Prep the data for 4D Triangle
        self.valuation_date = development_date.max()
        origin_date = pd.PeriodIndex(origin_date, freq=self.origin_grain).to_timestamp()
        # Assign object properties
        date_axes = self._get_date_axes(origin_date, development_date) # cartesian product
        dev_lag_unique = TriangleBase._development_lag(date_axes['origin'], date_axes['development'])
        dev_lag = TriangleBase._development_lag(pd.Series(origin_date), pd.Series(development_date))
        dev = np.sort(dev_lag_unique.unique())
        orig = np.sort(date_axes['origin'].unique())
        key = data_agg[index].drop_duplicates().reset_index(drop=True)
        dev = dict(zip(dev, range(len(dev))))
        orig = dict(zip(orig, range(len(orig))))
        kdims = {v:k for k, v in key.sum(axis=1).to_dict().items()}
        orig_idx = origin_date.map(orig).values[np.newaxis].T
        if development:
            dev_idx = dev_lag.map(dev).values[np.newaxis].T
        else:
            dev_idx = (dev_lag*0).values[np.newaxis].T
        data_agg = data_agg[origin_date<=development_date]
        orig_idx = orig_idx[origin_date<=development_date]
        dev_idx = dev_idx[origin_date<=development_date]
        if sum(origin_date>development_date) > 0:
            warnings.warn("Observations with development before origin start have been removed.")
        key_idx = data_agg[index].sum(axis=1).map(kdims).values[np.newaxis].T
        val_idx = ((np.ones(len(data_agg))[np.newaxis].T)*range(len(columns))).reshape((1,-1), order='F').T
        coords = np.concatenate(tuple([np.concatenate((orig_idx, dev_idx), axis=1)]*len(columns)),  axis=0)
        coords = np.concatenate((np.concatenate(tuple([key_idx]*len(columns)),  axis=0), val_idx, coords), axis=1)
        amts = data_agg[columns].unstack().values.astype('float64')
        values = sparse.COO(coords.T, amts, shape=(len(key), len(columns), len(orig), len(dev) if development else 1)).todense()
        values[values==0.] = np.nan
        self.kdims = np.array(key)
        self.odims = np.sort(date_axes['origin'].unique())
        if development:
            self.ddims = np.sort(dev_lag_unique.unique())
            self.ddims = self.ddims*(m_cnt[self.development_grain])
        else:
            self.ddims = np.array([None])
        self.vdims = np.array(columns)
        self.key_labels = index
        self._set_slicers()
        # Create 4D Triangle
        xp = np
        if self.array_backend == 'numpy':
            xp = np
        else:
            xp = cp
            if cp == np:
                warnings.warn('Unable to load CuPY.  Using numpy instead.')
                self.array_backend = 'numpy'
        self.values = xp.array(values, dtype=kwargs.get('dtype', None))
        # Used to show NANs in lower part of triangle
        self.nan_override = False
        self.valuation = self._valuation_triangle()
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


    def _nan_triangle(self):
        '''Given the current triangle shape and grain, it determines the
           appropriate placement of NANs in the triangle for future valuations.
           This becomes useful when managing array arithmetic.
        '''
        xp = cp.get_array_module(self.values)
        if min(self.values.shape[2:]) == 1 or self.nan_override:
            return xp.ones(self.values.shape[2:], dtype='float16')
        if len(self.valuation) != len(self.odims)*len(self.ddims) or not \
           hasattr(self, '_nan_triangle_'):
            self.valuation = self._valuation_triangle()
            val_array = self.valuation
            val_array = val_array.values.reshape(self.shape[-2:], order='f')
            nan_triangle = xp.array(
                pd.DataFrame(val_array) > self.valuation_date)
            nan_triangle = xp.array(xp.where(nan_triangle, np.nan, 1), dtype='float16')
            self._nan_triangle_ = nan_triangle
        return self._nan_triangle_

    def _valuation_triangle(self, ddims=None):
        ''' Given origin and development, develop a triangle of valuation
        dates.
        '''
        ddims = self.ddims if ddims is None else ddims
        if type(ddims) == pd.DatetimeIndex:
            return pd.DatetimeIndex(pd.DataFrame(
                np.repeat(self.ddims.values[np.newaxis],
                          len(self.odims), 0))
                .unstack().values).to_period(self._lowest_grain()).to_timestamp(how='e')
        if ddims[0] is None:
            ddims = pd.Series([self.valuation_date]*len(self.origin))
            return pd.DatetimeIndex(ddims.values).to_period(self._lowest_grain()).to_timestamp(how='e')
        special_cases = dict(Ultimate='2262-03-01', Latest=self.valuation_date)
        if ddims[0] in special_cases.keys():
            return pd.DatetimeIndex(
                [pd.to_datetime(special_cases[ddims[0]])] *
                len(self.origin)).to_period(self._lowest_grain()).to_timestamp(how='e')
        if type(ddims[0]) in [np.str_, str]:
            ddims = np.array([int(item[:item.find('-'):]) for item in ddims])
        origin = pd.Series(self.odims)
        if type(self.valuation_date) is not pd.Timestamp:
            self.valuation_date = self.valuation_date.to_timestamp()
        # Limit origin to valuation date
        origin[origin > self.valuation_date] = self.valuation_date
        next_development = origin+pd.DateOffset(days=-1, months=ddims[0])
        val_array = np.array(next_development)[..., np.newaxis]
        ddim_arr = ddims - ddims[0]
        if ddims[-1] == 9999:
            val_array = (val_array.astype('datetime64[M]') +
                         ddim_arr[:-1][np.newaxis]).astype('datetime64[D]')
            next_development = pd.Series([pd.to_datetime('2262-03-01')] *
                                         len(origin)).values[..., np.newaxis]
            val_array = np.concatenate((val_array, next_development), -1)
        else:
            val_array = (val_array.astype('datetime64[M]') +
                         ddim_arr[np.newaxis]).astype('datetime64[D]')
        val_array = pd.DatetimeIndex(pd.DataFrame(val_array).unstack().values)
        return val_array.to_period('M').to_timestamp(how='e')

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
        return tuple([arg] if type(arg) is str else arg for arg in args)

    def _flatten(self, *args):
        return_list = []
        for item in args:
            if item:
                return_list = return_list + item
        return return_list
