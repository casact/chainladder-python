import pandas as pd
import numpy as np
import copy
import joblib
from chainladder.core.display import TriangleDisplay
from chainladder.core.dunders import TriangleDunders
from chainladder.core.pandas import TrianglePandas, TriangleGroupBy
from chainladder.core.slice import Ilocation, Location, TriangleSlicer


class IO:
    ''' Class intended to allow persistence of triangle or estimator objects
        to disk
    '''
    def to_pickle(self, path, protocol=None):
        joblib.dump(self, filename=path, protocol=protocol)

class TriangleBase(IO, TriangleDisplay, TriangleSlicer, TriangleDunders, TrianglePandas):
    def __init__(self, data=None, origin=None, development=None,
                 columns=None, index=None):
        if data is None and origin is None and development is None and \
           columns is None and index is None:
            return
        # Sanitize Inputs
        columns = [columns] if type(columns) is str else columns
        origin = [origin] if type(origin) is str else origin
        if development is not None and type(development) is str:
            development = [development]
        key_gr = origin if not development else origin+development
        if not index:
            index = ['Total']
            data_agg = data.groupby(key_gr).sum().reset_index()
            data_agg[index[0]] = 'Total'
        else:
            index = [index] if type(index) is str else index
            data_agg = data.groupby(key_gr+index) \
                           .sum().reset_index()
        # Convert origin/development to dates
        origin_date = TriangleBase.to_datetime(data_agg, origin)
        self.origin_grain = TriangleBase._get_grain(origin_date)
        # These only work with valuation periods and not lags
        if development:
            development_date = TriangleBase.to_datetime(data_agg, development,
                                                        period_end=True)
            self.development_grain = TriangleBase._get_grain(development_date)
            col = 'development'
        else:
            development_date = origin_date
            self.development_grain = self.origin_grain
            col = None
        # Prep the data for 4D Triangle
        data_agg = self._get_axes(data_agg, index, columns,
                                 origin_date, development_date)
        data_agg = pd.pivot_table(data_agg, index=index+['origin'],
                                  columns=col, values=columns,
                                  aggfunc='sum')
        # Assign object properties
        self.kdims = np.array(data_agg.index.droplevel(-1).unique())
        self.odims = np.array(data_agg.index.levels[-1].unique())
        if development:
            self.ddims = np.array(data_agg.columns.levels[-1].unique())
            self.ddims = self.ddims*({'Y': 12, 'Q': 3, 'M': 1}
                                     [self.development_grain])
            self.vdims = np.array(data_agg.columns.levels[0].unique())
        else:
            self.ddims = np.array([None])
            self.vdims = np.array(data_agg.columns.unique())
        self.ddims = self.ddims
        self.valuation_date = development_date.max()
        self.key_labels = index
        self.iloc, self.loc = Ilocation(self), Location(self)
        # Create 4D Triangle
        triangle = \
            np.reshape(np.array(data_agg), (len(self.kdims), len(self.odims),
                       len(self.vdims), len(self.ddims)))
        triangle = np.swapaxes(triangle, 1, 2)
        # Set all 0s to NAN for nansafe ufunc arithmetic
        triangle[triangle == 0] = np.nan
        self.values = triangle
        # Used to show NANs in lower part of triangle
        self.nan_override = False
        self.valuation = self._valuation_triangle()

    # ---------------------------------------------------------------- #
    # ----------------------- Class Properties ----------------------- #
    # ---------------------------------------------------------------- #
    def _len_check(self, x, y):
        if len(x) != len(y):
            raise ValueError('Length mismatch: Expected axis has ',
                             '{} elements, new values have'.format(len(x)),
                             ' {} elements'.format(len(y)))


    # ---------------------------------------------------------------- #
    # ------------------- Data Ingestion Functions ------------------- #
    # ---------------------------------------------------------------- #

    def _get_date_axes(self, origin_date, development_date):
        ''' Function to find any missing origin dates or development dates that
            would otherwise mess up the origin/development dimensions.
        '''
        def complete_date_range(origin_date, development_date,
                                origin_grain, development_grain):
            ''' Determines origin/development combinations in full.  Useful for
                when the triangle has holes in it. '''
            origin_unique = \
                pd.period_range(start=origin_date.min(),
                                end=origin_date.max(),
                                freq=origin_grain).to_timestamp()
            development_unique = \
                pd.period_range(start=origin_date.min(),
                                end=development_date.max(),
                                freq=development_grain).to_timestamp()
            development_unique = TriangleBase._period_end(development_unique)
            # Let's get rid of any development periods before origin periods
            cart_prod = TriangleBase._cartesian_product(origin_unique,
                                                       development_unique)
            cart_prod = cart_prod[cart_prod[:, 0] <= cart_prod[:, 1], :]
            return pd.DataFrame(cart_prod, columns=['origin', 'development'])

        cart_prod_o = \
            complete_date_range(pd.Series(origin_date.min()), development_date,
                                self.origin_grain, self.development_grain)
        cart_prod_d = \
            complete_date_range(origin_date, pd.Series(origin_date.max()),
                                self.origin_grain, self.development_grain)
        cart_prod_t = pd.DataFrame({'origin': origin_date,
                                   'development': development_date})
        cart_prod = cart_prod_o.append(cart_prod_d, sort=True) \
                               .append(cart_prod_t, sort=True) \
                               .drop_duplicates()
        cart_prod = cart_prod[cart_prod['development'] >= cart_prod['origin']]
        return cart_prod

    def _get_axes(self, data_agg, groupby, columns,
                  origin_date, development_date):
        ''' Preps axes for the 4D triangle
        '''
        date_axes = self._get_date_axes(origin_date, development_date)
        kdims = data_agg[groupby].drop_duplicates()
        kdims['key'] = date_axes['key'] = 1
        all_axes = pd.merge(date_axes, kdims, on='key').drop('key', axis=1)
        data_agg = \
            all_axes.merge(data_agg, how='left',
                           left_on=['origin', 'development'] + groupby,
                           right_on=[origin_date, development_date] + groupby) \
                    .fillna(0)[['origin', 'development'] + groupby + columns]
        data_agg['development'] = \
            TriangleBase.development_lag(data_agg['origin'],
                                         data_agg['development'])
        return data_agg

    # ---------------------------------------------------------------- #
    # ------------------- Class Utility Functions -------------------- #
    # ---------------------------------------------------------------- #
    def nan_triangle(self):
        '''Given the current triangle shape and grain, it determines the
           appropriate placement of NANs in the triangle for future valuations.
           This becomes useful when managing array arithmetic.
        '''
        if self.values.shape[2] == 1 or \
           self.values.shape[3] == 1 or \
           self.nan_override:
            # This is reserved for summary arrays, e.g. LDF, Diagonal, etc
            # and does not need nan overrides
            return np.ones(self.values.shape[2:])
        if len(self.valuation) != len(self.odims)*len(self.ddims) or not \
           hasattr(self, '_nan_triangle'):
            self.valuation = self._valuation_triangle()
            val_array = self.valuation
            val_array = val_array.to_timestamp().values.reshape(self.shape[-2:], order='f')
            nan_triangle = np.array(
                pd.DataFrame(val_array) > self.valuation_date)
            nan_triangle = np.where(nan_triangle, np.nan, 1)
            self._nan_triangle = nan_triangle
        return self._nan_triangle

    def _valuation_triangle(self, ddims=None):
        ''' Given origin and development, develop a triangle of valuation
        dates.
        '''
        ddims = self.ddims if ddims is None else ddims
        if type(ddims) == pd.PeriodIndex:
            return
        if ddims[0] is None:
            ddims = pd.Series([self.valuation_date]*len(self.origin))
            return pd.DatetimeIndex(ddims.values).to_period(self._lowest_grain())
        special_cases = dict(Ultimate='2262-03-01', Latest=self.valuation_date)
        if ddims[0] in special_cases.keys():
            return pd.DatetimeIndex([pd.to_datetime(special_cases[ddims[0]])] *
                                    len(self.origin)).to_period(self._lowest_grain())
        if type(ddims[0]) is np.str_:
            ddims = [int(item[:item.find('-'):]) for item in ddims]
        origin = pd.PeriodIndex(self.odims, freq=self.origin_grain) \
                   .to_timestamp(how='s')
        origin = pd.Series(origin)
        if type(self.valuation_date) is not pd.Timestamp:
            self.valuation_date = self.valuation_date.to_timestamp()
        # Limit origin to valuation date
        origin[origin > self.valuation_date] = self.valuation_date
        next_development = origin+pd.DateOffset(days=-1, months=ddims[0])
        val_array = np.expand_dims(np.array(next_development), -1)
        for item in ddims[1:]:
            if item == 9999:
                next_development = pd.Series([pd.to_datetime('2262-03-01')] *
                                             len(origin))
                next_development = np.expand_dims(np.array(
                    next_development), -1)
            else:
                next_development = np.expand_dims(
                    np.array(origin+pd.DateOffset(days=-1, months=item)), -1)
            val_array = np.concatenate((val_array, next_development), -1)
        val_array = pd.DatetimeIndex(pd.DataFrame(val_array).unstack().values)
        return val_array.to_period(self._lowest_grain())

    def _lowest_grain(self):
        my_list = ['M', 'Q', 'Y']
        my_dict = {item: num for num, item in enumerate(my_list)}
        lowest_grain = my_list[min(my_dict[self.origin_grain],
                                   my_dict[self.development_grain])]
        return lowest_grain

    def expand_dims(self, tri_2d):
        '''Expands from one 2D triangle to full 4D object
        '''
        k, v = len(self.kdims), len(self.vdims)
        tri_3d = np.repeat(tri_2d[np.newaxis], v, axis=0)
        return np.repeat(tri_3d[np.newaxis], k, axis=0)

    @staticmethod
    def to_datetime(data, fields, period_end=False):
        '''For tabular form, this will take a set of data
        column(s) and return a single date array.
        '''
        # Concat everything into one field
        if len(fields) > 1:
            target_field = pd.Series(index=data.index).fillna('')
            for item in fields:
                target_field = target_field + data[item].astype(str)
        else:
            target_field = data[fields[0]]
        # pandas is not good at inferring YYYYMM format so trying that first
        # and if it fails, move on to how pandas infers things.
        datetime_arg = target_field.unique()
        date_inference_list = \
            [{'arg': datetime_arg, 'format': '%Y%m'},
             {'arg': datetime_arg, 'format': '%Y'},
             {'arg': datetime_arg, 'infer_datetime_format': True}]
        for item in date_inference_list:
            try:
                arr = dict(zip(datetime_arg, pd.to_datetime(**item)))
                break
            except:
                pass
        target = target_field.map(arr)
        if period_end:
            target = TriangleBase._period_end(target)
        target.name = 'valuation'
        return target

    @staticmethod
    def development_lag(origin, development):
        ''' For tabular format, this will convert the origin/development
            difference to a development lag '''
        year_diff = development.dt.year - origin.dt.year
        if np.all(origin != development):
            development_grain = TriangleBase._get_grain(development)
        else:
            development_grain = 'M'
        if development_grain == 'Y':
            return year_diff + 1
        if development_grain == 'Q':
            quarter_diff = development.dt.quarter - origin.dt.quarter
            return year_diff * 4 + quarter_diff + 1
        if development_grain == 'M':
            month_diff = development.dt.month - origin.dt.month
            return year_diff * 12 + month_diff + 1

    @staticmethod
    def _period_end(array):
        if type(array) is not pd.DatetimeIndex:
            array_lookup = len(set(array.dt.month))
        else:
            array_lookup = len(set(array.month))
        offset = {12: pd.tseries.offsets.MonthEnd(),
                  4: pd.tseries.offsets.QuarterEnd(),
                  1: pd.tseries.offsets.YearEnd()}
        return array + offset[array_lookup]

    @staticmethod
    def _get_grain(array):
        return {1: 'Y', 4: 'Q', 12: 'M'}[len(array.dt.month.unique())]

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

    def _set_ograin(self, grain, dev_mode):
        origin_grain = grain[1:2]
        orig = self.dev_to_val() if dev_mode else self
        o_dt = pd.Series(self.odims)
        if origin_grain == 'Q':
            o = np.array(pd.to_datetime(o_dt.dt.year.astype(str) + 'Q' +
                                        o_dt.dt.quarter.astype(str)))
        elif origin_grain == 'Y':
            o = np.array(pd.to_datetime(o_dt.dt.year, format='%Y'))
        else:
            o = self.odims
        o_new = np.unique(o)
        o = np.repeat(np.expand_dims(o, axis=1), len(o_new), axis=1)
        o_new = np.repeat(np.expand_dims(o_new, axis=0), len(o), axis=0)
        o_bool = np.repeat((o == o_new)[:, np.newaxis],
                           len(orig.ddims), axis=1)
        o_bool = self.expand_dims(o_bool)
        new_tri = np.repeat(np.expand_dims(np.nan_to_num(orig.values),
                            axis=-1), o_bool.shape[-1], axis=-1)
        new_tri = np.swapaxes(np.sum(new_tri*o_bool, axis=2), -1, -2)
        orig.values = new_tri
        orig.odims = np.unique(o)
        if type(orig.ddims) is list:
            orig.ddims = np.array(orig.ddims)
        if orig.shape[-1] == 1:
            orig.valuation = orig.valuation[:len(orig.odims)]
        else:
            orig.valuation = pd.PeriodIndex(
                np.repeat(orig.development.values[np.newaxis],
                          len(orig.origin)).reshape(1, -1).flatten())
        return orig
