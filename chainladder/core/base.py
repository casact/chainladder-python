import pandas as pd
import numpy as np
import copy


def check_triangle_postcondition(f):
    ''' Post-condition check to ensure the integrity of the triangle object
        remains intact
    '''
    def wrapper(*args, **kwargs):
        X = f(*args, **kwargs)
        if not hasattr(X, 'triangle'):
            raise ValueError('X is missing triangle attribute')
        if X.triangle.ndim != 4:
            raise ValueError('X.triangle must be a 4-dimensional array')
        if len(X.kdims) != X.triangle.shape[0]:
            raise ValueError('X.keys and X.triangle are misaligned')
        if len(X.vdims) != X.triangle.shape[1]:
            raise ValueError('X.values and X.triangle are misaligned')
        return X
    return wrapper


class TriangleBase:
    # ---------------------------------------------------------------- #
    # ----------------------- Class Properties ----------------------- #
    # ---------------------------------------------------------------- #
    @property
    def shape(self):
        ''' Returns 4D triangle shape '''
        return self.triangle.shape

    @property
    def keys(self):
        return pd.DataFrame(list(self.kdims), columns=self.key_labels)

    @property
    def values(self):
        return pd.Series(list(self.vdims), name='values')

    @property
    def origin(self):
        return pd.DatetimeIndex(self.odims, name='origin')

    @property
    def development(self):
        return pd.Series(list(self.ddims), name='development')

    @property
    def latest_diagonal(self):
        return self.get_latest_diagonal()

    @property
    @check_triangle_postcondition
    def link_ratio(self):
        obj = copy.deepcopy(self)
        temp = obj.triangle.copy()
        temp[temp == 0] = np.nan
        obj.triangle = temp[:, :, :, 1:]/temp[:, :, :, :-1]
        obj.ddims = np.array([f'{obj.ddims[i]}-{obj.ddims[i+1]}'
                              for i in range(len(obj.ddims)-1)])
        # Check whether we want to eliminate the last origin period
        if np.max(np.sum(~np.isnan(self.triangle[:, :, -1, :]), 2)-1) == 0:
            obj.triangle = obj.triangle[:, :, :-1, :]
            obj.odims = obj.odims[:-1]
        return obj

    @property
    def age_to_age(self):
        return self.link_ratio

    # ---------------------------------------------------------------- #
    # ---------------------- End User Methods ------------------------ #
    # ---------------------------------------------------------------- #
    @check_triangle_postcondition
    def get_latest_diagonal(self, compress=True):
        ''' Method to return the latest diagonal of the triangle.  Requires
            self.nan_overide == False.
        '''
        nan_tri = self.nan_triangle()
        latest_start = int(np.nansum(nan_tri[-1]))-1
        diagonal = nan_tri[:, latest_start:]*np.fliplr(np.fliplr(nan_tri[:, latest_start:]).T).T
        if latest_start != 0:
            filler = (np.ones((nan_tri.shape[0], 1))*np.nan)
            diagonal = np.concatenate((filler, diagonal), 1)
        diagonal = self.expand_dims(diagonal)*self.triangle
        obj = copy.deepcopy(self)
        if compress:
            diagonal = np.expand_dims(np.nansum(diagonal, 3), 3)
            obj.ddims = ['Latest']
        obj.triangle = diagonal
        return obj

    @check_triangle_postcondition
    def incr_to_cum(self, inplace=False):
        '''Method to convert an incremental triangle into a cumulative triangle.

            Arguments:
                inplace: bool
                    Set to True will update the instance data attribute inplace

            Returns:
                Updated instance of triangle accumulated along the origin
        '''

        if inplace:
            nan_tri = self.expand_dims(self.nan_triangle())
            self.triangle = np.nan_to_num(self.triangle).cumsum(axis=3)*nan_tri
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.incr_to_cum(inplace=True)

    @check_triangle_postcondition
    def cum_to_incr(self, inplace=False):
        '''Method to convert an cumlative triangle into a incremental triangle.

            Arguments:
                inplace: bool
                    Set to True will update the instance data attribute inplace

            Returns:
                Updated instance of triangle accumulated along the origin
        '''

        if inplace:
            nan_tri = self.expand_dims(self.nan_triangle())
            temp = np.nan_to_num(self.triangle)[:, :, :, 1:] - \
                np.nan_to_num(self.triangle)[:, :, :, :-1]
            temp = np.concatenate((self.triangle[:, :, :, 0:1], temp), axis=3)
            self.triangle = temp*nan_tri
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.cum_to_incr(inplace=True)

    def _set_ograin(self, grain, incremental):
        origin_grain = grain[1:2]
        orig = np.nan_to_num(self._slide(self.triangle))
        o_dt = pd.Series(self.odims)
        if origin_grain == 'Q':
            o = np.array(pd.to_datetime(o_dt.dt.year.astype(str) + 'Q' +
                                        o_dt.dt.quarter.astype(str)))
        elif origin_grain == 'Y':
            o = np.array(pd.to_datetime(o_dt.dt.year, format='%Y'))
        else:
            o = self.odims
        # Get unique new origin array
        o_new = np.unique(o)
        # Make it 2D so we have an old to new map
        o = np.repeat(np.expand_dims(o, axis=1), len(o_new), axis=1)
        o_new = np.repeat(np.expand_dims(o_new, axis=0), len(o), axis=0)
        # Make a boolean array out of the old to new map
        # and add ddims to it
        o_bool = np.repeat(np.expand_dims((o == o_new), axis=1),
                           len(self.ddims), axis=1)
        # Add kdims and vdims as well
        o_bool = self.expand_dims(o_bool)
        # Expand actual triangle to have o_new length on last axis
        new_tri = np.repeat(np.expand_dims(orig, axis=-1),
                            o_bool.shape[-1], axis=-1)
        # Multiply the triangle by the boolean array and aggregate out
        # The olf odims
        new_tri = np.swapaxes(np.sum(new_tri*o_bool, axis=2), -1, -2)
        return new_tri, o

    # ---------------------------------------------------------------- #
    # ------------------------ Display Options ----------------------- #
    # ---------------------------------------------------------------- #
    def __repr__(self):
        if (self.triangle.shape[0], self.triangle.shape[1]) == (1, 1):
            data = self._repr_format()
            return data.to_string()
        else:
            data = 'Valuation: ' + self.valuation_date.strftime('%Y-%m') + \
                   '\nGrain:     ' + 'O' + self.origin_grain + \
                                     'D' + self.development_grain + \
                   '\nShape:     ' + str(self.shape) + \
                   '\nKeys:      ' + str(self.key_labels) + \
                   '\nValues:    ' + str(list(self.vdims))
            return data

    def _repr_html_(self):
        ''' Jupyter/Ipython HTML representation '''
        if (self.triangle.shape[0], self.triangle.shape[1]) == (1, 1):
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
            return default.replace('<th></th>\n      <th>1</th>',
                                   '<th>Origin</th>\n      <th>1</th>')
            # .replace('<th></th>\n      <th colspan',
            #          '<th>Origin</th>\n      <th colspan')
        else:
            data = pd.Series([self.valuation_date.strftime('%Y-%m'),
                             'O' + self.origin_grain + 'D'
                              + self.development_grain,
                              self.shape, self.key_labels, list(self.vdims)],
                             index=['Valuation:', 'Grain:', 'Shape',
                                    'Keys:', "Values:"],
                             name='Triangle Summary').to_frame()
            pd.options.display.precision = 0
            return data.to_html(max_rows=pd.options.display.max_rows,
                                max_cols=pd.options.display.max_columns)

    def _repr_format(self):
        ''' Flatten to 2D DataFrame '''
        x = np.nansum(self.triangle, axis=0)
        x = np.nansum(x, axis=0)*self.nan_triangle()
        if type(self.odims[0]) == np.datetime64:
            origin = pd.Series(self.odims).dt.to_period(self.origin_grain)
        else:
            origin = pd.Series(self.odims)
        return pd.DataFrame(x, index=origin, columns=self.ddims)

    def to_clipboard(self):
        if (self.triangle.shape[0], self.triangle.shape[1]) == (1, 1):
            self._repr_format().to_clipboard()

    # ---------------------------------------------------------------- #
    # ---------------------- Arithmetic Overload --------------------- #
    # ---------------------------------------------------------------- #
    @check_triangle_postcondition
    def __add__(self, other):
        obj = copy.deepcopy(self)
        if type(other) not in [int, float]:
            other = other.triangle
        obj.triangle = np.nan_to_num(self.triangle) + np.nan_to_num(other)
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = np.array([None])
        return obj

    @check_triangle_postcondition
    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    @check_triangle_postcondition
    def __sub__(self, other):
        obj = copy.deepcopy(self)
        if type(other) not in [int, float]:
            other = other.triangle
        obj.triangle = np.nan_to_num(self.triangle) - \
            np.nan_to_num(other)
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = np.array([None])
        return obj

    @check_triangle_postcondition
    def __rsub__(self, other):
        obj = copy.deepcopy(self)
        if type(other) not in [int, float]:
            other = other.triangle
        obj.triangle = np.nan_to_num(other) - \
            np.nan_to_num(self.triangle)
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = np.array([None])
        return obj

    @check_triangle_postcondition
    def __mul__(self, other):
        obj = copy.deepcopy(self)
        obj.triangle = np.nan_to_num(self.triangle) * \
            np.nan_to_num(other.triangle)
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = np.array([None])
        return obj

    @check_triangle_postcondition
    def __truediv__(self, other):
        obj = copy.deepcopy(self)
        temp = other.triangle.copy()
        temp[temp == 0] = np.nan
        obj.vdims = np.array([None])
        obj.triangle = np.nan_to_num(self.triangle) / temp
        return obj

    @check_triangle_postcondition
    def __rtruediv__(self, other):
        obj = copy.deepcopy(self)
        obj.triangle = other / self.triangle
        return obj

    def __eq__(self, other):
        if np.all(np.nan_to_num(self.triangle) ==
           np.nan_to_num(other.triangle)):
            return True
        else:
            return False

    def sum(self):
        return TriangleBase.TriangleGroupBy(self, by=-1).sum(axis=1)

    def groupby(self, by):
        return TriangleBase.TriangleGroupBy(self, by)

    class TriangleGroupBy:
        def __init__(self, old_obj, by):
            obj = copy.deepcopy(old_obj)
            v1_len = len(obj.keys.index)
            if by != -1:
                indices = obj.keys.groupby(by).indices
                new_index = obj.keys.groupby(by).count().index
            else:
                indices = {'All': np.arange(len(obj.keys))}
                new_index = pd.Index(['All'], name='All')
            groups = list(indices.values())
            v2_len = len(groups)
            old_k_by_new_k = np.zeros((v1_len, v2_len))
            for num, item in enumerate(groups):
                old_k_by_new_k[:, num][item] = 1
            old_k_by_new_k = np.swapaxes(old_k_by_new_k, 0, 1)
            for i in range(3):
                old_k_by_new_k = np.expand_dims(old_k_by_new_k, axis=-1)
            new_tri = obj.triangle
            new_tri = np.repeat(np.expand_dims(new_tri, 0), v2_len, 0)

            obj.triangle = new_tri
            obj.kdims = np.array(list(new_index))
            obj.key_labels = list(new_index.names)
            self.obj = obj
            self.old_k_by_new_k = old_k_by_new_k

        @check_triangle_postcondition
        def sum(self, axis=1):
            self.obj.triangle = np.nansum((self.obj.triangle *
                                           self.old_k_by_new_k), axis=axis)
            self.obj.triangle[self.obj.triangle == 0] = np.nan
            return self.obj

    # ---------------------------------------------------------------- #
    # ----------------------Slicing and indexing---------------------- #
    # ---------------------------------------------------------------- #
    class LocBase:
        ''' Base class for pandas style indexing '''
        def __init__(self, obj):
            self.obj = obj

        @check_triangle_postcondition
        def get_idx(self, idx):
            obj = copy.deepcopy(self.obj)
            obj.kdims = np.array(idx.index.unique())
            obj.vdims = np.array(idx.columns.unique())
            obj.key_labels = list(idx.index.names)
            obj.iloc = TriangleBase.Ilocation(obj)
            obj.loc = TriangleBase.Location(obj)
            idx_slice = np.array(idx).flatten()
            x = tuple([np.unique(np.array(item))
                       for item in list(zip(*idx_slice))])
            obj.triangle = obj.triangle[x[0]][:, x[1]]
            obj.triangle[obj.triangle == 0] = np.nan
            return obj

    class Location(LocBase):
        ''' Class for pandas style .loc indexing '''
        def __getitem__(self, key):
            idx = self.obj.idx_table().loc[key]
            idx = self.obj.idx_table_format(idx)
            return self.get_idx(idx)

    class Ilocation(LocBase):
        ''' Class for pandas style .iloc indexing '''
        def __getitem__(self, key):
            idx = self.obj.idx_table().iloc[key]
            idx = self.obj.idx_table_format(idx)
            return self.get_idx(idx)

    def idx_table_format(self, idx):
        if type(idx) is pd.Series:
            # One row or one column selection is it k or v?
            if len(set(idx.index).intersection(set(self.vdims))) == len(idx):
                # One column selection
                idx = idx.to_frame().T
                idx.index.names = self.key_labels
            else:
                # One row selection
                idx = idx.to_frame()
        elif type(idx) is tuple:
            # Single cell selection
            idx = self.idx_table().iloc[idx[0]:idx[0] + 1,
                                        idx[1]:idx[1] + 1]
        return idx

    def idx_table(self):
        idx = self.kdims
        temp = pd.DataFrame(list(idx), columns=self.key_labels)
        for num, item in enumerate(self.vdims):
            temp[item] = list(zip(np.arange(len(temp)),
                              (np.ones(len(temp))*num).astype(int)))
        temp.set_index(self.key_labels, inplace=True)
        return temp

    def __getitem__(self, key):
        ''' Function for pandas style column indexing getting '''
        if type(key) is pd.Series:
            if key.name == 'development':
                return self._slice_development(key)
            else:
                # Boolean-indexing of all keys
                return self.iloc[list(self.keys[key].index)]
        if type(key) is np.ndarray:
            return self._slice_origin(key)
        if key in self.key_labels:
            # Boolean-indexing of a particular key
            return self.keys[key]
        idx = self.idx_table()[key]
        idx = self.idx_table_format(idx)
        return TriangleBase.LocBase(self).get_idx(idx)

    def __setitem__(self, key, value):
        ''' Function for pandas style column indexing setting '''
        idx = self.idx_table()
        idx[key] = 1
        self.vdims = np.array(idx.columns.unique())
        self.triangle = np.append(self.triangle, value.triangle, axis=1)

    @check_triangle_postcondition
    def append(self, obj, index):
        return_obj = copy.deepcopy(self)
        x = pd.DataFrame(list(return_obj.kdims), columns=return_obj.key_labels)
        new_idx = pd.DataFrame([index], columns=return_obj.key_labels)
        x = x.append(new_idx)
        x.set_index(return_obj.key_labels, inplace=True)
        return_obj.triangle = np.append(return_obj.triangle, obj.triangle,
                                        axis=0)
        return_obj.kdims = np.array(x.index.unique())
        return return_obj

    @check_triangle_postcondition
    def _slice_origin(self, key):
        obj = copy.deepcopy(self)
        obj.odims = obj.odims[key]
        obj.triangle = obj.triangle[:, :, key, :]
        return obj

    @check_triangle_postcondition
    def _slice_development(self, key):
        obj = copy.deepcopy(self)
        obj.ddims = obj.ddims[key]
        obj.triangle = obj.triangle[:, :, :, key]
        return obj

    # ---------------------------------------------------------------- #
    # ------------------- Data Ingestion Functions ------------------- #
    # ---------------------------------------------------------------- #

    def get_date_axes(self, origin_date, development_date):
        ''' Function to find any missing origin dates or development dates that
            would otherwise mess up the origin/development dimensions.
        '''
        def complete_date_range(origin_date, development_date,
                                origin_grain, development_grain):
            ''' Determines origin/development combinations in full.  Useful for
                when the triangle has holes in it. '''
            origin_unique = \
                pd.PeriodIndex(start=origin_date.min(),
                               end=origin_date.max(),
                               freq=origin_grain).to_timestamp()
            development_unique = \
                pd.PeriodIndex(start=origin_date.min(),
                               end=development_date.max(),
                               freq=development_grain).to_timestamp()
            development_unique = TriangleBase.period_end(development_unique)
            # Let's get rid of any development periods before origin periods
            cart_prod = TriangleBase.cartesian_product(origin_unique,
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
        cart_prod = cart_prod_o.append(cart_prod_d) \
                               .append(cart_prod_t).drop_duplicates()
        cart_prod = cart_prod[cart_prod['development'] >= cart_prod['origin']]
        return cart_prod

    def get_axes(self, data_agg, groupby, values,
                 origin_date, development_date):
        ''' Preps axes for the 4D triangle
        '''
        date_axes = self.get_date_axes(origin_date, development_date)
        kdims = data_agg[groupby].drop_duplicates()
        kdims['key'] = 1
        date_axes['key'] = 1
        all_axes = pd.merge(date_axes, kdims, on='key').drop('key', axis=1)
        data_agg = \
            all_axes.merge(data_agg, how='left',
                           left_on=['origin', 'development'] + groupby,
                           right_on=[origin_date, development_date] + groupby) \
                    .fillna(0)[['origin', 'development'] + groupby + values]
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
        if self.triangle.shape[2] == 1 or \
           self.triangle.shape[3] == 1 or \
           self.nan_override:
            # This is reserved for summary arrays, e.g. LDF, Diagonal, etc
            # and does not need nan overrides
            return np.ones(self.triangle.shape[2:])
        grain_dict = {'Y': {'Y': 1, 'Q': 4, 'M': 12},
                      'Q': {'Q': 1, 'M': 3},
                      'M': {'M': 1}}
        val_lag = self.triangle.shape[3] % \
            grain_dict[self.origin_grain][self.development_grain]
        val_lag = 1 if val_lag == 0 else val_lag
        goods = (np.arange(self.triangle.shape[2]) *
                 grain_dict[self.origin_grain][self.development_grain] +
                 val_lag)[::-1]
        blank_bool = np.ones(self.triangle[0, 0].shape).cumsum(axis=1) <= \
            np.repeat(np.expand_dims(goods, axis=1),
                      self.triangle[0, 0].shape[1], axis=1)
        blank = (blank_bool*1.)
        blank[~blank_bool] = np.nan
        return blank

    def _slide(self, triangle, direction='r'):
        ''' Facilitates swapping alignment of triangle between development
            period and development date. '''
        nan_tri = self.nan_triangle()
        r = (nan_tri.shape[1] - np.nansum(nan_tri, axis=1)).astype(int)
        r = -r if direction == 'l' else r
        k, v, rows, column_indices = \
            np.ogrid[:triangle.shape[0], :triangle.shape[1],
                     :triangle.shape[2], :triangle.shape[3]]
        r[r < 0] += nan_tri.shape[1]
        column_indices = column_indices - r[:, np.newaxis]
        return triangle[k, v, rows, column_indices]

    def expand_dims(self, tri_2d):
        '''Expands from one 2D triangle to full 4D object
        '''
        k = len(self.kdims)
        v = len(self.vdims)
        tri_3d = np.repeat(np.expand_dims(tri_2d, axis=0), v, axis=0)
        return np.repeat(np.expand_dims(tri_3d, axis=0), k, axis=0)

    @check_triangle_postcondition
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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
            target = TriangleBase.period_end(target)
        return target

    @staticmethod
    def development_lag(origin, development):
        ''' For tabular format, this will convert the origin/development
            difference to a development lag '''
        year_diff = development.dt.year - origin.dt.year
        development_grain = TriangleBase.get_grain(development)
        if development_grain == 'Y':
            return year_diff + 1
        if development_grain == 'Q':
            quarter_diff = development.dt.quarter - origin.dt.quarter
            return year_diff * 4 + quarter_diff + 1
        if development_grain == 'M':
            month_diff = development.dt.month - origin.dt.month
            return year_diff * 12 + month_diff + 1

    @staticmethod
    def period_end(array):
        if type(array) is not pd.DatetimeIndex:
            array_lookup = len(set(array.dt.month))
        else:
            array_lookup = len(set(array.month))
        offset = {12: pd.tseries.offsets.MonthEnd(),
                  4: pd.tseries.offsets.QuarterEnd(),
                  1: pd.tseries.offsets.YearEnd()}
        return array + offset[array_lookup]

    @staticmethod
    def get_grain(array):
        num_months = len(array.dt.month.unique())
        return {1: 'Y', 4: 'Q', 12: 'M'}[num_months]

    @staticmethod
    def cartesian_product(*arrays, pandas=False):
        '''A fast implementation of cartesian product, used for filling in gaps
        in triangles (if any)'''
        if pandas:
            # Pandas can support mixed datatypes, but is slower?
            arr = arrays[0].to_frame(index=[1]*len(arrays[0]))
            for num, array in enumerate(arrays):
                if num > 0:
                    temp = array.to_frame(index=[1]*len(array))
                arr.merge(temp, how='inner', left_index=True, right_index=True)
            return arr
        else:
            # Numpy approach needs all the same datatype.
            length_arrays = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([len(a) for a in arrays]+[length_arrays],
                           dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[..., i] = a
            arr = arr.reshape(-1, length_arrays)
            return arr
