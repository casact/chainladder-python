import pandas as pd
import numpy as np
from functools import wraps
import copy

# Pass through pd.DataFrame methods for a (1,1,o,d) shaped triangle:
df_passthru = ['to_clipboard', 'to_csv', 'to_pickle', 'to_excel', 'to_json',
               'to_html', 'to_dict', 'unstack', 'pivot', 'drop_duplicates',
               'describe', 'melt']

# Aggregate method overridden to the 4D Triangle Shape
agg_funcs = ['sum', 'mean', 'median', 'max', 'min', 'prod', 'var', 'std']
agg_funcs = {item: 'nan'+item for item in agg_funcs}

# def check_triangle_postcondition(f):
#     ''' Post-condition check to ensure the integrity of the triangle object
#         remains intact. (used for debugging)
#     '''
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         X = f(*args, **kwargs)
#         if not hasattr(X, 'triangle'):
#             raise ValueError('X is missing triangle attribute')
#         if X.triangle.ndim != 4:
#             raise ValueError('X.triangle must be a 4-dimensional array')
#         if len(X.kdims) != X.triangle.shape[0]:
#             raise ValueError('X.index and X.triangle are misaligned')
#         if len(X.vdims) != X.triangle.shape[1]:
#             raise ValueError('X.columns and X.triangle are misaligned')
#         return X
#     return wrapper



class TriangleBase:
    def __init__(self, data=None, origin=None, development=None,
                 columns=None, index=None):
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
            data_agg = data.groupby(key_gr+index) \
                           .sum().reset_index()
        # Convert origin/development to dates
        origin_date = TriangleBase.to_datetime(data_agg, origin)
        self.origin_grain = TriangleBase.get_grain(origin_date)
        # These only work with valuation periods and not lags
        if development:
            development_date = TriangleBase.to_datetime(data_agg, development,
                                                        period_end=True)
            self.development_grain = TriangleBase.get_grain(development_date)
            col = 'development'
        else:
            development_date = origin_date
            self.development_grain = self.origin_grain
            col = None
        # Prep the data for 4D Triangle
        data_agg = self.get_axes(data_agg, index, columns,
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
        self.iloc = _Ilocation(self)
        self.loc = _Location(self)
        # Create 4D Triangle
        triangle = \
            np.reshape(np.array(data_agg), (len(self.kdims), len(self.odims),
                       len(self.vdims), len(self.ddims)))
        triangle = np.swapaxes(triangle, 1, 2)
        # Set all 0s to NAN for nansafe ufunc arithmetic
        triangle[triangle == 0] = np.nan
        self.triangle = triangle
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

    @property
    def shape(self):
        return self.triangle.shape

    @property
    def index(self):
        return pd.DataFrame(list(self.kdims), columns=self.key_labels)

    @property
    def columns(self):
        return self.idx_table().columns

    @columns.setter
    def columns(self, value):
        self._len_check(self.columns, value)
        self.vdims = [value] if type(value) is str else value

    @property
    def origin(self):
        return pd.DatetimeIndex(self.odims, name='origin')

    @origin.setter
    def origin(self, value):
        self._len_check(self.origin, value)
        self.odims = [value] if type(value) is str else value

    @property
    def development(self):
        return pd.Series(list(self.ddims), name='development').to_frame()

    @development.setter
    def development(self, value):
        self._len_check(self.development, value)
        self.ddims = [value] if type(value) is str else value

    @property
    def latest_diagonal(self):
        return self.get_latest_diagonal()

    @property
    # @check_triangle_postcondition
    def link_ratio(self):
        obj = copy.deepcopy(self)
        temp = obj.triangle.copy()
        temp[temp == 0] = np.nan
        val_array = obj.valuation.values.reshape(obj.shape[-2:],order='f')[:, 1:]
        obj.triangle = temp[..., 1:]/temp[..., :-1]
        obj.ddims = np.array(['{}-{}'.format(obj.ddims[i], obj.ddims[i+1])
                              for i in range(len(obj.ddims)-1)])
        # Check whether we want to eliminate the last origin period
        if np.max(np.sum(~np.isnan(self.triangle[..., -1, :]), 2)-1) == 0:
            obj.triangle = obj.triangle[..., :-1, :]
            obj.odims = obj.odims[:-1]
            val_array = val_array[:-1, :]
        obj.valuation = pd.DatetimeIndex(pd.DataFrame(val_array).unstack().values)
        return obj

    @property
    def age_to_age(self):
        return self.link_ratio

    # ---------------------------------------------------------------- #
    # ---------------------- End User Methods ------------------------ #
    # ---------------------------------------------------------------- #
    # @check_triangle_postcondition
    def get_latest_diagonal(self, compress=True):
        ''' Method to return the latest diagonal of the triangle.  Requires
            self.nan_overide == False.
        '''
        obj = copy.deepcopy(self)
        diagonal = obj[obj.valuation == obj.valuation_date].triangle
        if compress:
            diagonal = np.expand_dims(np.nansum(diagonal, 3), 3)
            obj.ddims = ['Latest']
            obj.valuation = pd.DatetimeIndex(
                [pd.to_datetime(obj.valuation_date)]*len(obj.odims))
        obj.triangle = diagonal
        return obj

    # @check_triangle_postcondition
    def incr_to_cum(self, inplace=False):
        """Method to convert an incremental triangle into a cumulative triangle.

        Parameters
        ----------
        inplace: bool
            Set to True will update the instance data attribute inplace

        Returns
        -------
            Updated instance of triangle accumulated along the origin
        """

        if inplace:
            np.cumsum(np.nan_to_num(self.triangle), axis=3, out=self.triangle)
            self.triangle = self.expand_dims(self.nan_triangle())*self.triangle
            self.triangle[self.triangle == 0] = np.nan
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.incr_to_cum(inplace=True)

    # @check_triangle_postcondition
    def cum_to_incr(self, inplace=False):
        """Method to convert an cumlative triangle into a incremental triangle.

        Parameters
        ----------
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns
        -------
            Updated instance of triangle accumulated along the origin
        """

        if inplace:
            temp = np.nan_to_num(self.triangle)[..., 1:] - \
                np.nan_to_num(self.triangle)[..., :-1]
            temp = np.concatenate((self.triangle[..., 0:1], temp), axis=3)
            temp = temp*self.expand_dims(self.nan_triangle())
            temp[temp == 0] = np.nan
            self.triangle = temp
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.cum_to_incr(inplace=True)

    # @check_triangle_postcondition
    def grain(self, grain='', incremental=False, inplace=False):
        """Changes the grain of a cumulative triangle.

        Parameters
        ----------
        grain : str
            The grain to which you want your triangle converted, specified as
            'O<x>D<y>' where <x> and <y> can take on values of ``['Y', 'Q', 'M']``
            For example, 'OYDY' for Origin Year/Development Year, 'OQDM' for
            Origin quarter, etc.
        incremental : bool
            Not implemented yet
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Triangle
        """
        if inplace:
            origin_grain = grain[1:2]
            development_grain = grain[-1]
            new_tri, o = self._set_ograin(grain=grain, incremental=incremental)
            # Set development Grain
            dev_grain_dict = {'M': {'Y': 12, 'Q': 3, 'M': 1},
                              'Q': {'Y': 4, 'Q': 1},
                              'Y': {'Y': 1}}
            if self.shape[3] != 1:
                keeps = dev_grain_dict[self.development_grain][development_grain]
                keeps = np.where(np.arange(new_tri.shape[3]) % keeps == 0)[0]
                keeps = -(keeps + 1)[::-1]
                new_tri = new_tri[..., keeps]
                self.ddims = self.ddims[keeps]
            self.odims = np.unique(o)
            self.origin_grain = origin_grain
            self.development_grain = development_grain
            self.triangle = self._slide(new_tri, direction='l')
            self.triangle[self.triangle == 0] = np.nan
            self.valuation = self._valuation_triangle()
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.grain(grain=grain, incremental=incremental, inplace=True)
            return new_obj

    def trend(self, trend=0.0, axis=None):
        """  Allows for the trending along origin or development

        Parameters
        ----------
        trend : float
            The amount of the trend
        axis : str ('origin' or 'development')
            The axis along which to apply the trend factors.  The latest period
            of the axis is the trend-to period.

        Returns
        -------
            Triangle updated with multiplicative trend applied.
        """
        axis = {'origin': -2, 'development': -1}.get(axis, None)
        if axis is None:
            if self.shape[-2] == 1 and self.shape[-1] != 1:
                axis = -1
            elif self.shape[-2] != 1 and self.shape[-1] == 1:
                axis = -2
            else:
                raise ValueError('Cannot infer axis, please supply')
        trend = (1+trend)**np.arange(self.shape[axis])[::-1]
        trend = np.expand_dims(self.expand_dims(trend), -1)
        if axis == -1:
            trend = np.swapaxes(trend, -2, -1)
        obj = copy.deepcopy(self)
        obj.triangle = obj.triangle*trend
        return obj

    def rename(self, axis, value):
        if axis == 'index' or axis == 0:
            self.index = value
        if axis == 'columns' or axis == 1:
            self.columns = value
        if axis == 'origin' or axis == 2:
            self.origin = value
        if axis == 'development' or axis == 3:
            self.development = value
        return self

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
                   '\nindex:      ' + str(self.key_labels) + \
                   '\ncolumns:    ' + str(list(self.vdims))
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
            return default.replace(
                '<th></th>\n      <th>{}</th>'.format(self.development.values[0][0]),
                '<th>Origin</th>\n      <th>{}</th>'.format(self.development.values[0][0]))
        else:
            data = pd.Series([self.valuation_date.strftime('%Y-%m'),
                             'O' + self.origin_grain + 'D'
                              + self.development_grain,
                              self.shape, self.key_labels, list(self.vdims)],
                             index=['Valuation:', 'Grain:', 'Shape',
                                    'Index:', "Columns:"],
                             name='Triangle Summary').to_frame()
            pd.options.display.precision = 0
            return data.to_html(max_rows=pd.options.display.max_rows,
                                max_cols=pd.options.display.max_columns)

    def _repr_format(self):
        ''' Flatten to 2D DataFrame '''
        x = self.triangle[0, 0]
        if type(self.odims[0]) == np.datetime64:
            origin = pd.Series(self.odims).dt.to_period(self.origin_grain)
        else:
            origin = pd.Series(self.odims)
        return pd.DataFrame(x, index=origin, columns=self.ddims)

    # ---------------------------------------------------------------- #
    # ----------------------- Pandas Passthrus ----------------------- #
    # ---------------------------------------------------------------- #
    def to_frame(self, *args, **kwargs):
        """ Converts a triangle to a pandas.DataFrame.  Requires an individual
        index and column selection to appropriately grab the 2D DataFrame.

        Returns
        -------
            pandas.DataFrame representation of the Triangle.
        """
        axes = [num for num, item in enumerate(self.shape) if item > 1]
        if self.shape[:2] == (1, 1):
            return self._repr_format()
        elif len(axes) == 2 or len(axes) == 1:
            tri = np.squeeze(self.triangle)
            axes_lookup = {0: self.kdims, 1: self.vdims,
                           2: self.odims, 3: self.ddims}
            if len(axes) == 2:
                return pd.DataFrame(tri, index=axes_lookup[axes[0]],
                                    columns=axes_lookup[axes[1]])
            if len(axes) == 1:
                return pd.Series(tri, index=axes_lookup[axes[0]])
        else:
            raise ValueError('len(index) and len(columns) must be 1.')

    def plot(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        return self.to_frame().plot(*args, **kwargs)

    @property
    def T(self):
        """ Passthrough of pandas functionality """
        return self.to_frame().T

    # ---------------------------------------------------------------- #
    # ---------------------- Arithmetic Overload --------------------- #
    # ---------------------------------------------------------------- #
    def _validate_arithmetic(self, obj, other):
        other = copy.deepcopy(other)
        ddims = None
        odims = None
        if type(other) not in [int, float, np.float64, np.int64]:
            if len(self.vdims) != len(other.vdims):
                raise ValueError('Triangles must have the same number of \
                                  columns')
            if len(self.kdims) != len(other.kdims):
                raise ValueError('Triangles must have the same number of',
                                 'index')
            if len(self.vdims) == 1:
                other.vdims = np.array([None])
            # If broadcasting doesn't work, then try intersecting before
            # failure
            a, b = self.shape[-2:], other.shape[-2:]
            if not (a[0] == 1 or b[0] == 1 or a[0] == b[0]) and \
               not (a[1] == 1 or b[1] == 1 or a[1] == b[1]):
                ddims = set(self.ddims).intersection(set(other.ddims))
                odims = set(self.odims).intersection(set(other.odims))
                # Need to set string vs int type-casting
                obj = obj[obj.origin.isin(odims)][obj.development.isin(ddims)]
                other = other[other.origin.isin(odims)][other.development.isin(ddims)]
                obj.odims = np.sort(np.array(list(odims)))
                obj.ddims = np.sort(np.array(list(ddims)))
            other = other.triangle
        return obj, other

    # @check_triangle_postcondition
    def __add__(self, other):
        obj = copy.deepcopy(self)
        obj, other = self._validate_arithmetic(obj, other)
        obj.triangle = np.nan_to_num(obj.triangle) + np.nan_to_num(other)
        obj.triangle = obj.triangle * self.expand_dims(obj.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    # @check_triangle_postcondition
    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    # @check_triangle_postcondition
    def __sub__(self, other):
        obj = copy.deepcopy(self)
        obj, other = self._validate_arithmetic(obj, other)
        obj.triangle = np.nan_to_num(obj.triangle) - \
            np.nan_to_num(other)
        obj.triangle = obj.triangle * self.expand_dims(obj.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    # @check_triangle_postcondition
    def __rsub__(self, other):
        obj = copy.deepcopy(self)
        obj, other = self._validate_arithmetic(obj, other)
        obj.triangle = np.nan_to_num(other) - \
            np.nan_to_num(obj.triangle)
        obj.triangle = obj.triangle * self.expand_dims(obj.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    def __len__(self):
        return self.shape[0]

    # @check_triangle_postcondition
    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.triangle = -obj.triangle
        return obj

    # @check_triangle_postcondition
    def __pos__(self):
        return self

    # @check_triangle_postcondition
    def __mul__(self, other):
        obj = copy.deepcopy(self)
        obj, other = self._validate_arithmetic(obj, other)
        obj.triangle = np.nan_to_num(obj.triangle)*other
        obj.triangle = obj.triangle * self.expand_dims(obj.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    # @check_triangle_postcondition
    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    # @check_triangle_postcondition
    def __truediv__(self, other):
        obj = copy.deepcopy(self)
        obj, other = self._validate_arithmetic(obj, other)
        obj.triangle = np.nan_to_num(obj.triangle)/other
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    # @check_triangle_postcondition
    def __rtruediv__(self, other):
        obj = copy.deepcopy(self)
        obj.triangle = other / self.triangle
        obj.triangle[obj.triangle == 0] = np.nan
        return obj

    def __eq__(self, other):
        if np.all(np.nan_to_num(self.triangle) ==
           np.nan_to_num(other.triangle)):
            return True
        else:
            return False

    def quantile(self, q, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return self.to_frame().quantile(q, *args, **kwargs)
        return _TriangleGroupBy(self, by=-1).quantile(q, axis=1)

    def groupby(self, by, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return self.to_frame().groupby(*args, **kwargs)
        return _TriangleGroupBy(self, by)

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
        ''' Function for pandas style column indexing'''
        if type(key) is pd.DataFrame and 'development' in key.columns:
            return self._slice_development(key['development'])
        if type(key) is np.ndarray:
            # Presumes that if I have a 1D array, I will want to slice origin.
            if len(key) == self.shape[-2]*self.shape[-1] and self.shape[-1] > 1:
                return self._slice_valuation(key)
            return self._slice_origin(key)
        if type(key) is pd.Series:
            return self.iloc[list(self.index[key].index)]
        if key in self.key_labels:
            # Boolean-indexing of a particular key
            return self.index[key]
        idx = self.idx_table()[key]
        idx = self.idx_table_format(idx)
        return _LocBase(self).get_idx(idx)

    def __setitem__(self, key, value):
        ''' Function for pandas style column indexing setting '''
        idx = self.idx_table()
        idx[key] = 1
        self.vdims = np.array(idx.columns.unique())
        self.triangle = np.append(self.triangle, value.triangle, axis=1)

    # @check_triangle_postcondition
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

    # @check_triangle_postcondition
    def _slice_origin(self, key):
        obj = copy.deepcopy(self)
        obj.odims = obj.odims[key]
        obj.triangle = obj.triangle[..., key, :]
        return self._cleanup_slice(obj)

    # @check_triangle_postcondition
    def _slice_valuation(self, key):
        obj = copy.deepcopy(self)
        obj.valuation_date = obj.valuation[key].max()
        key = key.reshape(self.shape[-2:], order='f')
        nan_tri = np.ones(self.shape[-2:])
        nan_tri = key*nan_tri
        nan_tri[nan_tri == 0] = np.nan
        o, d = nan_tri.shape
        o_idx = np.arange(o)[list(np.sum(np.isnan(nan_tri), 1) != d)]
        d_idx = np.arange(d)[list(np.sum(np.isnan(nan_tri), 0) != o)]
        obj.odims = obj.odims[np.sum(np.isnan(nan_tri), 1) != d]
        if len(obj.ddims) > 1:
            obj.ddims = obj.ddims[np.sum(np.isnan(nan_tri), 0) != o]
        obj.triangle = (obj.triangle*nan_tri)
        obj.triangle = np.take(np.take(obj.triangle, o_idx, -2), d_idx, -1)
        return self._cleanup_slice(obj)

    # @check_triangle_postcondition
    def _slice_development(self, key):
        obj = copy.deepcopy(self)
        obj.ddims = obj.ddims[key]
        obj.triangle = obj.triangle[..., key]
        return self._cleanup_slice(obj)

    def _cleanup_slice(self, obj):
        obj.valuation = obj._valuation_triangle()
        if hasattr(obj, '_nan_triangle'):
            # Force update on _nan_triangle at next access.
            del obj._nan_triangle
            obj._nan_triangle = obj.nan_triangle()
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
                pd.period_range(start=origin_date.min(),
                                end=origin_date.max(),
                                freq=origin_grain).to_timestamp()
            development_unique = \
                pd.period_range(start=origin_date.min(),
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

    def get_axes(self, data_agg, groupby, columns,
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
        if self.triangle.shape[2] == 1 or \
           self.triangle.shape[3] == 1 or \
           self.nan_override:
            # This is reserved for summary arrays, e.g. LDF, Diagonal, etc
            # and does not need nan overrides
            return np.ones(self.triangle.shape[2:])
        if len(self.valuation) != len(self.odims)*len(self.ddims) or not \
           hasattr(self, '_nan_triangle'):
            self.valuation = self._valuation_triangle()
            val_array = self.valuation
            val_array = val_array.values.reshape(self.shape[-2:], order='f')
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
        if ddims[0] is None:
            ddims = pd.Series([self.valuation_date]*len(self.origin))
            return pd.DatetimeIndex(ddims.values)
        special_cases = dict(Ultimate='2262-03-01', Latest=self.valuation_date)
        if ddims[0] in special_cases.keys():
            return pd.DatetimeIndex([pd.to_datetime(special_cases[ddims[0]])] *
                                    len(self.origin))
        if type(ddims[0]) is np.str_:
            ddims = [int(item[:item.find('-'):]) for item in ddims]
        origin = pd.PeriodIndex(self.odims, freq=self.origin_grain) \
                   .to_timestamp(how='s')
        origin = pd.Series(origin)
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
        return pd.DatetimeIndex(pd.DataFrame(val_array).unstack().values)

    def _slide(self, triangle, direction='r'):
        ''' Facilitates swapping alignment of triangle between development
            period and development date. '''
        obj = copy.deepcopy(self)
        obj.triangle = triangle
        nan_tri = obj.nan_triangle()
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

    # @check_triangle_postcondition
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
        if np.all(origin != development):
            development_grain = TriangleBase.get_grain(development)
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
    def cartesian_product(*arrays):
        '''A fast implementation of cartesian product, used for filling in gaps
        in triangles (if any)'''
        length_arrays = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays]+[length_arrays],
                       dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        arr = arr.reshape(-1, length_arrays)
        return arr

    def _set_ograin(self, grain, incremental):
        origin_grain = grain[1:2]
        tri = np.nan_to_num(self.triangle)*self.nan_triangle()
        orig = np.nan_to_num(self._slide(tri))
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
        # The old odims
        new_tri = np.swapaxes(np.sum(new_tri*o_bool, axis=2), -1, -2)
        return new_tri, o


# ---------------------------------------------------------------- #
# ----------------------Slicing and indexing---------------------- #
# ---------------------------------------------------------------- #
class _LocBase:
    ''' Base class for pandas style indexing '''
    def __init__(self, obj):
        self.obj = obj

    # @check_triangle_postcondition
    def get_idx(self, idx):
        obj = copy.deepcopy(self.obj)
        vdims = pd.Series(obj.vdims)
        obj.kdims = np.array(idx.index.unique())
        obj.vdims = np.array(vdims[vdims.isin(idx.columns.unique())])
        obj.key_labels = list(idx.index.names)
        obj.iloc = _Ilocation(obj)
        obj.loc = _Location(obj)
        idx_slice = np.array(idx).flatten()
        x = tuple([np.unique(np.array(item))
                   for item in list(zip(*idx_slice))])
        obj.triangle = obj.triangle[x[0]][:, x[1]]
        obj.triangle[obj.triangle == 0] = np.nan
        return obj


class _Location(_LocBase):
    def __getitem__(self, key):
        idx = self.obj.idx_table().loc[key]
        idx = self.obj.idx_table_format(idx)
        return self.get_idx(idx)


class _Ilocation(_LocBase):
    def __getitem__(self, key):
        idx = self.obj.idx_table().iloc[key]
        idx = self.obj.idx_table_format(idx)
        return self.get_idx(idx)


# ---------------------------------------------------------------- #
# ---------------------Groupby Functionality---------------------- #
# ---------------------------------------------------------------- #
class _TriangleGroupBy:
    def __init__(self, old_obj, by):
        obj = copy.deepcopy(old_obj)
        v1_len = len(obj.index.index)
        if by != -1:
            indices = obj.index.groupby(by).indices
            new_index = obj.index.groupby(by).count().index
        else:
            indices = {'All': np.arange(len(obj.index))}
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

    def quantile(self, q, axis=1, *args, **kwargs):
        x = self.obj.triangle*self.old_k_by_new_k
        ignore_vector = np.sum(np.isnan(x), axis=1, keepdims=True) == \
            x.shape[1]
        x = np.where(ignore_vector, 0, x)
        self.obj.triangle = \
            getattr(np, 'nanpercentile')(x, q*100, axis=1, *args, **kwargs)
        self.obj.triangle[self.obj.triangle == 0] = np.nan
        return self.obj
# ---------------------------------------------------------------- #
# ------------------------- Meta methods ------------------------- #
# ---------------------------------------------------------------- #


def set_method(cls, func, k):
    ''' Assigns methods to a class '''
    func.__doc__ = 'Refer to pandas for ``{}`` functionality.'.format(k)
    func.__name__ = k
    setattr(cls, func.__name__, func)


def add_triangle_agg_func(cls, k, v):
    ''' Aggregate Overrides in Triangle '''
    def agg_func(self, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return getattr(pd.DataFrame, k)(self.to_frame(), *args, **kwargs)
        else:
            return getattr(_TriangleGroupBy(self, by=-1), k)(axis=1)
    set_method(cls, agg_func, k)


def add_groupby_agg_func(cls, k, v):
    ''' Aggregate Overrides in GroupBy '''
    def agg_func(self, axis=1, *args, **kwargs):
        x = self.obj.triangle*self.old_k_by_new_k
        ignore_vector = np.sum(np.isnan(x), axis=1, keepdims=True) == \
            x.shape[1]
        x = np.where(ignore_vector, 0, x)
        self.obj.triangle = \
            getattr(np, v)(x, axis=1, *args, **kwargs)
        self.obj.triangle[self.obj.triangle == 0] = np.nan
        return self.obj
    set_method(cls, agg_func, k)


def add_df_passthru(cls, k):
    '''Pass Through of pandas functionality '''
    def df_passthru(self, *args, **kwargs):
        return getattr(pd.DataFrame, k)(self.to_frame(), *args, **kwargs)
    set_method(cls, df_passthru, k)


for k, v in agg_funcs.items():
    add_triangle_agg_func(TriangleBase, k, v)
    add_groupby_agg_func(_TriangleGroupBy, k, v)

for item in df_passthru:
    add_df_passthru(TriangleBase, item)
