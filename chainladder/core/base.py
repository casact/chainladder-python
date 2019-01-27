import pandas as pd
import numpy as np
from functools import wraps
import copy


def check_triangle_postcondition(f):
    ''' Post-condition check to ensure the integrity of the triangle object
        remains intact
    '''
    @wraps(f)
    def wrapper(*args, **kwargs):
        X = f(*args, **kwargs)
        if not hasattr(X, 'triangle'):
            raise ValueError('X is missing triangle attribute')
        if X.triangle.ndim != 4:
            raise ValueError('X.triangle must be a 4-dimensional array')
        if len(X.kdims) != X.triangle.shape[0]:
            raise ValueError('X.index and X.triangle are misaligned')
        if len(X.vdims) != X.triangle.shape[1]:
            print(X.vdims, X.shape)
            raise ValueError('X.columns and X.triangle are misaligned')
        return X
    return wrapper


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
            development_date = TriangleBase.to_datetime(data_agg, development, period_end=True)
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
            self.ddims = self.ddims*({'Y': 12, 'Q': 3, 'M': 1}[self.development_grain])
            self.vdims = np.array(data_agg.columns.levels[0].unique())
        else:
            self.ddims = np.array([None])
            self.vdims = np.array(data_agg.columns.unique())
        self.valuation_date = development_date.max()
        self.key_labels = index
        self.iloc = TriangleBase.Ilocation(self)
        self.loc = TriangleBase.Location(self)
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
    # ---------------------------------------------------------------- #
    # ----------------------- Class Properties ----------------------- #
    # ---------------------------------------------------------------- #
    @property
    def shape(self):
        ''' Returns 4D triangle shape '''
        return self.triangle.shape

    @property
    def index(self):
        return pd.DataFrame(list(self.kdims), columns=self.key_labels)

    @property
    def columns(self):
        return pd.Series(list(self.vdims), name='columns').to_frame()

    @property
    def origin(self):
        return pd.DatetimeIndex(self.odims, name='origin')
        return pd.Series(self.odims, name='origin') \
                 .dt.to_period(self.origin_grain).to_frame()

    @property
    def development(self):
        return pd.Series(list(self.ddims), name='development').to_frame()

    @property
    def latest_diagonal(self):
        return self.get_latest_diagonal()

    @property
    @check_triangle_postcondition
    def link_ratio(self):
        obj = copy.deepcopy(self)
        temp = obj.triangle.copy()
        temp[temp == 0] = np.nan
        obj.triangle = temp[..., 1:]/temp[..., :-1]
        obj.ddims = np.array([f'{obj.ddims[i]}-{obj.ddims[i+1]}'
                              for i in range(len(obj.ddims)-1)])
        # Check whether we want to eliminate the last origin period
        if np.max(np.sum(~np.isnan(self.triangle[..., -1, :]), 2)-1) == 0:
            obj.triangle = obj.triangle[..., :-1, :]
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
            np.cumsum(np.nan_to_num(self.triangle), axis=3, out=self.triangle)
            self.triangle = self.expand_dims(self.nan_triangle())*self.triangle
            self.triangle[self.triangle == 0] = np.nan
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

    @check_triangle_postcondition
    def grain(self, grain='', incremental=False, inplace=False):
        """Changes the grain of a cumulative triangle.

        Parameters
        ----------
        grain : str
            The grain to which you want your triangle converted, specified as
            'O<x>D<y>' where <x> and <y> can take on values of `['Y', 'Q', 'M']`
            For example, 'OYDY' for Origin Year/Development Year, 'OQDM' for
            Origin quarter, etc.
        incremental : bool
            Not implemented yet
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new one.

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
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.grain(grain=grain, incremental=incremental, inplace=True)
            return new_obj

    def trend(self, trend=0.0, axis=None):
        '''  Allows for the trending along origin or development
        '''
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

    def rename(self, index=None, columns=None, origin=None, development=None):
        if index is not None:
            self.kdims = [index] if type(index) is str else index
        if columns is not None:
            self.vdims = [columns] if type(columns) is str else columns
        if origin is not None:
            self.odims = [origin] if type(origin) is str else origin
        if development is not None:
            self.ddims = [development] if type(development) is str else development
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
            return default.replace('<th></th>\n      <th>1</th>',
                                   '<th>Origin</th>\n      <th>1</th>')
        else:
            data = pd.Series([self.valuation_date.strftime('%Y-%m'),
                             'O' + self.origin_grain + 'D'
                              + self.development_grain,
                              self.shape, self.key_labels, list(self.vdims)],
                             index=['Valuation:', 'Grain:', 'Shape',
                                    'index:', "columns:"],
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

    # ---------------------------------------------------------------- #
    # ----------------------- Pandas Passthrus ----------------------- #
    # ---------------------------------------------------------------- #
    def to_frame(self, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return self._repr_format()
        else:
            raise ValueError('len(index) and len(columns) must be 1.')

    def to_clipboard(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        self.to_frame().to_clipboard(*args, **kwargs)

    def to_csv(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        self.to_frame().to_csv(*args, **kwargs)

    def to_pickle(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        self.to_frame().to_pickle(*args, **kwargs)

    def to_excel(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        self.to_frame().to_excel(*args, **kwargs)

    def to_json(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        self.to_frame().to_json(*args, **kwargs)

    def to_html(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        self.to_frame().to_html(*args, **kwargs)

    def unstack(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        return self.to_frame().unstack(*args, **kwargs)

    # ---------------------------------------------------------------- #
    # ---------------------- Arithmetic Overload --------------------- #
    # ---------------------------------------------------------------- #
    def _validate_arithmetic(self, other):
        other = copy.deepcopy(other)
        if type(other) not in [int, float, np.float64, np.int64]:
            if len(self.vdims) != len(other.vdims):
                raise ValueError('Triangles must have the same number of \
                                  columns')
            if len(self.kdims) != len(other.kdims):
                raise ValueError('Triangles must have the same number of index')
            if len(self.vdims) == 1:
                other.vdims = np.array([None])
            other = other.triangle
        return other

    @check_triangle_postcondition
    def __add__(self, other):
        obj = copy.deepcopy(self)
        other = self._validate_arithmetic(other)
        obj.triangle = np.nan_to_num(self.triangle) + np.nan_to_num(other)
        obj.triangle = obj.triangle * self.expand_dims(self.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    @check_triangle_postcondition
    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    @check_triangle_postcondition
    def __sub__(self, other):
        obj = copy.deepcopy(self)
        other = self._validate_arithmetic(other)
        obj.triangle = np.nan_to_num(self.triangle) - \
            np.nan_to_num(other)
        obj.triangle = obj.triangle * self.expand_dims(self.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    @check_triangle_postcondition
    def __rsub__(self, other):
        obj = copy.deepcopy(self)
        other = self._validate_arithmetic(other)
        obj.triangle = np.nan_to_num(other) - \
            np.nan_to_num(self.triangle)
        obj.triangle = obj.triangle * self.expand_dims(self.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    def __len__(self):
        return self.shape[0]

    @check_triangle_postcondition
    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.triangle = -obj.triangle
        return obj

    @check_triangle_postcondition
    def __pos__(self):
        return self

    @check_triangle_postcondition
    def __mul__(self, other):
        obj = copy.deepcopy(self)
        other = self._validate_arithmetic(other)
        obj.triangle = np.nan_to_num(self.triangle)*other
        obj.triangle = obj.triangle * self.expand_dims(self.nan_triangle())
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    @check_triangle_postcondition
    def __rmul__(self, other):
        return self if other == 1 else self.__mul__(other)

    @check_triangle_postcondition
    def __truediv__(self, other):
        obj = copy.deepcopy(self)
        other = self._validate_arithmetic(other)
        obj.triangle = np.nan_to_num(self.triangle)/other
        obj.triangle[obj.triangle == 0] = np.nan
        obj.vdims = [None] if len(obj.vdims) == 1 else obj.vdims
        return obj

    @check_triangle_postcondition
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

    def sum(self, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return self.to_frame().sum(*args, **kwargs)
        return TriangleBase.TriangleGroupBy(self, by=-1).sum(axis=1)

    def groupby(self, by, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return self.to_frame().groupby(*args, **kwargs)
        return TriangleBase.TriangleGroupBy(self, by)

    class TriangleGroupBy:
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
            vdims = pd.Series(obj.vdims)
            obj.kdims = np.array(idx.index.unique())
            obj.vdims = np.array(vdims[vdims.isin(idx.columns.unique())])
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
        def __getitem__(self, key):
            idx = self.obj.idx_table().loc[key]
            idx = self.obj.idx_table_format(idx)
            return self.get_idx(idx)

    class Ilocation(LocBase):
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
        ''' Function for pandas style column indexing'''
        if type(key) is pd.DataFrame and 'development' in key.columns:
            return self._slice_development(key['development'])
        if type(key) is np.ndarray:
            return self._slice_origin(key)
        if type(key) is pd.Series:
            return self.iloc[list(self.index[key].index)]
        if key in self.key_labels:
            # Boolean-indexing of a particular key
            return self.index[key]
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
        obj.triangle = obj.triangle[..., key, :]
        return obj

    @check_triangle_postcondition
    def _slice_development(self, key):
        obj = copy.deepcopy(self)
        obj.ddims = obj.ddims[key]
        obj.triangle = obj.triangle[..., key]
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
           appropriate placement of NANs in the triangle for futurilocae valuations.
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
        if val_lag == 0:
            val_lag = grain_dict[self.origin_grain][self.development_grain]
        goods = (np.arange(self.triangle.shape[2]) *
                 grain_dict[self.origin_grain][self.development_grain] +
                 val_lag)[::-1]
        blank_bool = np.ones(self.triangle[0, 0].shape).cumsum(axis=1) <= \
            np.repeat(np.expand_dims(goods, axis=1),
                      self.triangle[0, 0].shape[1], axis=1)
        blank = (blank_bool*1.)
        blank[~blank_bool] = np.nan
        return blank

    def nan_triangle_x_latest(self):
        ''' Same as nan_triangle but sets latest diagonal to nan as well. '''
        nans = np.expand_dims(np.expand_dims(self.nan_triangle(), 0), 0)
        k, v, o, d = self.shape
        nans = nans * np.ones((k, v, o, d))
        nans = np.concatenate((nans, np.ones((k, v, o, 1))*np.nan), 3)
        nans[nans == 0] = np.nan
        return nans[0, 0, :, 1:]

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
        # The olf odims
        new_tri = np.swapaxes(np.sum(new_tri*o_bool, axis=2), -1, -2)
        return new_tri, o
