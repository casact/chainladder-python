import pandas as pd
import numpy as np
import copy


from chainladder.core.base import TriangleBase


class Triangle(TriangleBase):
    """
    The core data structure of the chainladder package

    Parameters
    ----------
    data : DataFrame
        A single dataframe that contains columns represeting all other
        arguments to the Triangle constructor
    origin : str or list
         A representation of the accident, reporting or more generally the
         origin period of the triangle that will map to the Origin dimension
    development : str or list
        A representation of the development/valuation periods of the triangle
        that will map to the Development dimension
    columns : str or list
        A representation of the numeric data of the triangle that will map to
        the columns dimension.  If None, then a single 'Total' key will be
        generated.
    index : str or list or None
        A representation of the index of the triangle that will map to the
        index dimension.  If None, then a single 'Total' key will be generated.
    origin_format : optional str
        A string representation of the date format of the origin arg. If
        omitted then date format will be inferred by pandas.
    development_format : optional str
        A string representation of the date format of the development arg. If
        omitted then date format will be inferred by pandas.

    Attributes
    ----------
    index : Series
        Represents all available levels of the index dimension.
    columns : Series
        Represents all available levels of the value dimension.
    origin : DatetimeIndex
        Represents all available levels of the origin dimension.
    development : Series
        Represents all available levels of the development dimension.
    valuation : DatetimeIndex
        Represents all valuation dates of each cell in the Triangle.
    shape : tuple
        The 4D shape of the triangle instance
    link_ratio, age_to_age
        Displays age-to-age ratios for the triangle.
    valuation_date : date
        The latest valuation date of the data
    loc : Triangle
        pandas-style ``loc`` accessor
    iloc : Triangle
        pandas-style ``iloc`` accessor
    latest_diagonal : Triangle
        The latest diagonal of the triangle
    values : array
        4D numpy array underlying the Triangle instance
    T : Triangle
        Transpose index and columns of object.  Only available when Triangle is
        convertible to DataFrame.
    """
    @property
    def shape(self):
        return self.values.shape

    @property
    def index(self):
        return pd.DataFrame(list(self.kdims), columns=self.key_labels)

    @property
    def columns(self):
        return self._idx_table().columns

    @columns.setter
    def columns(self, value):
        self._len_check(self.columns, value)
        self.vdims = [value] if type(value) is str else value
        self.set_slicers()

    @property
    def origin(self):
        return pd.DatetimeIndex(self.odims, name='origin') \
                 .to_period(self.origin_grain)

    @origin.setter
    def origin(self, value):
        self._len_check(self.origin, value)
        value = pd.PeriodIndex([item for item in list(value)],
                               freq=self.origin_grain).to_timestamp()
        self.odims = value.values

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
    def link_ratio(self):
        obj = copy.deepcopy(self)
        temp = obj.values.copy()
        temp[temp == 0] = np.nan
        val_array = obj.valuation.to_timestamp().values.reshape(
            obj.shape[-2:], order='f')[:, 1:]
        obj.values = temp[..., 1:]/temp[..., :-1]
        obj.ddims = np.array(['{}-{}'.format(obj.ddims[i], obj.ddims[i+1])
                              for i in range(len(obj.ddims)-1)])
        # Check whether we want to eliminate the last origin period
        if np.max(np.sum(~np.isnan(self.values[..., -1, :]), 2)-1) == 0:
            obj.values = obj.values[..., :-1, :]
            obj.odims = obj.odims[:-1]
            val_array = val_array[:-1, :]
        obj.valuation = pd.DatetimeIndex(
            pd.DataFrame(val_array).unstack().values).to_period(self._lowest_grain())
        return obj

    @property
    def age_to_age(self):
        return self.link_ratio

    # ---------------------------------------------------------------- #
    # ---------------------- End User Methods ------------------------ #
    # ---------------------------------------------------------------- #
    def get_latest_diagonal(self, compress=True):
        ''' Method to return the latest diagonal of the triangle.  Requires
            self.nan_overide == False.
        '''
        obj = copy.deepcopy(self)
        diagonal = obj[obj.valuation == obj.valuation_date].values
        if compress:
            diagonal = np.expand_dims(np.nansum(diagonal, 3), 3)
            obj.ddims = np.array([None])
            obj.valuation = pd.DatetimeIndex(
                [pd.to_datetime(obj.valuation_date)] *
                len(obj.odims)).to_period(self._lowest_grain())
        obj.values = diagonal
        return obj

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
            np.cumsum(np.nan_to_num(self.values), axis=3, out=self.values)
            self.values = self.expand_dims(self.nan_triangle())*self.values
            self.values[self.values == 0] = np.nan
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.incr_to_cum(inplace=True)

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
            temp = np.nan_to_num(self.values)[..., 1:] - \
                np.nan_to_num(self.values)[..., :-1]
            temp = np.concatenate((self.values[..., 0:1], temp), axis=3)
            temp = temp*self.expand_dims(self.nan_triangle())
            temp[temp == 0] = np.nan
            self.values = temp
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.cum_to_incr(inplace=True)



    def dev_to_val(self, inplace=False):
        ''' Converts triangle from a development lag triangle to a valuation
        triangle.

        Returns
        -------
            Updated instance of triangle with valuation periods.
        '''
        if inplace:
            dev_mode = (type(self.ddims) == np.ndarray)
            self = self._val_dev_chg('dev_to_val') if dev_mode else self
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.dev_to_val(inplace=True)


    def val_to_dev(self, inplace=False):
        ''' Converts triangle from a valuation triangle to a development lag
        triangle.

        Returns
        -------
            Updated instance of triangle with development lags
        '''
        if inplace:
            dev_mode = (type(self.ddims) == np.ndarray)
            return self if dev_mode else self._val_dev_chg('val_to_dev')
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.val_to_dev(inplace=True)

    def _val_dev_chg(self, kind):
        obj = copy.deepcopy(self)
        o_vals = obj.expand_dims(np.arange(len(obj.origin))[:, np.newaxis])
        if self.shape[-1] == 1:
            return obj
        if kind == 'val_to_dev':
            step = {'Y': 12, 'Q': 3, 'M': 1}[obj.development_grain]
            mtrx = \
                12*(obj.ddims.to_timestamp(how='e').year.values[np.newaxis] -
                    obj.origin.to_timestamp(how='s')
                       .year.values[:, np.newaxis]) + \
                   (obj.ddims.to_timestamp(how='e').month.values[np.newaxis] -
                    obj.origin.to_timestamp(how='s')
                       .month.values[:, np.newaxis]) + 1
            rng = range(mtrx[mtrx > 0].min(), mtrx.max()+1, step)
        else:
            rng = obj.valuation.unique().sort_values()
        for item in rng:
            if kind == 'val_to_dev':
                val = np.where(mtrx == item)
            else:
                val = np.where(obj.expand_dims(obj.valuation == item)
                                  .reshape(obj.shape, order='f'))[-2:]
            val = np.unique(np.array(list(zip(val[0], val[1]))), axis=0)
            arr = np.expand_dims(obj.values[:, :, val[:, 0], val[:, 1]], -1)
            if val[0, 0] != 0:
                prepend = obj.expand_dims(np.array([np.nan]*(val[0, 0]))[:, np.newaxis])
                arr = np.concatenate((prepend, arr), -2)
            if len(obj.origin)-1-val[-1, 0] != 0:
                append = obj.expand_dims(
                    np.array([np.nan]*(len(obj.origin)-1-val[-1, 0]))[:, np.newaxis])
                arr = np.concatenate((arr, append), -2)
            o_vals = np.append(o_vals, arr, -1)
        obj.values = o_vals[..., 1:]
        keep = np.where(np.sum(
            (np.nansum(np.nansum(obj.values, 0), 0)), -2) !=
            len(obj.origin))[-1]
        obj.values = obj.values[..., keep]
        if kind == 'val_to_dev':
            obj.ddims = np.array([item for item in rng])
            obj.ddims = obj.ddims[keep]
            obj.valuation = obj._valuation_triangle()
        else:
            obj.ddims = obj.valuation.unique().sort_values()
            obj.ddims = obj.ddims[keep]
            obj.values = obj.values[..., :np.where(
                obj.ddims.to_timestamp() <= obj.valuation_date)[0].max()+1]
            obj.ddims = obj.ddims[obj.ddims.to_timestamp()
                                  <= obj.valuation_date]
            obj.valuation = pd.PeriodIndex(
                np.repeat(obj.ddims.values[np.newaxis],
                          len(obj.origin)).reshape(1, -1).flatten())
        return obj


    def grain(self, grain='', incremental=False, inplace=False):
        """Changes the grain of a cumulative triangle.

        Parameters
        ----------
        grain : str
            The grain to which you want your triangle converted, specified as
            'O<x>D<y>' where <x> and <y> can take on values of ``['Y', 'Q', 'M'
            ]`` For example, 'OYDY' for Origin Year/Development Year, 'OQDM'
            for Origin quarter/Development Month, etc.
        incremental : bool
            Grain does not work on incremental triangles and this argument let's
            the function know to make it cumuative before operating on the grain.
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Triangle
        """
        if inplace:
            origin_grain = grain[1:2]
            origin_grain_dict = {'Y':'%Y', 'Q':'%Y-Q%q','M':'%Y-%m'}
            dev_grain_dict = {'M': {'Y': 12, 'Q': 3, 'M': 1},
                            'Q': {'Y': 4, 'Q': 1},
                            'Y': {'Y': 1}}
            new_o = pd.period_range(
                start=self.origin.strftime(origin_grain_dict[origin_grain]).min(),
                end=self.valuation_date, freq=origin_grain)
            new_d = None
            if len(new_o)>len(self.odims) and len(self.ddims) > 1:
                addl_dev = dev_grain_dict[self.development_grain][self.origin_grain]*(len(new_o)-len(self.origin))
                print(addl_dev)
                new_d = np.append(
                    self.ddims,
                    [self.ddims[-1] + item * dev_grain_dict['M'][self.development_grain]
                    for item in range(1, addl_dev + 1)])
                new_tri = copy.deepcopy(self)
                new_tri.values = np.zeros((self.shape[0], self.shape[1], len(new_o), len(new_d)))
                new_tri.ddims = new_d
                new_tri.odims = np.array(new_o.start_time)
                new_tri.valuation = new_tri._valuation_triangle()
                self = self + new_tri
            development_grain = grain[-1]
            if incremental:
                self = self.incr_to_cum()
            dev_mode = (type(self.ddims) == np.ndarray)
            new_tri = self._set_ograin(grain=grain, dev_mode=dev_mode)
            # Set development Grain
            if self.shape[3] != 1:
                keeps = dev_grain_dict[self.development_grain][development_grain]
                keeps = np.where(np.arange(new_tri.shape[3]) % keeps == 0)[0]
                keeps = -(keeps + 1)[::-1]
                new_tri.values = new_tri.values[..., keeps]
                new_tri.ddims = new_tri.ddims[keeps]
            self.origin_grain = new_tri.origin_grain = origin_grain
            self.development_grain = new_tri.development_grain = development_grain
            new_tri.values[new_tri.values == 0] = np.nan
            if hasattr(self, '_nan_triangle'):
                # Force update on _nan_triangle at next access.
                del self._nan_triangle
            new_tri = new_tri.val_to_dev() if dev_mode else new_tri
            self.values, self.valuation = new_tri.values, new_tri.valuation
            self.odims, self.ddims = new_tri.odims, new_tri.ddims
            if incremental:
                self = self.cum_to_incr()
            return self if new_d is None else self[self.development<=new_d.max()]
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.grain(grain=grain, incremental=incremental,
                                 inplace=True)

    def trend(self, trend=0.0, axis='origin'):
        """  Allows for the trending of a Triangle object or an origin vector.
        This method trends using days and assumes a years is 365.25 days long.

        Parameters
        ----------
        trend : float
            The annual amount of the trend
        axis : str (options: ['origin', 'valuation'])
            The axis on which to apply the trend
        Returns
        -------
        Triangle
            updated with multiplicative trend applied.
        """
        days = np.datetime64(self.valuation_date)
        if axis == 'origin':
            trend = np.array((1 + trend)**-(
                pd.Series(self.origin.end_time.values-days).dt.days/365.25)
                )[np.newaxis, np.newaxis, ..., np.newaxis]
        elif axis == 'valuation':
            trend = (1 + trend)**-(
                pd.Series(self.valuation.end_time.values-days)
                .dt.days.values.reshape(self.shape[-2:], order='f')/365.25)
        obj = copy.deepcopy(self)
        obj.values = obj.values*trend
        return obj
