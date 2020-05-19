# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
import copy

from chainladder.core.base import TriangleBase
from chainladder.core.correlation import DevelopmentCorrelation, ValuationCorrelation


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
    cumulative : bool
        Whether the triangle is cumulative or incremental.  This attribute is
        required to use the `grain` and `dev_to_val` methods and will be
        automatically set when invoking `cum_to_incr` or `incr_to_cum` methods.

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
    key_labels : list
        Represents the `index` axis labels
    valuation : DatetimeIndex
        Represents all valuation dates of each cell in the Triangle.
    origin_grain : str
        The grain of the origin vector ('Y', 'Q', 'M')
    development_grain : str
        The grain of the development vector ('Y', 'Q', 'M')
    shape : tuple
        The 4D shape of the triangle instance with axes corresponding to (index,
         columns, origin, development)
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
    is_cumulative: bool
        Whether the triangle is cumulative or not
    is_ultimate: bool
        Whether the triangle has an ultimate valuation
    is_full: bool
        Whether lower half of Triangle has been filled in
    is_val_tri:
        Whether the triangle development period is expressed as valuation
        periods.
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

    @index.setter
    def index(self, value):
        self._len_check(self.index, value)
        if type(value) is pd.DataFrame:
            self.kdims = value.values
            self._set_slicers()
        else:
            raise TypeError('index must be a pandas DataFrame')

    def set_index(self, value, inplace=False):
        """ Sets the index of the Triangle """
        if inplace:
            self.index = value
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.set_index(value=value, inplace=True)

    @property
    def columns(self):
        return self._idx_table().columns

    @columns.setter
    def columns(self, value):
        self._len_check(self.columns, value)
        self.vdims = [value] if type(value) is str else value
        if type(self.vdims) is list:
            self.vdims = np.array(self.vdims)
        self._set_slicers()

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
    def is_full(self):
        return self._nan_triangle().sum().sum() == np.prod(self.shape[-2:])

    @property
    def is_ultimate(self):
        return sum(self.valuation >= '2262') > 0

    @property
    def is_val_tri(self):
        return (type(self.ddims) == pd.DatetimeIndex)

    @property
    def latest_diagonal(self):
        """ The latest diagonal of the Triangle """
        return self._get_latest_diagonal()

    @property
    def link_ratio(self):
        xp = cp.get_array_module(self.values)
        obj = copy.deepcopy(self)
        if hasattr(obj, '_nan_triangle_'):
            del obj._nan_triangle_
        temp = obj.values.copy()
        temp[temp == 0] = np.nan
        val_array = obj.valuation.values.reshape(
            obj.shape[-2:], order='f')[:, 1:]
        obj.values = temp[..., 1:]/temp[..., :-1]
        obj.ddims = np.array(['{}-{}'.format(obj.ddims[i], obj.ddims[i+1])
                              for i in range(len(obj.ddims)-1)])
        # Check whether we want to eliminate the last origin period
        if xp.max(xp.sum(~xp.isnan(self.values[..., -1, :]), 2)-1) == 0:
            obj.values = obj.values[..., :-1, :]
            obj.odims = obj.odims[:-1]
            val_array = val_array[:-1, :]
        obj.valuation = pd.DatetimeIndex(
            pd.DataFrame(val_array).unstack().values).to_period(self._lowest_grain())
        if hasattr(obj, 'w_'):
            obj = obj*obj.w_[..., 0:1, :len(obj.odims), :]
        return obj

    @property
    def age_to_age(self):
        return self.link_ratio

    def _get_latest_diagonal(self, compress=True):
        ''' Method to return the latest diagonal of the triangle.  Requires
            self.nan_overide == False.

        Parameters
        ----------
        compress: bool
            Whether to collapse the diagonal into a single columns
        '''
        obj = copy.deepcopy(self)
        xp = cp.get_array_module(self.values)
        if not compress:
            offset = {'M': {'Y': 12, 'Q': 3, 'M': 1},
                      'Q': {'Q': 1, 'Y': 4},
                      'Y': {'Y': 1}}
            val_idx = self.valuation.unique().sort_values()
            val_idx = val_idx[val_idx <= self.valuation_date]
            val_idx = val_idx[-offset[self.development_grain][self.origin_grain]:]
            val_min, val_max = val_idx.min(), val_idx.max()
            diagonal = obj[obj.valuation >= val_min]
            diagonal = diagonal[diagonal.valuation <= val_max]
            return diagonal
        if compress:
            diagonal = obj[obj.valuation == obj.valuation_date].values
            diagonal = xp.expand_dims(xp.nansum(diagonal, 3), 3)
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
        xp = cp.get_array_module(self.values)
        if inplace:
            if not self.is_cumulative:
                xp.cumsum(xp.nan_to_num(self.values), axis=3, out=self.values)
                self.values = self._expand_dims(self._nan_triangle())*self.values
                self.values[self.values == 0] = np.nan
                self.is_cumulative = True
                self._set_slicers()
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
        xp = cp.get_array_module(self.values)
        if inplace:
            if self.is_cumulative:
                temp = xp.nan_to_num(self.values)[..., 1:] - \
                    xp.nan_to_num(self.values)[..., :-1]
                temp = xp.concatenate((self.values[..., 0:1], temp), axis=3)
                temp = temp*self._expand_dims(self._nan_triangle())
                temp[temp == 0] = np.nan
                self.values = temp
                self.is_cumulative = False
                self._set_slicers()
            return self
        else:
            new_obj = copy.deepcopy(self)
            return new_obj.cum_to_incr(inplace=True)

    def dev_to_val(self, inplace=False):
        ''' Converts triangle from a development lag triangle to a valuation
        triangle.

        Parameters
        ----------
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Updated instance of triangle with valuation periods.
        '''
        if inplace:
            if self.is_val_tri:
                return self
            self = self._val_dev_chg('dev_to_val') if not self.is_val_tri else self
            return self
        return self._val_dev_chg('dev_to_val') if not self.is_val_tri else copy.deepcopy(self)

    def val_to_dev(self, inplace=False):
        ''' Converts triangle from a valuation triangle to a development lag
        triangle.

        Parameters
        ----------
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Updated instance of triangle with development lags
        '''
        xp = cp.get_array_module(self.values)
        if not self.is_val_tri:
            ret_val = self
        else:
            if self.is_ultimate:
                obj = self[self.valuation<'2262']
                non_ultimates = obj.values
                obj = obj._val_dev_chg('val_to_dev')
                ultimate = self[self.valuation>='2262']
                max_val = self[self.valuation<'2262'][self.origin == self.origin.max()].valuation.max()
                max_dev = obj[obj.origin == obj.origin.max()]
                max_dev = max_dev[max_dev.valuation == max_val].ddims[0]
                obj = obj[obj.development <= max_dev]
                obj.values = xp.concatenate(
                    (obj.values, ultimate.values), axis=-1)
                obj.ddims = np.concatenate((obj.ddims, np.array([9999])))
                obj.valuation = obj._valuation_triangle(obj.ddims)
                obj.valuation_date = max(obj.valuation)
                ret_val = obj
            else:
                ret_val = self._val_dev_chg('val_to_dev')
        ret_val.values = self._expand_dims(ret_val._nan_triangle())*ret_val.values
        if inplace:
            self = ret_val
        return ret_val


    def _val_dev_chg(self, kind):
        xp = cp.get_array_module(self.values)
        obj = copy.deepcopy(self)
        if hasattr(obj, '_nan_triangle_'):
            del obj._nan_triangle_
        o_vals = obj._expand_dims(xp.arange(len(obj.origin))[:, xp.newaxis])
        if self.shape[-1] == 1:
            return obj
        if kind == 'val_to_dev':
            step = {'Y': 12, 'Q': 3, 'M': 1}[obj.development_grain]
            mtrx = \
                12*(obj.ddims.year.values[np.newaxis] -
                    obj.origin.to_timestamp(how='s')
                       .year.values[:, np.newaxis]) + \
                   (obj.ddims.month.values[np.newaxis] -
                    obj.origin.to_timestamp(how='s')
                       .month.values[:, np.newaxis]) + 1
            rng = range(mtrx[mtrx > 0].min(), mtrx.max()+1, step)
        else:
            rng = obj.valuation.unique().sort_values()
        old_arr = None
        for item in rng:
            if kind == 'val_to_dev':
                val = np.where(mtrx == item)
            else:
                val = np.where(obj._expand_dims(obj.valuation == item)
                                  .reshape(obj.shape, order='f'))[-2:]
            val = np.unique(np.array(list(zip(val[0], val[1]))), axis=0)
            arr = xp.expand_dims(obj.values[:, :, val[:, 0], val[:, 1]], -1)
            if val[0, 0] != 0:
                prepend = obj._expand_dims(xp.array([xp.nan]*(val[0, 0]))[:, xp.newaxis])
                arr = xp.concatenate((prepend, arr), -2)
            if len(obj.origin)-1-val[-1, 0] != 0:
                append = obj._expand_dims(
                    xp.array([np.nan]*(len(obj.origin)-1-val[-1, 0]))[:, xp.newaxis])
                arr = xp.concatenate((arr, append), -2)
            if obj.is_cumulative and old_arr is not None:
                arr = xp.isnan(arr)*xp.nan_to_num(old_arr) + xp.nan_to_num(arr)
            old_arr = arr.copy()
            o_vals = xp.concatenate((o_vals, arr), -1)
        obj.values = o_vals[..., 1:]
        obj.values[obj.values == 0] = xp.nan
        if kind == 'val_to_dev':
            obj.ddims = np.array([item for item in rng])
            obj.valuation = obj._valuation_triangle()
        else:
            obj.ddims = obj.valuation.unique().sort_values()
            if type(obj.ddims) == pd.PeriodIndex:
                obj.ddims = obj.ddims.to_timestamp()
            obj.values = obj.values[..., :np.where(
                obj.ddims <= obj.valuation_date)[0].max()+1]
            obj.ddims = obj.ddims[obj.ddims <= obj.valuation_date]
            obj.valuation = pd.DatetimeIndex(
                np.repeat(obj.ddims.values[np.newaxis],
                          len(obj.origin)).reshape(1, -1).flatten())
        obj._set_slicers()
        return obj


    def grain(self, grain='', trailing=False, inplace=False):
        """Changes the grain of a cumulative triangle.

        Parameters
        ----------
        grain : str
            The grain to which you want your triangle converted, specified as
            'OXDY' where X and Y can take on values of ``['Y', 'Q', 'M'
            ]`` For example, 'OYDY' for Origin Year/Development Year, 'OQDM'
            for Origin quarter/Development Month, etc.
        trailing : bool
            For partial years/quarters, trailing will set the year/quarter end to
            that of the latest available form the data.
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Triangle
        """
        if not self.is_cumulative:
            # Must be cumulative to work
            obj = self.incr_to_cum().dev_to_val(inplace=True)
        else:
            obj = self.dev_to_val(inplace=True)
        # put data in valuation mode
        ograin_new = grain[1:2]
        ograin_old = obj.origin_grain
        xp = cp.get_array_module(self.values)
        if ograin_new != ograin_old:
            o_dt = pd.Series(obj.odims)
            if trailing:
                mn = self.origin[-1].strftime('%b').upper() if trailing else 'DEC'
                if ograin_new == 'Q':
                    o = np.array(pd.PeriodIndex(self.origin, freq='Q-' + mn).to_timestamp(how='s'))
                elif ograin_new == 'Y':
                    o = np.array(pd.PeriodIndex(self.origin, freq='A-' + mn).to_timestamp(how='s'))
                else:
                    o = obj.odims
            else:
                if ograin_new == 'Q':
                    o = np.array(pd.to_datetime(
                        o_dt.dt.year.astype(str) + 'Q' + o_dt.dt.quarter.astype(str)))
                elif ograin_new == 'Y':
                    o = np.array(pd.to_datetime(o_dt.dt.year, format='%Y'))
                else:
                    o = obj.odims
            o_new = np.unique(o)
            o = np.repeat(np.expand_dims(o, axis=1), len(o_new), axis=1)
            o_new = np.repeat(o_new[np.newaxis], len(o), axis=0)
            o_bool = np.repeat((o == o_new)[:, np.newaxis],
                               len(obj.ddims), axis=1)
            o_bool = obj._expand_dims(o_bool)
            new_tri = xp.repeat(xp.nan_to_num(obj.values)[..., xp.newaxis],
                                o_bool.shape[-1], axis=-1)
            new_tri[~np.isfinite(new_tri)] = 0
            new_tri = np.swapaxes(np.sum(new_tri*o_bool, axis=2), -1, -2)
            obj.values = new_tri
            obj.odims = np.unique(o)
            obj.valuation = obj._valuation_triangle()
            if hasattr(obj, '_nan_triangle_'):
                del obj._nan_triangle_
        obj = obj.val_to_dev(inplace=True)
        # Now do development
        dev_grain_dict = {'M': {'Y': 12, 'Q': 3, 'M': 1},
                          'Q': {'Y': 4, 'Q': 1},
                          'Y': {'Y': 1}}
        dgrain_new = grain[-1]
        dgrain_old = obj.development_grain
        if obj.shape[3] != 1:
            keeps = dev_grain_dict[dgrain_old][dgrain_new]
            keeps = np.where(np.arange(obj.shape[3]) % keeps == 0)[0]
            keeps = -(keeps + 1)[::-1]
            obj.values = obj.values[..., keeps]
            obj.ddims = obj.ddims[keeps]
        obj.origin_grain = ograin_new
        obj.development_grain = dgrain_new
        obj.values[obj.values == 0] = xp.nan
        obj.valuation = obj._valuation_triangle()
        if hasattr(obj, '_nan_triangle_'):
            # Force update on _nan_triangle at next access.
            del obj._nan_triangle_
        if not self.is_cumulative:
            obj = obj.cum_to_incr()
        if self.is_val_tri:
            obj = obj.dev_to_val()
        if inplace:
            self = obj
            return self
        else:
            return obj


    def trend(self, trend=0.0, axis='origin', valuation_date=None, ultimate_lag=None):
        """  Allows for the trending of a Triangle object or an origin vector.
        This method trends using days and assumes a years is 365.25 days long.

        Parameters
        ----------
        trend : float
            The annual amount of the trend. Use 1/(1+trend)-1 to detrend.
        axis : str (options: ['origin', 'valuation'])
            The axis on which to apply the trend
        valuation_date: date
            The terminal date from which trend should be calculated.
        ultimate_lag : int
            If ultimate valuations are in the triangle, you can set the overall
            age of the ultimate to be some lag from the latest non-Ultimate
            development

        Returns
        -------
        Triangle
            updated with multiplicative trend applied.
        """
        xp = cp.get_array_module(self.values)
        if not valuation_date:
            days = np.datetime64(self.valuation_date)
        else:
            days = np.datetime64(valuation_date)
        if axis == 'origin':
            trend = xp.array((1 + trend)**-(
                pd.Series(self.origin.end_time.values-days).dt.days/365.25)
                )[xp.newaxis, xp.newaxis, ..., xp.newaxis]
        elif axis == 'valuation':
            valuation  = self.valuation
            if self.is_ultimate and ultimate_lag is not None:
                unit_lag = self.valuation[1] - self.valuation[0]
                val_df = pd.DataFrame(
                    self.valuation.values.reshape(self.shape[-2:], order='f'))
                val_df.iloc[:, -1] = val_df.iloc[:, -2] + unit_lag * ultimate_lag
                valuation = pd.PeriodIndex(val_df.unstack().values)
            trend = (1 + trend)**-(
                pd.Series(valuation.end_time.values-days)
                .dt.days.values.reshape(self.shape[-2:], order='f')/365.25)
        obj = copy.deepcopy(self)
        obj.values = obj.values*trend
        return obj

    def broadcast_axis(self, axis, value):
        """ Broadcasts (i.e. repeats) triangles along an axis.  The axis to be
        broadcast must be of length 1.

        Parameters
        ----------
        axis : str or int
            the axis to be broadcast over.
        value : axis-like
            The value of the new axis.

        TODO: Should convert value to a primitive type
        """
        obj = copy.deepcopy(self)
        axis = self._get_axis(axis)
        xp = cp.get_array_module(self.values)
        if self.shape[axis] != 1:
            raise ValueError('Axis to be broadcast must be of length 1')
        elif axis > 1:
            raise ValueError('Only index and column axes are supported')
        else:
            obj.values = xp.repeat(obj.values, len(value), axis)
            if axis == 0:
                obj.key_labels = list(value.columns)
                obj.kdims = value.values
                obj.index = value
            if axis == 1:
                obj.vdims = value.values
                obj.columns = value
        return obj

    def copy(self):
        return self.iloc[:, :]

    def development_correlation(self, p_critical=0.5):
        """
        Mack (1997) test for correlations between subsequent development
        factors. Results should be within confidence interval range
        otherwise too much correlation

        Parameters
        ----------
        p_critical: float (default=0.10)
            Value between 0 and 1 representing the confidence level for the test. A
            value of 0.1 implies 90% confidence.
        Returns
        -------
            DevelopmentCorrelation object with t, t_critical, t_expectation,
            t_variance, and range attributes.
        """
        return DevelopmentCorrelation(self, p_critical)

    def valuation_correlation(self, p_critical=.1, total=False):
        """
        Mack test for calendar year effect
        A calendar period has impact across developments if the probability of
        the number of small (or large) development factors in that period
        occurring randomly is less than p_critical

        Parameters
        ----------
        p_critical: float (default=0.10)
            Value between 0 and 1 representing the confidence level for the test
        total:
            Whether to calculate valuation correlation in total across all
            years (True) consistent with Mack 1993 or for each year separately
            (False) consistent with Mack 1997.
        Returns
        -------
            ValuationCorrelation object with z, z_critical, z_expectation and
            z_variance attributes.

        """
        return ValuationCorrelation(self, p_critical, total)
