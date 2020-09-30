# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
import numpy as np
import copy
import warnings
from chainladder.core.base import TriangleBase
from chainladder.core.correlation import DevelopmentCorrelation, ValuationCorrelation
import datetime as dt


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
            self.key_labels = list(value.columns)
            self._set_slicers()
        else:
            raise TypeError("index must be a pandas DataFrame")

    def set_index(self, value, inplace=False):
        """ Sets the index of the Triangle """
        if inplace:
            self.index = value
            return self
        else:
            new_obj = self.copy()
            return new_obj.set_index(value=value, inplace=True)

    @property
    def columns(self):
        return pd.Index(self.vdims, name="columns")

    @columns.setter
    def columns(self, value):
        self._len_check(self.columns, value)
        self.vdims = [value] if type(value) is str else value
        if type(self.vdims) is list:
            self.vdims = np.array(self.vdims)
        self._set_slicers()

    @property
    def origin(self):
        return pd.DatetimeIndex(self.odims, name="origin").to_period(self.origin_grain)

    @origin.setter
    def origin(self, value):
        self._len_check(self.origin, value)
        value = pd.PeriodIndex(
            [item for item in list(value)], freq=self.origin_grain
        ).to_timestamp()
        self.odims = value.values

    @property
    def development(self):
        return pd.Series(list(self.ddims), name="development")

    @development.setter
    def development(self, value):
        self._len_check(self.development, value)
        self.ddims = [value] if type(value) is str else value

    @property
    def is_full(self):
        return self.nan_triangle.sum().sum() == np.prod(self.shape[-2:])

    @property
    def is_ultimate(self):
        from chainladder import ULT_VAL

        return sum(self.valuation >= ULT_VAL[:4]) > 0

    @property
    def is_val_tri(self):
        return type(self.ddims) == pd.DatetimeIndex

    @property
    def latest_diagonal(self):
        return self.dev_to_val().iloc[..., -1]

    @property
    def link_ratio(self):
        from chainladder.utils.utility_functions import num_to_nan

        xp = self.get_array_module()
        obj = self.copy()
        temp = num_to_nan(obj.values.copy())
        val_array = obj.valuation.values.reshape(obj.shape[-2:], order="f")[:, 1:]
        d = obj.ddims
        obj.ddims = ["{}-{}".format(d[i], d[i + 1]) for i in range(len(d) - 1)]
        obj.ddims = np.array(obj.ddims)
        obj.values = temp[..., 1:] / temp[..., :-1]
        if self.array_backend == "sparse":
            obj.values.shape = tuple(obj.values.coords.max(1) + 1)
        else:
            if xp.max(xp.sum(~xp.isnan(self.values[..., -1, :]), 2) - 1) <= 0:
                obj.values = obj.values[..., :-1, :]
        obj.odims = obj.odims[: obj.values.shape[2]]
        if hasattr(obj, "w_"):
            w_ = obj.w_[..., 0:1, : len(obj.odims), :]
            obj = obj * w_ if obj.shape == w_.shape else obj
        return obj

    @property
    def age_to_age(self):
        return self.link_ratio

    @property
    def valuation(self):
        from chainladder import ULT_VAL

        ddims = self.ddims
        is_val_tri = type(ddims) == pd.DatetimeIndex
        if is_val_tri:
            out = pd.DataFrame(np.repeat(self.ddims.values[None], len(self.odims), 0))
            return pd.DatetimeIndex(out.unstack().values)
        if type(ddims[0]) in [np.str_, str]:
            ddims = np.array([int(item[: item.find("-") :]) for item in ddims])
        ddim_arr = ddims - ddims[0]
        origin = np.minimum(self.odims, np.datetime64(self.valuation_date))
        val_array = origin.astype("datetime64[M]") + np.timedelta64(ddims[0], "M")
        val_array = val_array.astype("datetime64[ns]") - np.timedelta64(1, "ns")
        val_array = val_array[:, None]
        s = slice(None, -1) if ddims[-1] == 9999 else slice(None, None)
        val_array = (
            val_array.astype("datetime64[M]") + ddim_arr[s][None, :] + 1
        ).astype("datetime64[ns]") - np.timedelta64(1, "ns")
        if ddims[-1] == 9999:
            ult = np.repeat(np.datetime64(ULT_VAL), val_array.shape[0])[:, None]
            val_array = np.concatenate((val_array, ult,), axis=1,)
        return pd.DatetimeIndex(val_array.reshape(1, -1, order="F")[0])

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
        from chainladder.utils.utility_functions import num_to_nan

        xp = self.get_array_module()
        if inplace:
            if not self.is_cumulative:
                self.values = (
                    num_to_nan(xp.cumsum(xp.nan_to_num(self.values), axis=3))
                    * self.nan_triangle[None, None, ...]
                )
                self.is_cumulative = True
                self._set_slicers()
            return self
        else:
            new_obj = self.copy()
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
        xp = self.get_array_module()
        from chainladder.utils.utility_functions import num_to_nan

        if inplace:
            if self.is_cumulative or self.is_cumulative is None:
                temp = (
                    xp.nan_to_num(self.values)[..., 1:]
                    - xp.nan_to_num(self.values)[..., :-1]
                )
                temp = xp.concatenate(
                    (xp.nan_to_num(self.values[..., 0:1]), temp), axis=3
                )
                self.values = num_to_nan(temp * self.nan_triangle)
                self.is_cumulative = False
                self._set_slicers()
            return self
        else:
            new_obj = self.copy()
            return new_obj.cum_to_incr(inplace=True)

    def _dstep(self):
        return {
            "M": {"Y": 12, "Q": 3, "M": 1},
            "Q": {"Y": 4, "Q": 1},
            "Y": {"Y": 1},
        }

    def _val_dev(self, sign, inplace=False):
        backend = self.array_backend
        obj = self.set_backend("sparse")
        if not inplace:
            obj.values = obj.values.copy()
        scale = self._dstep()[obj.development_grain][obj.origin_grain]
        offset = np.arange(obj.shape[-2]) * scale
        offset = offset[obj.values.coords[-2]] * sign
        obj.values.coords[-1] = obj.values.coords[-1] + offset
        ddims = obj.valuation[obj.valuation <= obj.valuation_date]
        ddims = len(ddims.drop_duplicates())
        if obj.values.coords[-1].min() < 0:
            obj.values.coords[-1] = obj.values.coords[-1] - obj.values.coords[-1].min()
            ddims = np.max((len(np.unique(obj.values.coords[-1])), ddims))
        obj.values.shape = tuple(list(obj.shape[:-1]) + [ddims])
        obj = obj.set_backend(backend)
        return obj

    def dev_to_val(self, inplace=False):
        """ Converts triangle from a development lag triangle to a valuation
        triangle.

        Parameters
        ----------
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Updated instance of triangle with valuation periods.
        """
        from chainladder.utils.utility_functions import concat

        if self.is_val_tri:
            if inplace:
                return self
            else:
                return self.copy()
        is_cumulative = self.is_cumulative
        if self.is_ultimate:
            if is_cumulative:
                obj = self.cum_to_incr(inplace=inplace)
            else:
                obj = self.copy()
            ultimate = obj.iloc[..., -1:]
            obj = obj.iloc[..., :-1]
        else:
            obj = self
        obj = obj._val_dev(1, inplace)
        ddims = obj.valuation[obj.valuation <= obj.valuation_date]
        obj.ddims = ddims.drop_duplicates().sort_values()
        if self.is_ultimate:
            ultimate.ddims = pd.DatetimeIndex(ultimate.valuation[0:1])
            obj = concat((obj, ultimate), -1)
            if is_cumulative:
                obj = obj.incr_to_cum(inplace=inplace)
        return obj

    def val_to_dev(self, inplace=False):
        """ Converts triangle from a valuation triangle to a development lag
        triangle.

        Parameters
        ----------
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Updated instance of triangle with development lags
        """
        from chainladder.utils.utility_functions import concat

        if not self.is_val_tri:
            if inplace:
                return self
            else:
                return self.copy()
        if self.is_ultimate:
            ultimate = self.iloc[..., -1:]
            ultimate.ddims = np.array([9999])
            obj = self.iloc[..., :-1]._val_dev(-1, inplace)
        else:
            obj = self._val_dev(-1, inplace)
        val_0 = obj.valuation[0]
        if self.ddims.shape[-1] == 1 and self.ddims[0] == self.valuation_date:
            origin_0 = pd.to_datetime(obj.odims[-1])
        else:
            origin_0 = pd.to_datetime(obj.odims[0])
        lag_0 = (val_0.year - origin_0.year) * 12 + val_0.month - origin_0.month + 1
        scale = {"Y": 12, "Q": 3, "M": 1}[obj.development_grain]
        obj.ddims = np.arange(obj.values.shape[-1]) * scale + lag_0
        prune = obj[obj.origin == obj.origin.max()]
        if self.is_ultimate:
            obj = obj.iloc[..., : (prune.valuation <= prune.valuation_date).sum()]
            obj = concat((obj, ultimate), -1)
        return obj

    def grain(self, grain="", trailing=False, inplace=False):
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
            that of the latest available from the data.
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Triangle
        """
        ograin_old, ograin_new = self.origin_grain, grain[1:2]
        dgrain_old, dgrain_new = self.development_grain, grain[-1]
        valid = {"Y": ["Y"], "Q": ["Q", "Y"], "M": ["Y", "Q", "M"]}
        if ograin_new not in valid.get(ograin_old, []) or dgrain_new not in valid.get(
            dgrain_old, []
        ):
            raise ValueError("New grain not compatible with existing grain")
        if self.is_cumulative is None:
            raise AttributeError(
                "The is_cumulative attribute must be set before using grain method."
            )
        if valid["M"].index(ograin_new) > valid["M"].index(dgrain_new):
            raise ValueError("Origin grain must be coarser than development grain")
        # Start with origin
        obj = self.dev_to_val()
        xp = obj.get_array_module()
        if ograin_new != ograin_old:
            if trailing:
                mn = self.origin[-1].strftime("%b").upper() if trailing else "DEC"
                freq = "Q-" if ograin_new == "Q" else "A-"
                o = pd.PeriodIndex(self.origin, freq=freq + mn)
                o = np.array(o.to_timestamp(how="s"))
            else:
                freq = "%YQ%q" if ograin_new == "Q" else "%Y"
                o = pd.to_datetime(self.origin.strftime("freq")).values
            values = [
                getattr(obj.loc[..., i, :], "sum")(2, auto_sparse=False)
                .set_backend(self.array_backend)
                .values
                for i in self.origin.groupby(o).values()
            ]
            obj.values = xp.concatenate(values, 2)
            obj.odims = np.unique(o)
            obj.origin_grain = ograin_new
        if dgrain_old != dgrain_new and obj.shape[-1] > 1:
            step = self._dstep()[dgrain_old][dgrain_new]
            d = np.arange(0, len(obj.development), step)
            if obj.is_cumulative:
                obj = obj.iloc[..., d]
            else:
                obj = obj.val_to_dev()
                d = np.arange(0, len(obj.development), step)
                length = self._dstep()["M"][dgrain_old]
                d = np.repeat(((d + 1) * length)[::-1], step)[: len(obj.ddims)][::-1]
                values = [
                    getattr(obj.iloc[..., i], "sum")(3, auto_sparse=False)
                    .set_backend(self.array_backend)
                    .values
                    for i in obj.development.groupby(d).groups.values()
                ]
                obj.values = xp.concatenate(values, 3)
                obj.ddims = np.unique(d)
            obj.development_grain = dgrain_new
        obj = obj.dev_to_val() if self.is_val_tri else obj.val_to_dev()
        if inplace:
            self = obj
            return self
        return obj

    def trend(
        self,
        trend=0.0,
        axis="origin",
        start=None,
        end=None,
        ultimate_lag=None,
        **kwargs
    ):
        """  Allows for the trending of a Triangle object along either a valuation
        or origin axis.  This method trends using days and assumes a years is
        365.25 days long.

        Parameters
        ----------
        trend : float
            The annual amount of the trend. Use 1/(1+trend)-1 to detrend.
        axis : str (options: ['origin', 'valuation'])
            The axis on which to apply the trend
        start: date
            The terminal date from which trend should be calculated. If none is
            provided then the latest date of the triangle is used.
        end: date
            The terminal date to which the trend should be calculated. If none is
            provided then the earliest period of the triangle is used.
        ultimate_lag : int
            If ultimate valuations are in the triangle, you can set the overall
            age of the ultimate to be some lag from the latest non-Ultimate
            development

        Returns
        -------
        Triangle
            updated with multiplicative trend applied.
        """
        if kwargs.get("valuation_date", None):
            start = kwargs["valuation_date"]
            warnings.warn(
                "valuation_date is deprecated, and will be removed. Use start instead."
            )
        if axis not in ["origin", "valuation", 2, -2]:
            raise ValueError(
                "Only origin and valuation axes are supported for trending"
            )

        def val_vector(start, end, valuation):
            if end < start:
                val_start = np.maximum(valuation, start)
                val_end = np.maximum(valuation, end)
            else:
                val_start = np.minimum(valuation, start)
                val_end = np.minimum(valuation, end)
            return val_start, val_end

        xp = self.get_array_module()
        if not start:
            if axis == "valuation":
                start = np.datetime64(self.valuation_date)
            else:
                start = np.datetime64(np.max(self.odims))
        else:
            start = np.datetime64(start)
        if not end:
            if axis == "valuation":
                end = np.datetime64(np.min(self.valuation))
            else:
                end = np.datetime64(np.min(self.odims))
        else:
            end = np.datetime64(end)
        if axis in ["origin", 2, -2]:
            val_start, val_end = val_vector(start, end, self.origin.start_time.values)
            trend = xp.array(
                (1 + trend) ** -(pd.Series(val_end - val_start).dt.days / 365.25)
            )[None, None, ..., None]
        elif axis == "valuation":
            valuation = self.valuation
            if self.is_ultimate and ultimate_lag is not None:
                unit_lag = self.valuation[1] - self.valuation[0]
                val_df = pd.DataFrame(
                    self.valuation.values.reshape(self.shape[-2:], order="f")
                )
                val_df.iloc[:, -1] = val_df.iloc[:, -2] + unit_lag * ultimate_lag
                valuation = pd.PeriodIndex(val_df.unstack().values)
            val_start, val_end = val_vector(start, end, valuation)
            trend = xp.array(
                (1 + trend)
                ** -(
                    pd.Series(val_end - val_start).dt.days.values.reshape(
                        self.shape[-2:], order="f"
                    )
                    / 365.25
                )
            )
        obj = self.copy()
        obj.values = obj.values * trend
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
        obj = self.copy()
        axis = self._get_axis(axis)
        xp = self.get_array_module()
        if self.shape[axis] != 1:
            raise ValueError("Axis to be broadcast must be of length 1")
        elif axis > 1:
            raise ValueError("Only index and column axes are supported")
        else:
            obj.values = xp.repeat(obj.values.copy(), len(value), axis)
            if axis == 0:
                obj.key_labels = list(value.columns)
                obj.kdims = value.values
                obj.index = value
            if axis == 1:
                obj.vdims = value.values
                obj.columns = value
        return obj

    def copy(self):
        X = Triangle()
        X.__dict__.update(vars(self))
        X._set_slicers()
        return X

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

    def valuation_correlation(self, p_critical=0.1, total=False):
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
