# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
import numpy as np
import copy
import warnings
from chainladder.core.base import TriangleBase
from chainladder.utils.sparse import sp
from chainladder.core.slice import VirtualColumns
from chainladder.core.correlation import DevelopmentCorrelation, ValuationCorrelation
from chainladder.utils.utility_functions import concat, num_to_nan, num_to_value
from chainladder import options

try:
    import dask.bag as db
except:
    db = None


class Triangle(TriangleBase):
    """
    The core data structure of the chainladder package

    Parameters
    ----------
    data: DataFrame
        A single dataframe that contains columns represeting all other
        arguments to the Triangle constructor
    origin: str or list
         A representation of the accident, reporting or more generally the
         origin period of the triangle that will map to the Origin dimension
    development: str or list
        A representation of the development/valuation periods of the triangle
        that will map to the Development dimension
    columns: str or list
        A representation of the numeric data of the triangle that will map to
        the columns dimension.  If None, then a single 'Total' key will be
        generated.
    index: str or list or None
        A representation of the index of the triangle that will map to the
        index dimension.  If None, then a single 'Total' key will be generated.
    origin_format: optional str
        A string representation of the date format of the origin arg. If
        omitted then date format will be inferred by pandas.
    development_format: optional str
        A string representation of the date format of the development arg. If
        omitted then date format will be inferred by pandas.
    cumulative: bool
        Whether the triangle is cumulative or incremental.  This attribute is
        required to use the ``grain`` and ``dev_to_val`` methods and will be
        automatically set when invoking ``cum_to_incr`` or ``incr_to_cum`` methods.
    trailing: bool
        When partial origin periods are present, setting trailing to True will
        ensure the most recent origin period is a full period and the oldest
        origin is partial. If full origin periods are present in the data, then
        trailing has no effect.

    Attributes
    ----------
    index: Series
        Represents all available levels of the index dimension.
    columns: Series
        Represents all available levels of the value dimension.
    origin: DatetimeIndex
        Represents all available levels of the origin dimension.
    development: Series
        Represents all available levels of the development dimension.
    key_labels: list
        Represents the ``index`` axis labels
    virtual_columns: Series
        Represents the subset of columns of the triangle that are virtual.
    valuation: DatetimeIndex
        Represents all valuation dates of each cell in the Triangle.
    origin_grain: str
        The grain of the origin vector ('Y', 'S', 'Q', 'M')
    development_grain: str
        The grain of the development vector ('Y', 'S', 'Q', 'M')
    shape: tuple
        The 4D shape of the triangle instance with axes corresponding to (index, columns, origin, development)
    link_ratio, age_to_age
        Displays age-to-age ratios for the triangle.
    valuation_date : date
        The latest valuation date of the data
    loc: Triangle
        pandas-style ``loc`` accessor
    iloc: Triangle
        pandas-style ``iloc`` accessor
    latest_diagonal: Triangle
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
    values: array
        4D numpy array underlying the Triangle instance
    T: Triangle
        Transpose index and columns of object.  Only available when Triangle is
        convertible to DataFrame.
    """

    def __init__(
        self,
        data=None,
        origin=None,
        development=None,
        columns=None,
        index=None,
        origin_format=None,
        development_format=None,
        cumulative=None,
        array_backend=None,
        pattern=False,
        trailing=True,
        *args,
        **kwargs
    ):
        if data is None:
            return
        index, columns, origin, development = self._input_validation(
            data, index, columns, origin, development
        )
        # Handle any ultimate vectors in triangles separately
        data, ult = self._split_ult(data, index, columns, origin, development)
        # Conform origins and developments to datetimes and determine lowest grains
        origin_date = self._to_datetime(data, origin, format=origin_format).rename(
            "__origin__"
        )
        self.origin_grain = self._get_grain(
            origin_date, trailing=trailing, kind="origin"
        )
        development_date = self._set_development(
            data, development, development_format, origin_date
        )
        self.development_grain = self._get_grain(
            development_date, trailing=trailing, kind="development"
        )
        origin_date = origin_date.dt.to_period(self.origin_grain).dt.to_timestamp(
            how="s"
        )

        development_date = development_date.dt.to_period(
            self.development_grain
        ).dt.to_timestamp(how="e")
        # Aggregate dates to the origin/development grains
        data_agg = self._aggregate_data(
            data, origin_date, development_date, index, columns
        )

        # Fill in missing periods with zeros
        date_axes = self._get_date_axes(
            data_agg["__origin__"],
            data_agg["__development__"],
            self.origin_grain,
            self.development_grain,
        )
        # Deal with labels
        if not index:
            index = ["Total"]
            data_agg[index[0]] = "Total"    
        self.kdims, key_idx = self._set_kdims(data_agg, index)
        self.vdims = np.array(columns)
        self.odims, orig_idx = self._set_odims(data_agg, date_axes)
        self.ddims, dev_idx = self._set_ddims(data_agg, date_axes)
        # Set remaining triangle properties
        val_date = data_agg["__development__"].max()
        val_date = val_date.compute() if hasattr(val_date, "compute") else val_date
        self.key_labels = index
        self.valuation_date = val_date
        if cumulative is None:
            warnings.warn(
                """
            The cumulative property of your triangle is not set. This may result in
            undesirable behavior. In a future release this will result in an error."""
            )
        self.is_cumulative = cumulative
        self.virtual_columns = VirtualColumns(self)
        self.is_pattern = pattern
        split = self.origin_grain.split("-")
        self.origin_grain = {"A": "Y", "2Q": "S"}.get(split[0], split[0])
        if len(split) == 1:
            self.origin_close = "DEC"
        else:
            self.origin_close = split[1]
        split = self.development_grain.split("-")
        self.development_grain = {"A": "Y", "2Q": "S"}.get(split[0], split[0])
        grain_sort = ["Y", "S", "Q", "M"]
        self.development_grain = grain_sort[
            max(
                grain_sort.index(self.origin_grain),
                grain_sort.index(self.development_grain),
            )
        ]
        # Coerce malformed triangles to something more predictible
        check_origin = (
            pd.period_range(
                start=self.odims.min(), end=self.valuation_date, 
                freq=self.origin_grain.replace('S', '2Q')
            )
            .to_timestamp()
            .values
        )
        if (
            len(check_origin) != self.odims
            and pd.to_datetime(options.ULT_VAL) != self.valuation_date
            and not self.is_pattern
        ):
            self.odims = check_origin

        # Set the Triangle values
        coords, amts = self._set_values(data_agg, key_idx, columns, orig_idx, dev_idx)
        self.values = num_to_nan(
            sp(
                coords,
                amts,
                prune=True,
                has_duplicates=False,
                sorted=True,
                shape=(
                    len(self.kdims),
                    len(self.vdims),
                    len(self.odims),
                    len(self.ddims),
                ),
            )
        )
        # Deal with array backend
        self.array_backend = "sparse"
        if array_backend is None:
            array_backend = options.ARRAY_BACKEND
        if not options.AUTO_SPARSE or array_backend == "cupy":
            self.set_backend(array_backend, inplace=True)
        else:
            self = self._auto_sparse()
        self._set_slicers()
        # Deal with special properties
        if self.is_pattern:
            obj = self.dropna()
            self.odims = obj.odims
            self.ddims = obj.ddims
            self.values = obj.values
        if ult:
            obj = concat((self.dev_to_val().iloc[..., : len(ult.odims), :], ult), -1)
            obj = obj.val_to_dev()
            self.odims = obj.odims
            self.ddims = obj.ddims
            self.values = obj.values
            self.valuation_date = pd.Timestamp(options.ULT_VAL)

    @staticmethod
    def _split_ult(data, index, columns, origin, development):
        """ Deal with triangles with ultimate values """
        ult = None
        if (
            development
            and len(development) == 1
            and data[development[0]].dtype == "<M8[ns]"
        ):
            u = data[data[development[0]] == options.ULT_VAL].copy()
            if len(u) > 0 and len(u) != len(data):
                ult = Triangle(
                    u,
                    origin=origin,
                    development=development,
                    columns=columns,
                    index=index,
                )
                ult.ddims = pd.DatetimeIndex([options.ULT_VAL])
                data = data[data[development[0]] != options.ULT_VAL]
        return data, ult

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
        if self.is_pattern and len(self.odims) == 1:
            return pd.Series(["(All)"])
        else:
            freq = {"Y": "A", "S": "2Q", "H": "2Q"}.get(
                self.origin_grain, self.origin_grain
            )
            freq = freq if freq == "M" else freq + "-" + self.origin_close
            return pd.DatetimeIndex(self.odims, name="origin").to_period(freq=freq)

    @origin.setter
    def origin(self, value):
        self._len_check(self.origin, value)
        freq = {"Y": "A", "S": "2Q"}.get(self.origin_grain, self.origin_grain)
        freq = freq if freq == "M" else freq + "-" + self.origin_close
        value = pd.PeriodIndex(list(value), freq=freq)
        self.odims = value.to_timestamp().values

    @property
    def development(self):
        ddims = self.ddims.copy()
        if self.is_val_tri:
            formats = {"Y": "%Y", "S": "%YQ%q", "Q": "%YQ%q", "M": "%Y-%m"}
            ddims = ddims.to_period(freq=self.development_grain).strftime(
                formats[self.development_grain]
            )
        elif self.is_pattern:
            offset = self._dstep()["M"][self.development_grain]
            if self.is_ultimate:
                ddims[-1] = ddims[-2] + offset
            if self.is_cumulative:
                ddims = ["{}-Ult".format(ddims[i]) for i in range(len(ddims))]
            else:
                ddims = [
                    "{}-{}".format(ddims[i], ddims[i] + offset)
                    for i in range(len(ddims))
                ]
        return pd.Series(list(ddims), name="development")

    @development.setter
    def development(self, value):
        self._len_check(self.development, value)
        self.ddims = np.array([value] if type(value) is str else value)

    def set_index(self, value, inplace=False):
        """ Sets the index of the Triangle """
        if inplace:
            self.index = value
            return self
        else:
            new_obj = self.copy()
            return new_obj.set_index(value=value, inplace=True)

    @property
    def is_val_tri(self):
        return type(self.ddims) == pd.DatetimeIndex

    @property
    def is_full(self):
        return self.nan_triangle.sum().sum() == np.prod(self.shape[-2:])

    @property
    def is_ultimate(self):
        return sum(self.valuation >= options.ULT_VAL[:4]) > 0

    @property
    def latest_diagonal(self):
        return self[self.valuation == self.valuation_date].sum("development")

    @property
    def link_ratio(self):
        if not self.is_pattern:
            obj = (1 / self.iloc[..., :-1]) * self.iloc[..., 1:].values
            if not obj.is_full:
                obj = obj[obj.valuation < obj.valuation_date]
            if hasattr(obj, "w_"):
                w_ = obj.w_[..., 0:1, : len(obj.odims), :]
                obj = obj * w_ if obj.shape == w_.shape else obj
            obj.is_pattern = True
            obj.is_cumulative = False
            obj.values = num_to_nan(obj.values)
            return obj
        else:
            return self

    @property
    def age_to_age(self):
        return self.link_ratio

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
            xp = self.get_array_module()
            if not self.is_cumulative:
                if self.is_pattern:
                    values = xp.nan_to_num(self.values[..., ::-1])
                    values = num_to_value(values, 1)
                    values = xp.cumprod(values, -1)[..., ::-1]
                    self.values = values * self.nan_triangle
                    values = num_to_value(values, self.get_array_module(values).nan)
                else:
                    if self.array_backend not in ["sparse", "dask"]:
                        self.values = (
                            xp.cumsum(xp.nan_to_num(self.values), 3)
                            * self.nan_triangle[None, None, ...]
                        )
                    else:
                        values = xp.nan_to_num(self.values)
                        nan_triangle = xp.nan_to_num(self.nan_triangle)
                        l1 = lambda i: values[..., 0 : i + 1]
                        l2 = lambda i: l1(i) * nan_triangle[..., i : i + 1]
                        l3 = lambda i: l2(i).sum(3, keepdims=True)
                        if db:
                            bag = db.from_sequence(range(self.shape[-1]))
                            bag = bag.map(l3)
                            out = bag.compute(scheduler="threads")
                        else:
                            out = [l3(i) for i in range(self.shape[-1])]
                        self.values = xp.concatenate(out, axis=3)
                    self.values = num_to_nan(self.values)
                self.is_cumulative = True
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
        if inplace:
            v = self.valuation_date
            if self.is_cumulative or self.is_cumulative is None:
                if self.is_pattern:
                    xp = self.get_array_module()
                    self.values = xp.nan_to_num(self.values)
                    values = num_to_value(self.values, 1)
                    diff = self.iloc[..., :-1] / self.iloc[..., 1:].values
                    self = concat((diff, self.iloc[..., -1],), axis=3)
                    self.values = self.values * self.nan_triangle
                else:
                    diff = self.iloc[..., 1:] - self.iloc[..., :-1].values
                    self = concat((self.iloc[..., 0], diff), axis=3)
                self.is_cumulative = False
            self.valuation_date = v
            return self
        else:
            new_obj = self.copy()
            return new_obj.cum_to_incr(inplace=True)

    def _dstep(self):
        return {
            "M": {"Y": 12, "S": 6, "Q": 3, "M": 1},
            "Q": {"Y": 4, "S": 2, "Q": 1},
            "S": {"Y": 2, "S": 1},
            "Y": {"Y": 1},
        }

    def _val_dev(self, sign, inplace=False):
        backend = self.array_backend
        obj = self.set_backend("sparse")
        if not inplace:
            obj.values = obj.values.copy()
        scale = self._dstep()[obj.development_grain][obj.origin_grain]
        offset = np.arange(obj.shape[-2]) * scale
        min_slide = -offset.max()
        if (obj.values.coords[-2] == np.arange(1)).all():
            # Unique edge case #239
            offset = offset[-1:] * sign
        offset = offset[obj.values.coords[-2]] * sign  # [0]
        obj.values.coords[-1] = obj.values.coords[-1] + offset
        ddims = obj.valuation[obj.valuation <= obj.valuation_date]
        ddims = len(ddims.drop_duplicates())
        if ddims == 1 and sign == -1:
            ddims = len(obj.odims)
        if obj.values.density > 0 and obj.values.coords[-1].min() < 0:
            obj.values.coords[-1] = obj.values.coords[-1] - min(
                obj.values.coords[-1].min(), min_slide
            )
            ddims = np.max([np.max(obj.values.coords[-1]) + 1, ddims])
        obj.values.shape = tuple(list(obj.shape[:-1]) + [ddims])
        if options.AUTO_SPARSE == False or backend == "cupy":
            obj = obj.set_backend(backend)
        else:
            obj = obj._auto_sparse()
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
        Triangle
            Updated instance of the triangle with valuation periods.

        """
        if self.is_val_tri:
            if inplace:
                return self
            else:
                return self.copy()
        is_cumulative = self.is_cumulative
        if self.is_full:
            if is_cumulative:
                obj = self.cum_to_incr(inplace=inplace)
            else:
                obj = self.copy()
            if self.is_ultimate:
                ultimate = obj.iloc[..., -1:]
                obj = obj.iloc[..., :-1]
        else:
            obj = self
        obj = obj._val_dev(1, inplace)
        ddims = obj.valuation[obj.valuation <= obj.valuation_date]
        obj.ddims = ddims.drop_duplicates().sort_values()
        if self.is_full:
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
        if not self.is_val_tri:
            if inplace:
                return self
            else:
                return self.copy()
        if self.is_ultimate and self.shape[-1] > 1:
            ultimate = self.iloc[..., -1:]
            ultimate.ddims = np.array([9999])
            obj = self.iloc[..., :-1]._val_dev(-1, inplace)
        else:
            obj = self.copy()._val_dev(-1, inplace)
        val_0 = obj.valuation[0]
        if self.ddims.shape[-1] == 1 and self.ddims[0] == self.valuation_date:
            origin_0 = pd.to_datetime(obj.odims[-1])
        else:
            origin_0 = pd.to_datetime(obj.odims[0])
        lag_0 = (val_0.year - origin_0.year) * 12 + val_0.month - origin_0.month + 1
        scale = self._dstep()["M"][obj.development_grain]
        obj.ddims = np.arange(obj.values.shape[-1]) * scale + lag_0
        prune = obj[obj.origin == obj.origin.max()]
        if self.is_ultimate and self.shape[-1] > 1:
            obj = obj.iloc[..., : (prune.valuation <= prune.valuation_date).sum()]
            obj = concat((obj, ultimate), -1)
        return obj

    def grain(self, grain="", trailing=False, inplace=False):
        """Changes the grain of a cumulative triangle.

        Parameters
        ----------
        grain : str
            The grain to which you want your triangle converted, specified as
            'OXDY' where X and Y can take on values of ``['Y', 'S', 'Q', 'M'
            ]`` For example, 'OYDY' for Origin Year/Development Year, 'OQDM'
            for Origin quarter/Development Month, etc.
        trailing : bool
            For partial origin years/quarters, trailing will set the year/quarter
            end to that of the latest available from the origin data.
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Triangle
        """
        ograin_old, ograin_new = self.origin_grain, grain[1:2]
        dgrain_old, dgrain_new = self.development_grain, grain[-1]
        ograin_new = "S" if ograin_new == "H" else ograin_new
        valid = {
            "Y": ["Y"],
            "Q": ["Q", "S", "Y"],
            "M": ["Y", "S", "Q", "M"],
            "S": ["S", "Y"],
        }
        if ograin_new not in valid.get(ograin_old, []) or dgrain_new not in valid.get(
            dgrain_old, []
        ):
            raise ValueError("New grain not compatible with existing grain")
        if (
            self.is_cumulative is None
            and dgrain_old != dgrain_new
            and self.shape[-1] > 1
        ):
            raise AttributeError(
                "The is_cumulative attribute must be set before using grain method."
            )
        if valid["M"].index(ograin_new) > valid["M"].index(dgrain_new):
            raise ValueError("Origin grain must be coarser than development grain")
        if self.is_full and not self.is_ultimate and not self.is_val_tri:
            warnings.warn("Triangle includes extraneous development lags")
        obj = self.dev_to_val()
        if ograin_new != ograin_old:
            freq = {"Y": "A", "S": "2Q"}.get(ograin_new, ograin_new)
            mn = self.origin[-1].strftime("%b").upper() if trailing else "DEC"
            indices = (
                pd.Series(range(len(self.origin)), index=self.origin)
                .resample("-".join([freq, mn]))
                .indices
            )
            groups = pd.concat(
                [pd.Series([k] * len(v), index=v) for k, v in indices.items()], axis=0
            ).values
            obj = obj.groupby(groups, axis=2).sum()
            obj.origin_close = mn
            if len(obj.ddims) > 1 and pd.Timestamp(obj.odims[0]).strftime(
                "%Y%m"
            ) != obj.valuation[0].strftime("%Y%m"):
                addl_ts = (
                    pd.period_range(obj.odims[0], obj.valuation[0], freq="M")[:-1]
                    .to_timestamp()
                    .values
                )
                addl = obj.iloc[..., -len(addl_ts) :] * 0
                addl.ddims = addl_ts
                obj = concat((addl, obj), axis=-1)
                obj.values = num_to_nan(obj.values)
        if dgrain_old != dgrain_new and obj.shape[-1] > 1:
            step = self._dstep()[dgrain_old][dgrain_new]
            d = np.sort(
                len(obj.development) - np.arange(0, len(obj.development), step) - 1
            )
            if obj.is_cumulative:
                obj = obj.iloc[..., d]
            else:
                ddims = obj.ddims[d]
                d2 = [d[0]] * (d[0] + 1) + list(np.repeat(np.array(d[1:]), step))
                obj = obj.groupby(d2, axis=3).sum()
                obj.ddims = ddims
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
            The start date from which trend should be calculated. If none is
            provided then the latest date of the triangle is used.
        end: date
            The end date to which the trend should be calculated. If none is
            provided then the earliest period of the triangle is used.
        ultimate_lag : int
            If ultimate valuations are in the triangle, optionally set the overall
            age (in months) of the ultimate to be some lag from the latest non-Ultimate
            development

        Returns
        -------
        Triangle
            updated with multiplicative trend applied.
        """
        if axis not in ["origin", "valuation", 2, -2]:
            raise ValueError(
                "Only origin and valuation axes are supported for trending"
            )
        xp = self.get_array_module()
        start = pd.to_datetime(start) if type(start) is str else start
        start = self.valuation_date if start is None else start
        end = pd.to_datetime(end) if type(end) is str else end
        end = self.origin[0].to_timestamp() if end is None else end
        if axis in ["origin", 2, -2]:
            vector = pd.DatetimeIndex(
                np.tile(
                    self.origin.to_timestamp(how="e").values, self.shape[-1]
                ).flatten()
            )
        else:
            vector = self.valuation
        lower, upper = (end, start) if end > start else (start, end)
        vector = pd.DatetimeIndex(
            np.maximum(
                np.minimum(np.datetime64(lower), vector.values), np.datetime64(upper)
            )
        )
        vector = (
            (start.year - vector.year) * 12 + (start.month - vector.month)
        ).values.reshape(self.shape[-2:], order="f")
        if self.is_ultimate and ultimate_lag is not None and vector.shape[-1] > 1:
            vector[:, -1] = vector[:, -2] + ultimate_lag
        trend = (
            xp.array((1 + trend) ** (vector / 12))[None, None, ...] * self.nan_triangle
        )
        obj = self.copy()
        obj.values = obj.values * trend
        return obj

    def broadcast_axis(self, axis, value):
        warnings.warn(
            """
            Broadcast axis is deprecated in favor of broadcasting
            using Triangle arithmetic."""
        )
        return self

    def copy(self):
        X = Triangle()
        X.__dict__.update(vars(self))
        X._set_slicers()
        X.values = X.values.copy()
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

    def shift(self, periods=-1, axis=3):
        """ Shift elements along an axis by desired number of periods.

        Data that falls beyond the existing shape of the Triangle is eliminated
        and new cells default to zero.

        Parameters
        ----------
        periods : int
            Number of periods to shift. Can be positive or negative.
        axis : {2 or 'origin', 3 or 'development', None}, default 3
            Shift direction.

        Returns
        -------
        Triangle
            updated with shifted elements

        """
        axis = self._get_axis(axis)
        if axis < 2:
            raise AttributeError(
                "Lagging only supported for origin and development axes"
            )
        if periods == 0:
            return self
        if periods > 0:
            if axis == 3:
                out = concat(
                    (
                        self.iloc[..., 1:].rename("development", self.development[:-1]),
                        (self.iloc[..., -1:] * 0),
                    ),
                    axis=axis,
                )
            else:
                out = concat(
                    (
                        self.iloc[..., 1:, :].rename("origin", self.origin[:-1]),
                        (self.iloc[..., -1:, :] * 0),
                    ),
                    axis=axis,
                )
        else:
            if axis == 3:
                out = concat(
                    (
                        (self.iloc[..., :1] * 0),
                        self.iloc[..., :-1].rename("development", self.development[1:]),
                    ),
                    axis=axis,
                )
            else:
                out = concat(
                    (
                        (self.iloc[..., :1, :] * 0),
                        self.iloc[..., :-1, :].rename("origin", self.origin[1:]),
                    ),
                    axis=axis,
                )
        if abs(periods) == 1:
            return out
        else:
            return out.shift(periods - 1 if periods > 0 else periods + 1, axis)

    def sort_axis(self, axis):
        """ Method to sort a Triangle along a given axis

        Parameters
        ----------
        axis : in or str
            The axis for sorting

        Returns
        -------
        Triangle
            updated with shifted elements
        """

        axis = self._get_axis(axis)
        if axis == 0:
            return self.sort_index()
        obj = self.copy()
        if axis == 1:
            sort = pd.Series(self.vdims).sort_values().index
            if np.any(sort != pd.Series(self.vdims).index):
                obj.values = obj.values[:, list(sort), ...]
                obj.vdims = obj.vdims[list(sort)]
        if axis == 2:
            sort = pd.Series(self.odims).sort_values().index
            if np.any(sort != pd.Series(self.odims).index):
                obj.values = obj.values[..., list(sort), :]
                obj.odims = obj.odims[list(sort)]
        if axis == 3:
            sort = self.development.sort_values().index
            if np.any(sort != self.development.index):
                obj.values = obj.values[..., list(sort)]
                obj.ddims = obj.ddims[list(sort)]
        return obj
