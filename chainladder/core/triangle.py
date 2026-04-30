# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
from packaging import version
from chainladder.core.base import TriangleBase
from chainladder.utils.sparse import sp
from chainladder.core.slice import VirtualColumns
from chainladder.core.correlation import DevelopmentCorrelation, ValuationCorrelation
from chainladder.utils.utility_functions import concat, num_to_nan, num_to_value, to_period
from chainladder import options

try:
    import dask.bag as db
except ImportError:
    db = None

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame, Series
    from numpy.typing import ArrayLike
    from pandas._libs.tslibs.timestamps import Timestamp  # noqa
    from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
    from sparse import COO


class Triangle(TriangleBase):
    """
    The core data structure of the chainladder package

    Parameters
    ----------
    data: DataFrame or DataFrameXchg, or dict
        A single dataframe that contains columns representing all other
        arguments to the Triangle constructor. If using pandas version > 1.5.2,
        one may supply a DataFrame-like object (referred to as DataFrameXchg)
        supporting the __dataframe__ protocol, which will then be converted to
        a pandas DataFrame. If supplying a dict, it must be structured such that
        a pandas DataFrame created from it will be accepted by the constructor.
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
        Controls how the period-end month is inferred from origin and
        development dates. When False, December is treated as the period end
        (i.e., calendar fiscal periods). When True, the period end is inferred
        from the data itself. This is useful when origin dates do not align
        with calendar period boundaries.

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

    Examples
    --------

    Constructing a Triangle from a Pandas DataFrame.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        import pandas as pd
        df = pd.DataFrame(
            data={
                'origin': [1981, 1981, 1981, 1981, 1982, 1982, 1982, 1983, 1983, 1984],
                'development': [1981, 1982, 1983, 1984, 1982, 1983, 1984, 1983, 1984, 1984],
                'reported': [5012, 8269, 10907, 11805, 106, 4285, 5396, 3410, 8992, 5655],
            }
        )
        tr = cl.Triangle(
            data=df,
            origin='origin',
            development='development',
            columns=['reported'],
            cumulative=True,
        )
        print(tr)

    .. testoutput::

                  12      24       36       48
        1981  5012.0  8269.0  10907.0  11805.0
        1982   106.0  4285.0   5396.0      NaN
        1983  3410.0  8992.0      NaN      NaN
        1984  5655.0     NaN      NaN      NaN

    When another dimension is added, such as an additional column, the Triangle
    becomes multidimensional. In this case, printing displays the Triangle's
    metadata rather than its contents.

    .. testcode::

        df = pd.DataFrame(
            data={
                'origin': [1981, 1981, 1981, 1981, 1982, 1982, 1982, 1983, 1983, 1984],
                'development': [1981, 1982, 1983, 1984, 1982, 1983, 1984, 1983, 1984, 1984],
                'reported': [5012, 8269, 10907, 11805, 106, 4285, 5396, 3410, 8992, 5655],
                'paid': [2506, 4135, 5454, 5903, 53, 2143, 2698, 1705, 4496, 2828],
            }
        )
        tr = cl.Triangle(
            data=df,
            origin='origin',
            development='development',
            columns=['reported', 'paid'],
            cumulative=True,
        )
        print(tr)

    .. testoutput::

                    Triangle Summary
        Valuation:           1984-12
        Grain:                  OYDY
        Shape:          (1, 2, 4, 4)
        Index:               [Total]
        Columns:    [reported, paid]

    Using the ``index`` parameter creates a multi-dimensional Triangle split by a
    categorical grouping, for example Line of Business.

.. testcode::

        df = pd.DataFrame(
            data={
                 'lob': ['auto', 'auto', 'auto', 'home', 'home', 'home'],
                'origin': [2020, 2020, 2021, 2020, 2020, 2021],
                'development': [2020, 2021, 2021, 2020, 2021, 2021],
                'reported': [100, 150, 80, 200, 280, 160],
            }
        )
        tr = cl.Triangle(
            data=df,
            origin='origin',
            development='development',
            columns=['reported'],
            index=['lob'],
            cumulative=True,
        )
        print(tr)

    .. testoutput::

                   Triangle Summary
        Valuation:          2021-12
        Grain:                 OYDY
        Shape:         (2, 1, 2, 2)
        Index:                [lob]
        Columns:         [reported]

    Non-standard date strings can be parsed by specifying ``origin_format`` and
    ``development_format`` using Python ``strftime`` codes.

    .. testcode::

        df = pd.DataFrame(
            data={
                'origin': ['2020-01', '2020-01', '2020-02', '2020-02'],
                'development': ['2020-01', '2020-02', '2020-02', '2020-03'],
                'reported': [100, 150, 200, 280],
            }
        )
        tr = cl.Triangle(
            data=df,
            origin='origin',
            origin_format='%Y-%m',
            development='development',
            development_format='%Y-%m',
            columns=['reported'],
            cumulative=True,
        )
        print(tr)

    .. testoutput::

                     1      2   3
        2020-01  100.0  150.0 NaN
        2020-02  200.0  280.0 NaN
        2020-03    NaN    NaN NaN

    Setting ``cumulative=False`` builds an incremental Triangle, where each cell
    is the amount accrued within that development period rather than the
    cumulative total to date.

    .. testcode::

        df = pd.DataFrame(
            data={
                'origin': [1981, 1981, 1981, 1981, 1982, 1982, 1982, 1983, 1983, 1984],
                'development': [1981, 1982, 1983, 1984, 1982, 1983, 1984, 1983, 1984, 1984],
                'reported': [5012, 3257, 2638, 898, 106, 4179, 1111, 3410, 5582, 5655],
            }
        )
        tr = cl.Triangle(
            data=df,
            origin='origin',
            development='development',
            columns=['reported'],
            cumulative=False,
        )
        print(tr)

    .. testoutput::

                  12      24      36     48
        1981  5012.0  3257.0  2638.0  898.0
        1982   106.0  4179.0  1111.0    NaN
        1983  3410.0  5582.0     NaN    NaN
        1984  5655.0     NaN     NaN    NaN

    By default (``trailing=False``), chainladder uses December as the fiscal
    period end, so origin dates are assigned to calendar quarters. Setting
    ``trailing=True`` instead infers the period end from the data itself,
    producing quarters aligned to the origin dates.

    .. testcode::

        df = pd.DataFrame(
            data={
                'origin': ['2023-05', '2023-08', '2023-11', '2024-02'],
                'development': ['2024-04', '2024-04', '2024-04', '2024-04'],
                'premium': [100, 130, 160, 140],
            }
        )
        tr = cl.Triangle(
            data=df,
            origin='origin',
            origin_format='%Y-%m',
            development='development',
            development_format='%Y-%m',
            columns=['premium'],
            cumulative=True,
            trailing=False,
        )
        print(tr)

    .. testoutput::

                2024-04
        2023Q2    100.0
        2023Q3    130.0
        2023Q4    160.0
        2024Q1    140.0
        2024Q2      NaN

    .. testcode::

        tr = cl.Triangle(
            data=df,
            origin='origin',
            origin_format='%Y-%m',
            development='development',
            development_format='%Y-%m',
            columns=['premium'],
            cumulative=True,
            trailing=True,
        )
        print(tr)

    .. testoutput::

                2024Q2
        2024Q1   100.0
        2024Q2   130.0
        2024Q3   160.0
        2024Q4   140.0
    """

    def __init__(
        self,
        data: Optional[DataFrame | DataFrameXchg | dict] = None,
        origin: Optional[str | list] = None,
        development: Optional[str | list] = None,
        columns: Optional[str | list] = None,
        index: Optional[str | list] = None,
        origin_format: Optional[str] = None,
        development_format: Optional[str] = None,
        cumulative: Optional[bool] = None,
        array_backend: str = None,
        pattern=False,
        trailing: bool = True,
        *args,
        **kwargs,
    ):

        # If data are present, validate the dimensions.
        if data is None:
            return
        elif type(data) == dict:
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame) and hasattr(data, "__dataframe__"):
            data = self._interchange_dataframe(data)
        index, columns, origin, development = self._input_validation(
            data=data,
            index=index,
            columns=columns,
            origin=origin,
            development=development,
        )

        # Store dimension metadata.
        self.origin_label: list = origin
        
        # Handle any ultimate vectors in triangles separately.
        data, ult = self._split_ult(
            data=data,
            index=index,
            columns=columns,
            origin=origin,
            development=development,
        )
        # Conform origins and developments to datetimes and determine the lowest grains.
        origin_date: Series = self._to_datetime(
            data=data, fields=origin, date_format=origin_format
        ).rename("__origin__")

        self.origin_grain: str = self._get_grain(
            dates=origin_date, trailing=trailing, kind="origin"
        )

        development_date = self._set_development(
            data=data,
            development=development,
            development_format=development_format,
            origin_date=origin_date,
        )

        if len(development_date.unique()) == 1:
            if len(data) == 1 and self.origin_grain.split("-")[0] in ["Y", "A"]:
                self.development_grain = self.origin_grain
            else:
                dev_date = pd.to_datetime(development_date.iloc[0])
                dev_date_monthly_end = dev_date.to_period("M").to_timestamp(how="e")
                period_converted = dev_date_monthly_end.to_period(self.origin_grain).to_timestamp(how="e")
                if abs((period_converted - dev_date_monthly_end).total_seconds()) < 1e-6:
                    self.development_grain = self.origin_grain
                else:
                    self.development_grain = "M"
        else:
            self.development_grain = self._get_grain(
                dates=development_date, trailing=trailing, kind="development"
            )

        # Ensure that origin_date values represent the beginning of the period.
        # i.e., 1990 means the start of 1990.
        origin_date: Series = to_period(origin_date,self.origin_grain).dt.to_timestamp(how="s")
        
        # Ensure that development_date values represent the end of the period.
        # i.e., 1990 means the end of 1990 assuming annual development periods.
        development_date: Series = to_period(development_date,self.development_grain).dt.to_timestamp(how="e")
        
        # Aggregate dates to the origin/development grains.
        data_agg: DataFrame = self._aggregate_data(
            data=data,
            origin_date=origin_date,
            development_date=development_date,
            index=index,
            columns=columns,
        )
        
        # Fill in missing periods with zeros.
        date_axes: DataFrame = self._get_date_axes(
            data_agg["__origin__"],
            data_agg["__development__"],
            self.origin_grain,
            self.development_grain,
        )

        # Deal with labels.
        if not index:
            index: list = ["Total"]
            self.index_label: list = index
            data_agg[index[0]] = "Total"

        self.kdims: np.ndarray
        key_idx: np.ndarray
        self.vdims: np.ndarray
        self.odims: np.ndarray
        orig_idx: np.ndarray
        self.ddims: ArrayLike
        dev_idx: np.ndarray

        self.kdims, key_idx = self._set_kdims(data_agg, index)
        self.vdims = np.array(columns)
        self.odims, orig_idx = self._set_odims(data_agg, date_axes)
        self.ddims, dev_idx = self._set_ddims(data_agg, date_axes)

        # Set remaining triangle properties.
        val_date: Timestamp = data_agg["__development__"].max()
        val_date = val_date.compute() if hasattr(val_date, "compute") else val_date
        self.key_labels: list = index
        self.valuation_date: Timestamp = val_date

        if cumulative is None:
            warnings.warn(
                """
                The cumulative property of your triangle is not set. This may result in
                undesirable behavior. In a future release this will result in an error.
                """
            )

        self.is_cumulative: bool = cumulative
        self.virtual_columns = VirtualColumns(self)
        self._pattern: bool = pattern
        
        split: list[str] = self.origin_grain.split("-")
        self.origin_grain: str = {"A": "Y", "2Q": "S"}.get(split[0], split[0])

        if len(split) == 1:
            self.origin_close: str = "DEC"
        else:
            self.origin_close: str = split[1]

        split: list[str] = self.development_grain.split("-")
        self.development_grain: str = {"A": "Y", "2Q": "S"}.get(split[0], split[0])
        grain_sort: list = ["Y", "S", "Q", "M"]
        self.development_grain: str = grain_sort[
            max(
                grain_sort.index(self.origin_grain),
                grain_sort.index(self.development_grain),
            )
        ]

        # Coerce malformed triangles to something more predictable.
        check_origin: np.ndarray = (
            pd.period_range(
                start=self.odims.min(),
                end=self.valuation_date,
                freq=self.origin_grain.replace("S", "2Q") + ('' if self.origin_grain == "M" else '-' + self.origin_close),
            )
            .to_timestamp()
            .values
        )

        if (
            len(check_origin) != len(self.odims)
            and pd.to_datetime(options.ULT_VAL) != self.valuation_date
            and not self.is_pattern
        ):
            self.odims: np.ndarray = check_origin

        # Set the Triangle values.
        coords: np.ndarray
        amts: np.ndarray

        coords, amts = self._set_values(
            data_agg=data_agg,
            key_idx=key_idx,
            columns=columns,
            orig_idx=orig_idx,
            dev_idx=dev_idx,
        )

        # Construct Sparse multidimensional array.
        self.values: COO = num_to_nan(
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
        # Deal with array backend.
        self.array_backend = "sparse"
        if array_backend is None:
            array_backend: str = options.ARRAY_BACKEND
        if not options.AUTO_SPARSE or array_backend == "cupy":
            self.set_backend(
                backend=array_backend,
                inplace=True
            )
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
    def _split_ult(
        data: DataFrame,
        index: list,
        columns: list,
        origin: list,
        development: list
    ) -> tuple[DataFrame, Triangle]:
        """Deal with triangles with ultimate values."""
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
    def index(self) -> DataFrame:
        """
        Returns a DataFrame of the unique values of the index.
        """
        return pd.DataFrame(list(self.kdims), columns=self.key_labels)

    @index.setter
    def index(self, value) -> None:
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
    def columns_label(self) -> list:
        return self.columns.to_list()

    @property
    def origin(self):
        """
        Origin periods of the Triangle as a ``PeriodIndex``.

        The frequency of the index reflects ``origin_grain`` (e.g. annual,
        quarterly, monthly). When the Triangle holds aggregated patterns with a
        single origin row, the property returns ``Series(['(All)'])`` instead.

        Returns
        -------
        PeriodIndex or Series
            One entry per origin period of the Triangle.

        Examples
        --------
        Annual-origin Triangle.

        >>> tr = cl.load_sample('ukmotor')
        >>> tr.origin.year.tolist()
        [2007, 2008, 2009, 2010, 2011, 2012, 2013]

        The number of origin periods matches ``Triangle.shape[-2]``.

        >>> len(tr.origin) == tr.shape[-2]
        True
        """
        if self.is_pattern and len(self.odims) == 1:
            return pd.Series(["(All)"])
        else:
            freq = {
                "Y": (
                    "Y"
                    if version.Version(pd.__version__) >= version.Version("2.2.0")
                    else "A"
                ),
                "S": "2Q",
                "H": "2Q",
            }.get(self.origin_grain, self.origin_grain)
            freq = freq if freq == "M" else freq + "-" + self.origin_close
            return pd.DatetimeIndex(self.odims, name="origin").to_period(freq=freq)

    @origin.setter
    def origin(self, value):
        self._len_check(self.origin, value)
        freq = {
            "Y": "A" if float(".".join(pd.__version__.split(".")[:-1])) < 2.2 else "Y",
            "S": "2Q",
        }.get(self.origin_grain, self.origin_grain)
        freq = freq if freq == "M" else freq + "-" + self.origin_close
        value = pd.PeriodIndex(list(value), freq=freq)
        self.odims = value.to_timestamp().values

    @property
    def development(self):
        """
        Development periods of the Triangle as a Series.

        For a development-lag Triangle (``is_val_tri=False``), values are
        integer lags expressed in months from the start of the origin period.
        For a valuation Triangle (``is_val_tri=True``), values are calendar
        valuation labels formatted at ``development_grain``. For pattern
        Triangles, labels carry their from/to development range (e.g.
        ``"12-24"`` or ``"108-Ult"``).

        Returns
        -------
        Series
            One entry per development column of the Triangle.

        Examples
        --------
        Annual-grain development on a loss Triangle is reported as month lags.

        >>> tr = cl.load_sample('ukmotor')
        >>> tr.development.tolist()
        [12, 24, 36, 48, 60, 72, 84]

        On a valuation Triangle the labels become calendar periods.

        >>> tr.dev_to_val().development.tolist()
        ['2007', '2008', '2009', '2010', '2011', '2012', '2013']

        On a link-ratio (pattern) Triangle the labels span the from/to lags.

        >>> tr.link_ratio.development.tolist()
        ['12-24', '24-36', '36-48', '48-60', '60-72', '72-84']
        """
        ddims = self.ddims.copy()
        if self.is_val_tri:
            formats = {"Y": "%Y", "S": "%YQ%q", "Q": "%YQ%q", "M": "%Y-%m"}
            ddims = ddims.to_period(freq=self.development_grain.replace("S", "2Q")).strftime(
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
        """Sets the index of the Triangle"""
        if inplace:
            self.index = value
            return self
        else:
            new_obj = self.copy()
            return new_obj.set_index(value=value, inplace=True)

    @property
    def is_val_tri(self):
        """
        Indicates whether the development axis is expressed in valuation
        periods rather than development lags.

        Returns
        -------
        bool
            ``True`` if the development axis is a ``DatetimeIndex`` of
            valuation dates (as produced by :meth:`dev_to_val`); ``False`` if
            development is expressed as integer lags in months.

        Examples
        --------
        A development-lag Triangle has integer lags on the development axis.

        >>> tr = cl.load_sample('ukmotor')
        >>> tr.is_val_tri
        False

        Calling ``dev_to_val`` reshapes the development axis into calendar
        valuation periods.

        >>> tr.dev_to_val().is_val_tri
        True
        """
        return type(self.ddims) == pd.DatetimeIndex

    @property
    def is_full(self) -> bool:
        """
        Property that in indicates whether lower half of Triangle has been filled in.

        Returns
        -------

        bool

        Examples
        --------
        A loaded sample loss Triangle is upper-triangular: future cells below
        the latest diagonal are NaN, so ``is_full`` is ``False``.

        >>> tr = cl.load_sample('ukmotor')
        >>> bool(tr.is_full)
        False

        A ``cdf_`` Triangle from a fitted development model has every cell
        populated, so it is full.

        >>> cdf = cl.Development().fit(tr).cdf_
        >>> bool(cdf.is_full)
        True
        """

        return self.nan_triangle.sum().sum() == np.prod(self.shape[-2:])

        
    @property
    def is_pattern(self) -> bool:
        """
        Indicates whether the Triangle holds development patterns rather than
        observed values.

        Pattern Triangles are produced by methods such as :attr:`link_ratio`,
        :attr:`age_to_age`, and the ``ldf_`` / ``cdf_`` attributes of fitted
        development estimators. They typically carry ``is_cumulative=False`` and
        their cells are unitless ratios.

        Returns
        -------
        bool
            ``True`` if the Triangle is a set of development patterns,
            otherwise ``False``.

        Examples
        --------
        A loss Triangle is not a pattern.

        >>> tr = cl.load_sample('ukmotor')
        >>> tr.is_pattern
        False

        Calling ``link_ratio`` returns a Triangle of age-to-age factors, which
        is flagged as a pattern.

        >>> tr.link_ratio.is_pattern
        True
        """
        return self._pattern

    @is_pattern.setter
    def is_pattern(self, pattern: bool):
        self._pattern = pattern

    @property
    def is_ultimate(self) ->  np.bool:
        """
        Indicates whether the Triangle includes an ultimate valuation column.

        ``True`` when at least one cell carries the sentinel ultimate
        valuation date (``options.ULT_VAL``), as produced by reserving methods
        such as :class:`~chainladder.Chainladder` or any model returning an
        ``ultimate_`` Triangle.

        Returns
        -------
        bool
            ``True`` if the Triangle has an ultimate column, otherwise ``False``.

        Examples
        --------
        A loaded sample triangle has no ultimate column.

        >>> tr = cl.load_sample('ukmotor')
        >>> bool(tr.is_ultimate)
        False

        Fitting a chainladder model produces an ``ultimate_`` Triangle whose
        single development column is the ultimate valuation.

        >>> ult = cl.Chainladder().fit(tr).ultimate_
        >>> bool(ult.is_ultimate)
        True
        """
        return sum(self.valuation >= options.ULT_VAL[:4]) > 0

    @property
    def latest_diagonal(self) -> Triangle:
        """
        The latest diagonal of the Triangle, collapsed to a single development
        column.

        For each origin period, the cell on the most recent valuation diagonal
        is selected. The result is a Triangle with one development column
        labeled by the latest valuation date.

        Returns
        -------
        Triangle
            Single-development-column Triangle of the most recent value for
            each origin period.

        Examples
        --------
        >>> tr = cl.load_sample('ukmotor')
        >>> tr.latest_diagonal
                 2013
        2007  12690.0
        2008  12746.0
        2009  12993.0
        2010  11093.0
        2011  10217.0
        2012   9650.0
        2013   6283.0
        """
        return self[self.valuation == self.valuation_date].sum(axis="development")

    @property
    def link_ratio(self) -> Triangle:
        """
        Displays age-to-age ratios for the triangle. If the calling Triangle object already has the
        self.is_pattern set to true (i.e., it is already a set of link ratios or development patterns),
        this property simply returns itself.

        Returns
        -------

        Triangle object in link ratio form.

        Examples
        --------
        >>> tr = cl.load_sample('ukmotor')
        >>> tr.link_ratio
                 12-24     24-36     36-48     48-60     60-72    72-84
        2007  1.915694  1.336902  1.190391  1.098935  1.049902  1.02753
        2008  1.925269  1.295729  1.118225  1.085655  1.051911      NaN
        2009  1.902870  1.234826  1.148734  1.105317       NaN      NaN
        2010  1.804424  1.261032  1.135066       NaN       NaN      NaN
        2011  1.902892  1.293782       NaN       NaN       NaN      NaN
        2012  1.891415       NaN       NaN       NaN       NaN      NaN

        Each cell is the ratio of the cumulative value at the next development
        period to the value at the current period. ``link_ratio`` carries
        ``is_pattern=True`` and ``is_cumulative=False`` on the returned Triangle.
        """

        # Case where triangle is not a set of link ratios or development patterns.
        if not self.is_pattern:
            obj: Triangle = (1 / self.iloc[..., :-1]) * self.iloc[..., 1:].values
            if not obj.is_full:
                obj = obj[obj.valuation < obj.valuation_date]
            if hasattr(obj, "w_"):
                w_ = obj.w_[..., : len(obj.odims), :]
                obj = obj * w_ if obj.shape == w_.shape else obj
            obj.is_pattern = True
            obj.is_cumulative = False
            obj.values = num_to_nan(obj.values)
            return obj
        # Case where triangle already is a set of link ratios or development patterns.
        else:
            return self

    @property
    def age_to_age(self):
        """
        Alias for :attr:`link_ratio`. Returns the same Triangle of age-to-age
        development factors.

        Returns
        -------
        Triangle
            Triangle of age-to-age ratios.

        Examples
        --------
        >>> tr = cl.load_sample('ukmotor')
        >>> tr.age_to_age
                 12-24     24-36     36-48     48-60     60-72    72-84
        2007  1.915694  1.336902  1.190391  1.098935  1.049902  1.02753
        2008  1.925269  1.295729  1.118225  1.085655  1.051911      NaN
        2009  1.902870  1.234826  1.148734  1.105317       NaN      NaN
        2010  1.804424  1.261032  1.135066       NaN       NaN      NaN
        2011  1.902892  1.293782       NaN       NaN       NaN      NaN
        2012  1.891415       NaN       NaN       NaN       NaN      NaN
        """
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

        Examples
        --------
        Construct an incremental triangle and accumulate it along the development axis.

        >>> df = pd.DataFrame(
        ...     data={
        ...         'origin': [1981, 1981, 1981, 1981, 1982, 1982, 1982, 1983, 1983, 1984],
        ...         'development': [1981, 1982, 1983, 1984, 1982, 1983, 1984, 1983, 1984, 1984],
        ...         'reported': [5012, 3257, 2638, 898, 106, 4179, 1111, 3410, 5582, 5655],
        ...     }
        ... )
        >>> tr = cl.Triangle(
        ...     data=df,
        ...     origin='origin',
        ...     development='development',
        ...     columns=['reported'],
        ...     cumulative=False,
        ... )
        >>> tr
                  12      24      36     48
        1981  5012.0  3257.0  2638.0  898.0
        1982   106.0  4179.0  1111.0    NaN
        1983  3410.0  5582.0     NaN    NaN
        1984  5655.0     NaN     NaN    NaN

        >>> tr.incr_to_cum()
                  12      24       36       48
        1981  5012.0  8269.0  10907.0  11805.0
        1982   106.0  4285.0   5396.0      NaN
        1983  3410.0  8992.0      NaN      NaN
        1984  5655.0     NaN      NaN      NaN

        By default ``incr_to_cum`` returns a new Triangle. Pass ``inplace=True`` to
        mutate the calling Triangle instead.

        >>> tr.is_cumulative
        False
        >>> _ = tr.incr_to_cum(inplace=True)
        >>> tr.is_cumulative
        True
        """
        if inplace:
            xp = self.get_array_module()
            if not self.is_cumulative:
                if self.is_pattern:
                    if hasattr(self, "is_additive"):
                        if self.is_additive:
                            values = xp.nan_to_num(self.values[..., ::-1])
                            values = num_to_value(values, 0)
                            self.values = (
                                xp.cumsum(values, -1)[..., ::-1] * self.nan_triangle
                            )
                    else:
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

        Examples
        --------
        ``cl.load_sample('ukmotor')`` is a cumulative Triangle. ``cum_to_incr``
        differences each cell against the prior development period, returning
        per-period increments.

        >>> tr = cl.load_sample('ukmotor')
        >>> tr.cum_to_incr()
                  12      24      36      48      60     72     84
        2007  3511.0  3215.0  2266.0  1712.0  1059.0  587.0  340.0
        2008  4001.0  3702.0  2278.0  1180.0   956.0  629.0    NaN
        2009  4355.0  3932.0  1946.0  1522.0  1238.0    NaN    NaN
        2010  4295.0  3455.0  2023.0  1320.0     NaN    NaN    NaN
        2011  4150.0  3747.0  2320.0     NaN     NaN    NaN    NaN
        2012  5102.0  4548.0     NaN     NaN     NaN    NaN    NaN
        2013  6283.0     NaN     NaN     NaN     NaN    NaN    NaN
        """
        if inplace:
            v = self.valuation_date
            if self.is_cumulative or self.is_cumulative is None:
                if self.is_pattern:
                    xp = self.get_array_module()
                    self.values = xp.nan_to_num(self.values)
                    values = num_to_value(self.values, 1)
                    diff = self.iloc[..., :-1] / self.iloc[..., 1:].values
                    self = concat(
                        (
                            diff,
                            self.iloc[..., -1],
                        ),
                        axis=3,
                    )
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
        """Converts triangle from a development lag triangle to a valuation
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

        Examples
        --------
        ``cl.load_sample('ukmotor')`` is a 7x7 cumulative Triangle in development
        form. Each column represents months of development from the origin year.

        >>> tr = cl.load_sample('ukmotor')
        >>> tr
                  12      24       36       48       60       72       84
        2007  3511.0  6726.0   8992.0  10704.0  11763.0  12350.0  12690.0
        2008  4001.0  7703.0   9981.0  11161.0  12117.0  12746.0      NaN
        2009  4355.0  8287.0  10233.0  11755.0  12993.0      NaN      NaN
        2010  4295.0  7750.0   9773.0  11093.0      NaN      NaN      NaN
        2011  4150.0  7897.0  10217.0      NaN      NaN      NaN      NaN
        2012  5102.0  9650.0      NaN      NaN      NaN      NaN      NaN
        2013  6283.0     NaN      NaN      NaN      NaN      NaN      NaN

        Calling ``dev_to_val`` reshapes the columns from development lags to
        valuation periods, so each column corresponds to a calendar year.

        >>> tr.dev_to_val()
                2007    2008    2009     2010     2011     2012     2013
        2007  3511.0  6726.0  8992.0  10704.0  11763.0  12350.0  12690.0
        2008     NaN  4001.0  7703.0   9981.0  11161.0  12117.0  12746.0
        2009     NaN     NaN  4355.0   8287.0  10233.0  11755.0  12993.0
        2010     NaN     NaN     NaN   4295.0   7750.0   9773.0  11093.0
        2011     NaN     NaN     NaN      NaN   4150.0   7897.0  10217.0
        2012     NaN     NaN     NaN      NaN      NaN   5102.0   9650.0
        2013     NaN     NaN     NaN      NaN      NaN      NaN   6283.0
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
        """Converts triangle from a valuation triangle to a development lag
        triangle.

        Parameters
        ----------
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Updated instance of triangle with development lags

        Examples
        --------
        ``val_to_dev`` is the inverse of ``dev_to_val``. Round-tripping a
        development triangle through valuation form and back returns the
        original layout.

        >>> tr = cl.load_sample('ukmotor')
        >>> tr.dev_to_val().val_to_dev()
                  12      24       36       48       60       72       84
        2007  3511.0  6726.0   8992.0  10704.0  11763.0  12350.0  12690.0
        2008  4001.0  7703.0   9981.0  11161.0  12117.0  12746.0      NaN
        2009  4355.0  8287.0  10233.0  11755.0  12993.0      NaN      NaN
        2010  4295.0  7750.0   9773.0  11093.0      NaN      NaN      NaN
        2011  4150.0  7897.0  10217.0      NaN      NaN      NaN      NaN
        2012  5102.0  9650.0      NaN      NaN      NaN      NaN      NaN
        2013  6283.0     NaN      NaN      NaN      NaN      NaN      NaN
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

        Examples
        --------
        Build a quarterly origin / quarterly development Triangle (OQDQ).

        >>> df = pd.DataFrame(
        ...     data={
        ...         'origin': [
        ...             '2022Q1', '2022Q1', '2022Q1', '2022Q1', '2022Q1', '2022Q1', '2022Q1', '2022Q1',
        ...             '2022Q2', '2022Q2', '2022Q2', '2022Q2', '2022Q2', '2022Q2', '2022Q2',
        ...             '2022Q3', '2022Q3', '2022Q3', '2022Q3', '2022Q3', '2022Q3',
        ...             '2022Q4', '2022Q4', '2022Q4', '2022Q4', '2022Q4',
        ...             '2023Q1', '2023Q1', '2023Q1', '2023Q1',
        ...             '2023Q2', '2023Q2', '2023Q2',
        ...             '2023Q3', '2023Q3',
        ...             '2023Q4',
        ...         ],
        ...         'development': [
        ...             '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4',
        ...             '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4',
        ...             '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4',
        ...             '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4',
        ...             '2023Q1', '2023Q2', '2023Q3', '2023Q4',
        ...             '2023Q2', '2023Q3', '2023Q4',
        ...             '2023Q3', '2023Q4',
        ...             '2023Q4',
        ...         ],
        ...         'reported': [
        ...             100, 200, 300, 400, 480, 540, 580, 600,
        ...             110, 220, 320, 420, 500, 560, 600,
        ...             120, 240, 350, 450, 520, 580,
        ...             130, 250, 370, 470, 540,
        ...             140, 260, 380, 480,
        ...             150, 270, 390,
        ...             160, 280,
        ...             170,
        ...         ],
        ...     }
        ... )
        >>> tr = cl.Triangle(
        ...     data=df,
        ...     origin='origin',
        ...     development='development',
        ...     columns=['reported'],
        ...     cumulative=True,
        ... )
        >>> tr
                   3      6      9      12     15     18     21     24
        2022Q1  100.0  200.0  300.0  400.0  480.0  540.0  580.0  600.0
        2022Q2  110.0  220.0  320.0  420.0  500.0  560.0  600.0    NaN
        2022Q3  120.0  240.0  350.0  450.0  520.0  580.0    NaN    NaN
        2022Q4  130.0  250.0  370.0  470.0  540.0    NaN    NaN    NaN
        2023Q1  140.0  260.0  380.0  480.0    NaN    NaN    NaN    NaN
        2023Q2  150.0  270.0  390.0    NaN    NaN    NaN    NaN    NaN
        2023Q3  160.0  280.0    NaN    NaN    NaN    NaN    NaN    NaN
        2023Q4  170.0    NaN    NaN    NaN    NaN    NaN    NaN    NaN

        Convert to annual origin / annual development. Origins are summed within
        each calendar year and development periods are aggregated to year-end.

        >>> tr.grain('OYDY')
                  12      24
        2022  1090.0  2320.0
        2023  1320.0     NaN

        Convert origin to annual but keep development quarterly (``OYDQ``).

        >>> tr.grain('OYDQ')
                 3      6      9       12      15      18      21      24
        2022  100.0  310.0  640.0  1090.0  1500.0  1860.0  2130.0  2320.0
        2023  140.0  410.0  810.0  1320.0     NaN     NaN     NaN     NaN
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
            freq = {"Y": "Y", "S": "2Q"}.get(ograin_new, ograin_new)

            if trailing or (obj.origin.freqstr[-3:] != "DEC" and ograin_old != "M"):
                origin_period_end = self.origin[-1].strftime("%b").upper()
            else:
                origin_period_end = "DEC"

            indices = (
                pd.Series(range(len(self.origin)), index=self.origin)
                .resample("-".join([freq, origin_period_end]))
                .indices
            )

            groups = pd.concat(
                [pd.Series([k] * len(v), index=v) for k, v in indices.items()], axis=0
            ).values

            obj = obj.groupby(groups, axis=2).sum()
            obj.origin_close = origin_period_end

            d_start = pd.Period(
                obj.valuation[0],
                freq=dgrain_old.replace("S", "2Q") + ('' if dgrain_old == "M" else obj.origin.freqstr[-4:]),
            ).to_timestamp(how="s")

            if dgrain_old == "S":
                d_start = d_start +  pd.DateOffset(months=-3)

            if len(obj.ddims) > 1 and obj.origin.to_timestamp(how="s")[0] != d_start:
                addl_ts = (
                    pd.period_range(obj.odims[0], obj.valuation[0], freq=dgrain_old.replace("S","2Q"))[
                        :-1
                    ]
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
        **kwargs,
    ):
        """Allows for the trending of a Triangle object along either a valuation
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

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     data={
        ...         'origin': [2020, 2020, 2020, 2021, 2021, 2022],
        ...         'development': [2020, 2021, 2022, 2021, 2022, 2022],
        ...         'reported': [100, 200, 300, 110, 220, 120],
        ...     }
        ... )
        >>> tr = cl.Triangle(
        ...     data=df,
        ...     origin='origin',
        ...     development='development',
        ...     columns=['reported'],
        ...     cumulative=True,
        ... )

        Apply a 10% annual trend along the origin axis. The latest origin year
        (2022) is unchanged; older origins are scaled up by ``1.10`` per year
        of distance from the latest origin.

        >>> tr.trend(0.10, axis='origin')
                 12     24     36
        2020  121.0  242.0  363.0
        2021  121.0  242.0    NaN
        2022  120.0    NaN    NaN

        Apply a 10% annual trend along the valuation axis instead. The latest
        diagonal is unchanged and earlier diagonals are scaled up.

        >>> tr.trend(0.10, axis='valuation')
                 12     24     36
        2020  121.0  220.0  300.0
        2021  121.0  220.0    NaN
        2022  120.0    NaN    NaN
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
        X = object.__new__(self.__class__)
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

        Examples
        --------
        >>> tr = cl.load_sample('raa')
        >>> dc = tr.development_correlation()
        >>> bool(dc.t_critical.iloc[0, 0])
        False

        ``t_critical`` reports whether the calculated rank correlation falls
        outside the no-correlation confidence interval. ``False`` indicates the
        development factors are not significantly correlated.
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

        Examples
        --------
        >>> tr = cl.load_sample('raa')
        >>> vc = tr.valuation_correlation()
        >>> vc.z_critical
               1982   1983   1984   1985   1986   1987   1988   1989   1990
        1981  False  False  False  False  False  False  False  False  False

        Each cell of ``z_critical`` flags whether the calendar-period z-statistic
        for that valuation falls outside the no-effect confidence interval.
        ``False`` everywhere means no calendar period shows a significant
        large-or-small bias on its diagonal.
        """
        return ValuationCorrelation(self, p_critical, total)

    def shift(self, periods=-1, axis=3):
        """Shift elements along an axis by desired number of periods.

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

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     data={
        ...         'origin': [2020, 2020, 2020, 2021, 2021, 2022],
        ...         'development': [2020, 2021, 2022, 2021, 2022, 2022],
        ...         'reported': [100, 200, 300, 110, 220, 120],
        ...     }
        ... )
        >>> tr = cl.Triangle(
        ...     data=df,
        ...     origin='origin',
        ...     development='development',
        ...     columns=['reported'],
        ...     cumulative=True,
        ... )
        >>> tr
                 12     24     36
        2020  100.0  200.0  300.0
        2021  110.0  220.0    NaN
        2022  120.0    NaN    NaN

        Shift one period along the development axis (the default). Values move
        right by one column and the leading column is filled with zeros.

        >>> tr.shift()
               12     24     36
        2020  0.0  100.0  200.0
        2021  0.0  110.0  220.0
        2022  0.0  120.0    NaN

        Shift one period along the origin axis. Each origin row's data moves
        down by one and the first origin row is zeroed out.

        >>> tr.shift(periods=-1, axis='origin')
                 12     24     36
        2020    0.0    0.0    0.0
        2021  100.0  200.0  300.0
        2022  110.0  220.0    NaN
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
        """Method to sort a Triangle along a given axis

        Parameters
        ----------
        axis : int or str
            The axis on which to sort. May be specified as an integer
            (``0``, ``1``, ``2``, ``3``) or by name (``'index'``, ``'columns'``,
            ``'origin'``, ``'development'``).

        Returns
        -------
        Triangle
            New Triangle with the requested axis sorted in ascending order.

        Examples
        --------
        Build a Triangle with two columns supplied in non-alphabetical order.

        >>> df = pd.DataFrame(
        ...     data={
        ...         'origin': [2020, 2020, 2021, 2021],
        ...         'development': [2020, 2021, 2021, 2021],
        ...         'reported': [100, 200, 110, 110],
        ...         'paid': [50, 100, 60, 60],
        ...     }
        ... )
        >>> tr = cl.Triangle(
        ...     data=df,
        ...     origin='origin',
        ...     development='development',
        ...     columns=['reported', 'paid'],
        ...     cumulative=True,
        ... )
        >>> list(tr.columns)
        ['reported', 'paid']

        Sorting on the columns axis returns a new Triangle with columns in
        alphabetical order.

        >>> list(tr.sort_axis('columns').columns)
        ['paid', 'reported']
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

    def reindex(self, columns=None, fill_value=np.nan):
        obj = self.copy()
        for column in columns:
            if column not in obj.columns:
                obj[column] = fill_value
        return obj
