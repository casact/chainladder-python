import pandas as pd
import numpy as np
import polars as pl
import re
from .core import TriangleBase, PlTriangleGroupBy, vcol
import warnings
try:
    from IPython.core.display import HTML
except:
    HTML = None

class Triangle:
    """ Pandas API interface to Polars Backend """
    def __init__(self, data=None, *args, **kwargs):
        if kwargs.get('development', None):
            warnings.warn(
                """`development` argument is deprecated. Use `valuation` in the Triangle constructor.""")
            kwargs['valuation'] = kwargs['development']
        if kwargs.get('development_format', None):
            warnings.warn(
                """`development_format` argument is deprecated. Use `valuation_format` in the Triangle constructor.""")
            kwargs['valuation_format'] = kwargs['development_format']
        if data is None:
            self.triangle = None
        if type(data) == pd.DataFrame:
            self.triangle = TriangleBase(pl.DataFrame(data), *args, **kwargs)
        else:
            self.triangle = TriangleBase(data, *args, **kwargs)
        
    def copy(self):
        obj = Triangle()
        obj.triangle = TriangleBase.from_triangle(self.triangle)
        return obj
    
    @property
    def index(self):
        return self.triangle.index.lazy().collect().to_pandas()
    
    @property
    def columns(self):
        return pd.Index(self.triangle.columns, name='columns')
    
    @columns.setter
    def columns(self, value):
        self.triangle.columns = value
    
    @property
    def origin(self):
        return pd.PeriodIndex(
            self.triangle.origin.to_pandas(), 
            freq=f'{self.triangle.origin_grain}')
    
    def collect(self):
        self.triangle.data = self.triangle.data.collect()
        return self
    
    @property
    def development(self):
        if self.is_val_tri:
            formats = {"Y": "%Y", "S": "%YQ%q", "Q": "%YQ%q", "M": "%Y-%m"}
            return pd.Series(pd.to_datetime(self.triangle.valuation).to_period(
                freq=self.development_grain).strftime(
                formats[self.development_grain]),
                name='development')
        else:
            return self.triangle.development.to_pandas()
        
    @property
    def valuation_date(self):
        return (self.triangle.valuation_date + 
                pd.DateOffset(days=1) + 
                pd.DateOffset(nanoseconds=-1))
    
    @property
    def valuation(self):
        if self.is_val_tri:
            return pd.DatetimeIndex(
                self.triangle.origin.alias('__origin__').to_frame()
                .join(
                    self.triangle.valuation.alias('valuation').to_frame(), 
                    how='cross')
                .sort(['valuation'])
                .select(pl.col('valuation')).to_pandas().iloc[:, 0]
                ) + pd.DateOffset(days=1) + pd.DateOffset(nanoseconds=-1)
        else:
            return pd.DatetimeIndex(
                self.triangle.origin.alias('__origin__').to_frame()
                .join(
                    self.triangle.development.alias('__development__').to_frame(), 
                    how='cross')
                .sort(['__origin__', '__development__'])
                .select(vcol.alias('valuation')).to_pandas().iloc[:, 0]
                ) + pd.DateOffset(days=1) + pd.DateOffset(nanoseconds=-1)
    
    @property  
    def iloc(self):
        return Ilocation(self)
    
    @property  
    def loc(self):
        return Location(self)
    
    def __repr__(self):
        if self.shape[:2] == (1, 1):
            data = self._repr_format()
            return data.to_string()
        else:
            return self.triangle._summary_frame().to_pandas().set_index('').__repr__()

    def _repr_html_(self):
        """ Jupyter/Ipython HTML representation """
        if self.shape[:2] == (1, 1):
            data = self._repr_format()
            fmt_str = self._get_format_str(data)
            
            default = (
                data.to_html(
                    max_rows=pd.options.display.max_rows,
                    max_cols=pd.options.display.max_columns,
                    float_format=fmt_str.format,)
                .replace("nan", "")
                .replace("NaN", ""))
            return default
        else:
            return self.triangle._summary_frame().to_pandas().set_index('').to_html(
                max_rows=pd.options.display.max_rows,
                max_cols=pd.options.display.max_columns)

    def _get_format_str(self, data):
        if np.all(np.isnan(data)):
            return ""
        elif np.nanmean(abs(data)) < 10:
            return "{0:,.4f}"
        elif np.nanmean(abs(data)) < 1000:
            return "{0:,.2f}"
        else:
            return "{:,.0f}"

    def _repr_format(self, origin_as_datetime=False):
        out = self.triangle.wide()[:, 1:].to_numpy()
        if origin_as_datetime and not self.is_pattern:
            origin = self.origin.to_timestamp(how='s')
        else:
            origin = self.origin.copy()
        origin.name = None

        if self.origin_grain == "S" and not origin_as_datetime:
            origin_formatted = [""] * len(origin)
            for origin_index in range(len(origin)):
                origin_formatted[origin_index] = (
                    origin.astype("str")[origin_index]
                    .replace("Q1", "H1")
                    .replace("Q3", "H2")
                )
            origin = origin_formatted
        development = self.development.copy()
        development.name = None
        return pd.DataFrame(out, index=origin, columns=development)

    def heatmap(self, cmap="coolwarm", low=0, high=0, axis=0, subset=None):
        """ Color the background in a gradient according to the data in each
        column (optionally row). Requires matplotlib

        Parameters
        ----------

        cmap : str or colormap
            matplotlib colormap
        low, high : float
            compress the range by these values.
        axis : int or str
            The axis along which to apply heatmap
        subset : IndexSlice
            a valid slice for data to limit the style application to

        Returns
        -------
            Ipython.display.HTML

        """
        if self.shape[:2] == (1, 1):
            data = self._repr_format()
            fmt_str = self._get_format_str(data)

            axis = self.triangle._get_axis(axis)

            raw_rank = data.rank(axis=axis)
            shape_size = data.shape[axis]
            rank_size = data.rank(axis=axis).max(axis=axis)
            gmap = (raw_rank - 1).div(rank_size - 1, axis=not axis) * (
                shape_size - 1
            ) + 1
            gmap = gmap.replace(np.nan, (shape_size + 1) / 2)
            if pd.__version__ >= "1.3":
                default_output = (
                    data.style.format(fmt_str)
                    .background_gradient(
                        cmap=cmap,
                        low=low,
                        high=high,
                        axis=None,
                        subset=subset,
                        gmap=gmap,
                    )
                    .to_html()
                )
            else:
                default_output = (
                    data.style.format(fmt_str)
                    .background_gradient(cmap=cmap, low=low, high=high, axis=axis,)
                    .render()
                )
            output_xnan = re.sub("<td.*nan.*td>", "<td></td>", default_output)
        else:
            raise ValueError("heatmap only works with single triangles")
        if HTML:
            return HTML(output_xnan)
        elif HTML is None:
            raise ImportError("heatmap requires IPython")
    
    def __setitem__(self, key, value):
        if type(value) == type(self):
            value = value.triangle
        self.triangle.__setitem__(key, value)
    
    def __eq__(self, other):
        return self.triangle == other.triangle

    def __len__(self):
        return len(self)
    
    def to_frame(self, origin_as_datetime=True, keepdims=False,
                 implicit_axis=False, *args, **kwargs):
        """ Converts a triangle to a pandas.DataFrame.
        Parameters
        ----------
        origin_as_datetime : bool
            Whether the origin vector should be converted from PeriodIndex
            into a datetime dtype. Default is False.
        keepdims : bool
            If True, the triangle will be converted to a DataFrame with all
            dimensions intact.  The argument will force a consistent DataFrame
            format regardless of whether any dimensions are of length 1.
        implicit_axis : bool
            When keepdims is True, this denotes whether to include the implicit
            valuation axis in addition to the origin and development.
        Returns
        -------
            pandas.DataFrame representation of the Triangle.
        """
        df = self.triangle.to_frame(keepdims=keepdims, implicit_axis=implicit_axis).lazy().collect().to_pandas()
        if not origin_as_datetime:
            df['origin'] = df['origin'].map(dict(zip(self.triangle.origin, self.origin)))
        shape = tuple([num for num, i in enumerate(self.shape) if i > 1])
        if len(shape) == 2 and not keepdims:
            if shape == (0, 1):
                df = df.set_index(self.key_labels)[self.columns]
            if shape == (0, 2):
                df = df.pivot(index=self.key_labels, columns='origin', values=self.columns[0])
            if shape == (0, 3):
                df = df.pivot(index=self.key_labels, columns='development', values=self.columns[0])
            if shape == (1, 2):
                df = df.set_index('origin')[self.columns].T
            if shape == (1, 3):
                df = df.set_index('development')[self.columns].T
            if shape == (2, 3):
                df = df.set_index('origin')
            df.index.name = None
            df.columns.name = None
        if len(shape) == 1 and not keepdims:
            df = df.set_index(
                {0: self.key_labels,
                 1: self.columns,
                 2: 'origin',
                 3: 'development'}[shape[0]])
            df.index.name = None
            df.columns.name = None
        if self.triangle.shape[0] > 1 or len(shape) > 2 or keepdims:
            df = df.set_index(self.key_labels)
        return df
    
    @property
    def T(self):
        return self.to_frame(origin_as_datetime=False).T
    
    def groupby(self, by, axis=0, *args, **kwargs):
        return TriangleGroupBy(self.triangle, by, axis)

    def __array__(self):
        return self.triangle.data.select(self.columns)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        obj = self.copy()
        if method == "__call__":
            inputs = [pl.col(self.columns) if hasattr(i, "columns") else i for i in inputs]
            obj.triangle.data = self.triangle.data.select(
                pl.col(self.key_labels + ['__origin__', '__development__']), 
                ufunc(*inputs, **kwargs))
            obj.triangle.data.select(pl.all().exclude(self.columns), )
            return obj
        else:
            raise NotImplementedError()
        
    def minimum(self, other):
        return np.minimum(self, other)
    
    def maximum(self, other):
        return np.maximum(self, other)
    
    def log(self):
        return np.log(self)
    
    def sqrt(self):
        return np.sqrt(self)
    
    def exp(self):
        return np.exp(self)
    
    def __getitem__(self, key):
        obj = self.copy()
        development = type(key) is pd.Series
        origin = type(key) is np.ndarray and len(key) == len(self.origin)
        valuation = type(key) is np.ndarray and len(key) != len(self.origin)
        if not (origin or development or valuation):
            out = self.triangle[key]
            if type(out) is TriangleBase:
                obj.triangle = out
                return obj
            else:
                return out
        elif development:
            if self.is_val_tri:
                formats = {"Y": "%Y", "S": "%YQ%q", "Q": "%YQ%q", "M": "%Y-%m"}
                ddims = pd.to_datetime(
                    self.development, 
                    format=formats[self.development_grain]
                ).dt.to_period(self.development_grain).dt.to_timestamp(how='e')
                key = self.triangle.valuation.is_in(ddims[key].dt.date)
                obj.triangle = obj.triangle[key]
            else:
                key = self.triangle.development.is_in(self.development[key])
                obj.triangle = obj.triangle[key]
        elif origin:
            key = self.triangle.origin.is_in(self.origin[key].to_timestamp(how='s'))
            obj.triangle = obj.triangle[key]
        elif valuation:
            key = self.triangle.valuation.is_in(self.valuation[key].unique().date.tolist())
            obj.triangle = obj.triangle[key]
        return obj
    
    
    def val_to_dev(self, *args, **kwargs):
        obj = self.copy()
        obj.triangle = self.triangle.to_development(*args, **kwargs)
        return obj

    def dev_to_val(self, *args, **kwargs):
        obj = self.copy()
        obj.triangle = self.triangle.to_valuation(*args, **kwargs)
        return obj

    def cum_to_incr(self, *args, **kwargs):
        obj = self.copy()
        obj.triangle = self.triangle.to_incremental(*args, **kwargs)
        return obj

    def incr_to_cum(self, *args, **kwargs):
        obj = self.copy()
        obj.triangle = self.triangle.to_cumulative(*args, **kwargs)
        return obj

    def grain(self, *args, **kwargs):
        obj = self.copy()
        obj.triangle = self.triangle.to_grain(*args, **kwargs)
        return obj
    
    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)
    
    def append(self, other):
        from chainladder.utils.utility_functions import concat
        obj = self.copy()
        obj.triangle = concat((self.triangle, other.triangle), axis=0)
        return obj

class Ilocation:
    def __init__(self, obj):
        self.obj = obj
    
    def __getitem__(self, key):
        obj = self.obj.copy()
        obj.triangle = obj.triangle[key]
        return obj


class Location(Ilocation):        
    def _contig_slice(self, arr):
        """ Try to make a contiguous slicer from an array of indices """
        if type(arr) is slice:
            return arr
        if type(arr) in [int, np.int64, np.int32]:
            arr = [arr]
        if len(arr) == 1:
            return slice(arr[0], arr[0] + 1)
        diff = np.diff(arr)
        if len(diff) == 0:
            raise ValueError("Slice returns empty Triangle")
        if max(diff) == min(diff):
            step = max(diff)
        else:
            return arr
        step = None if step == 1 else step
        min_arr = None if min(arr) == 0 else min(arr)
        max_arr = max(arr) + 1
        if step and step < 0:
            min_arr, max_arr = max_arr - 1, min_arr - 1 if min_arr else min_arr
        return slice(min_arr, max_arr, step)
    
    def __getitem__(self, key):
        key = self.obj.triangle._normalize_slice(key)
        obj = self.obj.copy()
        full_slice = lambda x: x == slice(None, None, None) or x == slice(None, None, -1)
        if not full_slice(key[0]):
            idx_slice = obj.index.reset_index().set_index(obj.key_labels).loc[key[0]]
        else:
            idx_slice = key[0]
        key = (
            key[0] if full_slice(key[0]) else self._contig_slice(
                idx_slice.values.flatten().tolist()),
            key[1] if full_slice(key[1]) else self._contig_slice(
                pd.Series(obj.columns).reset_index().set_index('columns')
                .loc[key[1]].values.flatten().tolist()),
            key[2] if full_slice(key[2]) else self._contig_slice(
                pd.Series(obj.origin).reset_index().set_index('origin')
                .loc[key[2]].values.flatten().tolist()),
            key[3] if full_slice(key[3]) else self._contig_slice(
                obj.development.reset_index().set_index('development')
                .loc[key[3]].values.flatten().tolist()))
        obj.triangle = obj.triangle[key]
        if len(obj.key_labels) > 1 and not full_slice(key[0]):
            obj.triangle.data = obj.triangle.data.drop(set(obj.key_labels)-set(idx_slice.index.names))
            obj.triangle._properties.pop('index', None)
        return obj
    
    
class TriangleGroupBy(PlTriangleGroupBy):
    def _agg(self, agg, *args, **kwargs):
        obj = Triangle()
        obj.triangle = super()._agg(agg, *args, **kwargs)
        return obj


def add_tri_passthru(cls, k):
    """Pass Through of TriangleBase functionality"""

    def tri_passthru(self, *args, **kwargs):
        obj = self.copy()
        obj.triangle = getattr(TriangleBase, k)(obj.triangle, *args, **kwargs)
        
        if (k in ('max', 'mean', 'median', 'min', 'product', 'quantile', 'std', 'sum') and 
            obj.triangle.shape == (1, 1, 1, 1)):
            return obj.triangle.data[obj.triangle.columns][0, 0]
        return obj
    
    def set_method(cls, func, k):
        """Assigns methods to a class"""
        func.__name__ = k
        setattr(cls, func.__name__, func)

    set_method(cls, tri_passthru, k)

passthru = [
    '__abs__', '__neg__', '__pos__',  '__pow__', '__round__', 
    'collect', 'lazy', 'head', '_get_axis', 'filter', 'select',
    'max', 'mean', 'median', 'min', 'product', 'quantile', 'std', 
    'sum', 'tail', 'var']

for item in passthru:
    add_tri_passthru(Triangle, item)


def add_arithmetic_passthru(cls, k):
    """Pass Through of TriangleBase functionality"""

    def tri_passthru(self, other, *args, **kwargs):
        obj = self.copy()
        if type(other) == type(self):
            obj.triangle = getattr(self.triangle, k)(other.triangle , *args, **kwargs)
        else:
            obj.triangle = getattr(self.triangle, k)(other)
        return obj
    
    def set_method(cls, func, k):
        """Assigns methods to a class"""
        func.__name__ = k
        setattr(cls, func.__name__, func)

    set_method(cls, tri_passthru, k)
    
passthru = [
    '__add__', '__ge__', '__gt__', 
     '__le__', '__lt__', '__mul__', '__ne__', 
    '__radd__', '__rmul__', '__rsub__', 
    '__rtruediv__', '__sub__', '__truediv__'] 
for item in passthru:
    add_arithmetic_passthru(Triangle, item)


def add_property_passthru(cls, k):
    """Pass Through of TriangleBase functionality"""

    def tri_passthru(self):
        prop = getattr(self.triangle, k)
        if type(prop) is TriangleBase:
            obj = self.copy()
            obj.triangle = prop
            return obj 
        else:
            return prop

    def set_method(cls, func, k):
        """Assigns methods to a class"""
        func.__name__ = k
        setattr(cls, func.__name__, property(func))

    set_method(cls, tri_passthru, k)

property_passthru = [
    'key_labels', 'is_val_tri', 'origin_close', 'shape', 
    'latest_diagonal', 'link_ratio', 'origin_grain', 
    'development_grain', 'values']
for item in property_passthru:
    add_property_passthru(Triangle, item)