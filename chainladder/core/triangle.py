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
    def key_labels(self):
        return self.triangle.key_labels
    
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
        return pd.PeriodIndex(self.triangle.origin.to_pandas(), freq=f'{self.triangle.origin_grain}')
    
    @property
    def is_val_tri(self):
        return self.triangle.is_val_tri
    
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
    def shape(self):
        return self.triangle.shape
    
    @property
    def latest_diagonal(self):
        obj = self.copy()
        obj.triangle = self.triangle.latest_diagonal
        return obj

    @property
    def link_ratio(self):
        obj = self.copy()
        obj.triangle = self.triangle.link_ratio
        return obj
    
    @property
    def origin_grain(self):
        return self.triangle.origin_grain
    
    @property
    def development_grain(self):
        return self.triangle.development_grain
    
    @property  
    def iloc(self):
        return Ilocation(self)
    
    def __repr__(self):
        if self.shape[:2] == (1, 1):
            data = self._repr_format()
            return data.to_string()
        else:
            return self._summary_frame().__repr__()

    def _summary_frame(self):
        return pd.Series(
            [
                self.valuation_date.strftime("%Y-%m"),
                "O" + self.origin_grain + "D" + self.development_grain,
                self.shape,
                self.key_labels,
                self.columns.tolist(),
            ],
            index=["Valuation:", "Grain:", "Shape:", "Index:", "Columns:"],
            name="Triangle Summary",
        ).to_frame()

    def _repr_html_(self):
        """ Jupyter/Ipython HTML representation """
        if self.shape[:2] == (1, 1):
            data = self._repr_format()
            fmt_str = self._get_format_str(data)
            
            default = (
                data.to_html(
                    max_rows=pd.options.display.max_rows,
                    max_cols=pd.options.display.max_columns,
                    float_format=fmt_str.format,
                )
                .replace("nan", "")
                .replace("NaN", "")
            )
            return default
        else:
            return self._summary_frame().to_html(
                max_rows=pd.options.display.max_rows,
                max_cols=pd.options.display.max_columns,
            )

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
            
    def __getitem__(self, key):
        obj = self.copy()
        if type(key) is str:
            key = [key]
        columns = type(key) is list and len(set(self.columns).intersection(set(key))) == len(key)
        development = type(key) is pd.Series
        origin = type(key) is np.ndarray and len(key) == len(self.origin)
        valuation = type(key) is np.ndarray and len(key) != len(self.origin)
        if columns:
            obj.triangle = self.triangle.select(key)
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
        elif type(key) is tuple or type(key) is slice or type(key) is int:
            s0, s1, s2, s3 = self.triangle[key]
            obj.triangle.data = (
                obj.triangle.data
                .filter(pl.fold(
                    acc=pl.lit(True),
                    function=lambda acc, x: acc & x,
                    exprs=s0 + s2 + s3
                ))
                .select(self.key_labels + ['__origin__', '__development__'] + s1))
            obj.triangle.columns = s1
        else:
            raise NotImplementedError()
        return obj
    
    def __setitem__(self, key, value):
        if type(value) == type(self):
            value = value.triangle
        self.triangle.__setitem__(key, value)
        
    
    def to_sparse(self):
        from chainladder.core.slice import VirtualColumns
        from chainladder.core.triangle import Triangle
        import pandas as pd
        import sparse
        import polars as pl
        df = (
            self.triangle.data.lazy().collect()
             .join(self.triangle.development.to_frame().select(pl.int_range(0, self.shape[3]).alias('d_idx'), pl.col('development').alias('__development__')), how='left', on='__development__')
             .join(self.triangle.origin.to_frame().select(pl.int_range(0, self.shape[2]).alias('o_idx'), pl.col('origin').alias('__origin__')), how='left', on='__origin__')
             .join(self.triangle.index.lazy().collect().select(pl.int_range(0, self.shape[0]).alias('i_idx'), pl.col(self.triangle.key_labels)), how='left', on=self.triangle.key_labels)
             .select(['i_idx', 'o_idx', 'd_idx'] + self.triangle.columns))
        df = pl.concat([df.select(['i_idx', pl.lit(num).alias('c_idx'), 'o_idx','d_idx', pl.col(col).alias('value').cast(pl.Float64)]) for num, col in enumerate(self.triangle.columns)])
        sm = sparse.COO(coords=df.select(pl.all().exclude('value')).to_numpy().T, data=df['value'].to_numpy(), shape=self.triangle.shape)
        triangle = Triangle()
        triangle.values=sm
        triangle.key_labels = self.triangle.key_labels
        triangle.kdims=self.triangle.index.lazy().collect().to_numpy()
        triangle.vdims=self.triangle.columns
        triangle.odims = self.triangle.origin.cast(pl.Datetime).to_numpy()
        triangle.ddims = self.triangle.development.to_numpy()
        triangle.origin_grain = self.triangle.origin_grain
        triangle.development_grain = self.triangle.development_grain
        triangle.valuation_date = pd.PeriodIndex([self.triangle.valuation_date], freq='M').to_timestamp(how='e')[0]
        triangle.is_cumulative = self.triangle.is_cumulative
        triangle.is_pattern = self.triangle.is_pattern
        triangle.origin_close = self.triangle.origin_close
        triangle.array_backend = "sparse"
        triangle.virtual_columns = VirtualColumns(triangle)
        return triangle
    
    @property
    def values(self):
        return self.triangle.data.select(self.columns)
    
    def __eq__(self, other):
        return self.triangle == other.triangle
    
    def __eq__(self, other):
        return (
            self.triangle.data.sort(
                pl.col(self.key_labels + ['__origin__', '__development__'])
                ).select(self.triangle.columns).lazy().collect() == 
            other.triangle.data.sort(
                pl.col(other.key_labels + ['__origin__', '__development__'])
                ).select(other.columns).lazy().collect()
         ).min(axis=0).min(axis=1)[0]
    
    def __len__(self):
        return len(self)

    
    def to_frame(self, *args, **kwargs):
        df = self.triangle.to_frame(*args, **kwargs).lazy().collect().to_pandas()
        shape = tuple([num for num, i in enumerate(self.shape) if i > 1])
        if shape == (0, 1):
            df = df.set_index(self.key_labels)[self.columns]
        if shape == (0, 2):
            df = df.pivot(index=self.key_labels, columns='origin', values=self.columns)
        if shape == (0, 3):
            df = df.pivot(index=self.key_labels, columns='development', values=self.columns)
        if shape == (1, 2):
            df = df.set_index('origin')[self.columns].T
        if shape == (1, 3):
            df = df.set_index('development')[self.columns].T
        if shape == (2, 3):
            df = df.set_index('origin')
        return df
    
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
    
class Ilocation:
    def __init__(self, obj):
        self.obj = obj
    
    def __getitem__(self, key):
        return self.obj.__getitem__(key)
    
class TriangleGroupBy(PlTriangleGroupBy):            
    def _agg(self, agg, axis=1, *args, **kwargs):
        axis = self.obj._get_axis(axis)
        if axis == 0:
            self.obj.data = self.groups.agg(
                getattr(pl.col(self.columns), agg)(*args, **kwargs))
        else:
            raise ValueError(f'axis {axis} is not supported')
        self.obj.columns = self.columns
        obj = Triangle()
        obj.triangle = self.obj
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
    'collect', 'lazy', 'head',
    'max', 'mean', 'median', 'min', 'product', 'quantile', 'std', 
    'sum', 'tail', 'val_to_dev', 'var', 'val_to_dev', 'dev_to_val', 'cum_to_incr', 'incr_to_cum', 'grain']
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